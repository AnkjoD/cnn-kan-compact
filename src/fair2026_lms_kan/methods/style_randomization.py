"""Single-source style randomization experiment reported as a negative result."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm.auto import tqdm

from ..metrics import binary_metrics


class ImageStyleRandomization(nn.Module):
    def __init__(self, minimum: float = -2.2, maximum: float = 2.2, probability: float = 0.5):
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum
        self.probability = probability

    def forward(self, x):
        if not self.training or torch.rand(1, device=x.device).item() > self.probability:
            return x
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-6
        sampled_mean = torch.empty_like(mean).uniform_(self.minimum, self.maximum)
        sampled_std = torch.empty_like(std).uniform_(0.1, (self.maximum - self.minimum) / 4)
        return ((x - mean) / std) * sampled_std + sampled_mean


class StyleNetwork(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        middle = max(channels // 4, 8)
        self.network = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Conv2d(channels, middle, 3, padding=1),
            nn.Conv2d(middle, channels, 3, padding=1),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        return self.network(x)


class FeatureStyleRandomization(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = StyleNetwork(channels)
        self.beta = StyleNetwork(channels)

    def forward(self, content, style_reference):
        mean = content.mean(dim=(2, 3), keepdim=True)
        std = content.std(dim=(2, 3), keepdim=True) + 1e-6
        return self.gamma(style_reference) * ((content - mean) / std) + self.beta(style_reference)


class SRMWrapper(nn.Module):
    def __init__(self, model, image_probability: float = 0.5):
        super().__init__()
        self.model = model
        channels = model.backbone.out_channels[1]
        self.image_srm = ImageStyleRandomization(probability=image_probability)
        self.feature_srm = FeatureStyleRandomization(channels)

    def _remaining_stages(self, stage_one, stage_zero):
        features = [stage_zero, stage_one]
        x = stage_one
        for index in (2, 3):
            x = self.model.backbone.stages[index](self.model.backbone.downsample_layers[index](x))
            features.append(x)
        return features

    def forward_train(self, x):
        backbone = self.model.backbone
        styled_input = self.image_srm(x)
        original0 = backbone.stages[0](backbone.downsample_layers[0](backbone.stem(x)))
        original1 = backbone.stages[1](backbone.downsample_layers[1](original0))
        styled0 = backbone.stages[0](backbone.downsample_layers[0](backbone.stem(styled_input)))
        styled1 = backbone.stages[1](backbone.downsample_layers[1](styled0))
        permutation = torch.randperm(styled1.size(0), device=styled1.device)
        styled1 = self.feature_srm(styled1, styled1[permutation])
        original_features = self.model.neck(self._remaining_stages(original1, original0))
        styled_features = self.model.neck(self._remaining_stages(styled1, styled0))
        return (
            self.model.head(original_features),
            self.model.head(styled_features),
            original_features,
            styled_features,
        )

    def forward(self, x):
        return self.model(x)


def run_epoch_srm(wrapper, loader, criterion, device, optimizer=None, scaler=None, consistency_weight=0.3, description="SRM"):
    is_training = optimizer is not None
    wrapper.train(is_training)
    labels, probabilities = [], []
    total_loss = 0.0
    for inputs, targets in tqdm(loader, desc=description, leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.set_grad_enabled(is_training), autocast(device_type=device.type, enabled=device.type == "cuda"):
            if is_training:
                original_logits, styled_logits, original_features, styled_features = wrapper.forward_train(inputs)
                classification_loss = criterion(styled_logits, targets)
                content_loss = F.mse_loss(styled_features, original_features)
                forward_kl = F.kl_div(
                    F.log_softmax(styled_logits, dim=1),
                    F.softmax(original_logits.detach(), dim=1),
                    reduction="batchmean",
                )
                reverse_kl = F.kl_div(
                    F.log_softmax(original_logits, dim=1),
                    F.softmax(styled_logits.detach(), dim=1),
                    reduction="batchmean",
                )
                loss = classification_loss + consistency_weight * (content_loss + 0.5 * (forward_kl + reverse_kl))
                logits = styled_logits
            else:
                logits = wrapper(inputs)
                loss = criterion(logits, targets)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += float(loss.item()) * targets.size(0)
        labels.extend(targets.cpu().numpy())
        probabilities.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
    metrics = binary_metrics(labels, probabilities)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics

