"""Create an exploratory Grad-CAM overlay for a single image and checkpoint."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .config import load_config
from .data import LungCropCache, build_transforms
from .models.classifier import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('image')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--head', choices=['fastkan','linear','mlp'], help='for legacy checkpoints')
    parser.add_argument('--no-crop', action='store_true', help='for legacy checkpoints')
    parser.add_argument('--output', default='gradcam.png')
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta = ckpt.get('metadata', {})
    if args.head and meta.get('head') and args.head != meta['head']:
        parser.error('--head conflicts with checkpoint metadata')
    if args.no_crop and meta.get('use_lung_crop') is True:
        parser.error('--no-crop conflicts with checkpoint metadata')
    head = args.head or meta.get('head') or cfg['model']['head']
    model_cfg = dict(cfg['model']); model_cfg.pop('head', None)
    model = build_model(head=head, **model_cfg).to(device)
    model.load_state_dict(ckpt.get('model_state', ckpt))
    model.eval()
    use_crop = meta.get('use_lung_crop', not args.no_crop)
    image_path = Path(args.image).expanduser().resolve()
    if use_crop:
        crop = LungCropCache(cfg['paths']['segmentation_checkpoint'], cfg['paths']['crop_cache'], device)
        crop.ensure([(str(image_path), 0)])
        image_path = crop.path_for(str(image_path))
    image = Image.open(image_path).convert('RGB')
    _, transform = build_transforms(cfg['data']['image_size'])
    x = transform(image).unsqueeze(0).to(device)
    cache = {}
    def forward_hook(_module, _inputs, output):
        cache['activation'] = output
        output.register_hook(lambda gradient: cache.__setitem__('gradient', gradient))
    handle = model.backbone.stages[-1].register_forward_hook(forward_hook)
    try:
        with torch.enable_grad():
            logits = model(x)
            probability = float(torch.softmax(logits, dim=1)[0,1].detach().cpu())
            model.zero_grad(set_to_none=True)
            logits[0,1].backward()
        feature = cache['activation'][0].detach()
        weights = cache['gradient'][0].mean(dim=(1,2), keepdim=True)
        heat = torch.relu((weights * feature).sum(dim=0)).cpu().numpy()
    finally:
        handle.remove()
    heat = heat / (heat.max() + 1e-8)
    rgb = np.asarray(image.resize((512,512), Image.Resampling.BICUBIC))
    heat = cv2.resize(heat, (512,512))
    colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*heat), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    result = Image.fromarray(np.uint8(0.6*rgb + 0.4*colored))
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    result.save(path)
    print(f'{path} | positive-class probability={probability:.4f} | exploratory map; no causal interpretation')

if __name__ == '__main__':
    main()
