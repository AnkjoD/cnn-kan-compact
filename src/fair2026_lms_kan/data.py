"""Dataset, preprocessing, and MobileUNet lung-crop utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from .models.mobileunet import build_mobileunet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class UnifiedResize:
    def __init__(self, size: int = 256):
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.resize((self.size, self.size), Image.Resampling.BICUBIC)


class CLAHE:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)):
        self.operator = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, image: Image.Image) -> Image.Image:
        grayscale = np.asarray(image.convert("L"))
        return Image.fromarray(self.operator.apply(grayscale)).convert("RGB")


def build_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            UnifiedResize(256),
            CLAHE(),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, translate=(0.05, 0.05), shear=8),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 1.5))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            UnifiedResize(256),
            CLAHE(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def load_manifest(manifest_path: str | Path, data_root: str | Path | None = None) -> dict[str, list[tuple[str, int]]]:
    """Read the frozen RSNA manifest.

    Required columns are ``path``, ``label``, and ``split``. Relative paths are
    resolved under ``data_root``. The function never creates a new split.
    """
    frame = pd.read_csv(manifest_path)
    required = {"path", "label", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    root = Path(data_root).expanduser().resolve() if data_root else None
    split_data: dict[str, list[tuple[str, int]]] = {}
    for split_name in ("train", "val", "test"):
        rows = frame.loc[frame["split"] == split_name]
        samples = []
        for raw_path, label in zip(rows["path"].astype(str), rows["label"].astype(int)):
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                if root is None:
                    raise ValueError("Relative manifest paths require --data-root")
                path = root / path
            samples.append((str(path), int(label)))
        if not samples:
            raise ValueError(f"Manifest split '{split_name}' is empty")
        split_data[split_name] = samples
    return split_data


def discover_kermany(root: str | Path) -> list[tuple[str, int]]:
    """Return all Kermany train/val/test images for zero-shot external testing."""
    root = Path(root).expanduser().resolve()
    candidates = [root, *sorted(root.rglob("chest_xray"))]
    base = next((candidate for candidate in candidates
                 if any((candidate / split / label).is_dir()
                        for split in ("train", "val", "test")
                        for label in ("NORMAL", "PNEUMONIA"))), root)
    samples: list[tuple[str, int]] = []
    for split_name in ("train", "val", "test"):
        for folder, label in (("NORMAL", 0), ("PNEUMONIA", 1)):
            directory = base / split_name / folder
            if directory.is_dir():
                samples.extend((str(path), label) for path in sorted(directory.iterdir())
                               if path.is_file() and path.suffix.lower() in {".jpeg", ".jpg", ".png", ".bmp"})
    if not samples:
        raise FileNotFoundError(f"No Kermany images found below {root}")
    return samples


class LungCropCache:
    def __init__(
        self,
        checkpoint: str | Path,
        cache_dir: str | Path,
        device: torch.device,
        segmentation_size: int = 256,
        threshold: float = 0.5,
        padding_fraction: float = 0.04,
        minimum_mask_fraction: float = 0.05,
    ):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.segmentation_size = segmentation_size
        self.threshold = threshold
        self.padding_fraction = padding_fraction
        self.minimum_mask_fraction = minimum_mask_fraction
        checkpoint_object = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model = build_mobileunet().to(device)
        self.model.load_state_dict(checkpoint_object.get("model_state", checkpoint_object))
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        )

    def path_for(self, image_path: str | Path) -> Path:
        key = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.png"

    def _crop(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        if float(mask.mean()) < self.minimum_mask_fraction:
            return image
        rows, columns = np.where(mask)
        height, width = image.height, image.width
        y0, y1 = rows.min() / mask.shape[0], rows.max() / mask.shape[0]
        x0, x1 = columns.min() / mask.shape[1], columns.max() / mask.shape[1]
        pad_x = (x1 - x0) * self.padding_fraction
        pad_y = (y1 - y0) * self.padding_fraction
        box = (
            int(max(0.0, x0 - pad_x) * width),
            int(max(0.0, y0 - pad_y) * height),
            int(min(1.0, x1 + pad_x) * width),
            int(min(1.0, y1 + pad_y) * height),
        )
        return image.crop(box)

    @torch.inference_mode()
    def ensure(self, samples: Iterable[tuple[str, int]], batch_size: int = 48) -> None:
        pending = [(path, self.path_for(path)) for path, _ in samples if not self.path_for(path).exists()]
        for start in tqdm(range(0, len(pending), batch_size), desc="lung-crop cache"):
            batch = pending[start : start + batch_size]
            originals: list[tuple[Image.Image, Path]] = []
            tensors = []
            for source_path, destination in batch:
                try:
                    image = Image.open(source_path).convert("RGB")
                except (UnidentifiedImageError, OSError):
                    continue
                originals.append((image, destination))
                resized = image.resize(
                    (self.segmentation_size, self.segmentation_size),
                    Image.Resampling.BICUBIC,
                )
                tensors.append(self.transform(resized))
            if not tensors:
                continue
            inputs = torch.stack(tensors).to(self.device)
            masks = (torch.sigmoid(self.model(inputs))[:, 0] > self.threshold).cpu().numpy()
            for (image, destination), mask in zip(originals, masks):
                temporary = destination.with_suffix(".tmp.png")
                self._crop(image, mask).save(temporary, format="PNG")
                temporary.replace(destination)


class XRayDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        transform,
        crop_cache: LungCropCache | None = None,
        return_path: bool = False,
    ):
        self.samples = samples
        self.transform = transform
        self.crop_cache = crop_cache
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        source = self.crop_cache.path_for(path) if self.crop_cache else Path(path)
        try:
            image = Image.open(source).convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise RuntimeError(f"Could not read image: {source}") from error
        tensor = self.transform(image)
        if self.return_path:
            return tensor, int(label), path
        return tensor, int(label)


def build_loaders(
    split_data: dict[str, list[tuple[str, int]]],
    batch_size: int,
    seed: int,
    image_size: int = 224,
    crop_cache: LungCropCache | None = None,
    num_workers: int = 2,
) -> dict[str, DataLoader]:
    if crop_cache:
        all_samples = split_data["train"] + split_data["val"] + split_data["test"]
        crop_cache.ensure(all_samples)
    train_transform, eval_transform = build_transforms(image_size)
    generator = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(
            XRayDataset(split_data["train"], train_transform, crop_cache),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            XRayDataset(split_data["val"], eval_transform, crop_cache),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            XRayDataset(split_data["test"], eval_transform, crop_cache, return_path=True),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
