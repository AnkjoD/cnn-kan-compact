"""Checkpoint evaluation with per-image predictions and machine-readable metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import XRayDataset, build_transforms
from .metrics import binary_metrics


@torch.inference_mode()
def predict(model, loader, device, description: str):
    model.eval()
    labels, probabilities, paths = [], [], []
    for inputs, targets, batch_paths in tqdm(loader, desc=description, leave=False):
        logits = model(inputs.to(device, non_blocking=True))
        labels.extend(targets.numpy())
        probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        paths.extend(batch_paths)
    return np.asarray(labels), np.asarray(probabilities), paths


def evaluate_datasets(
    model,
    datasets: dict[str, list[tuple[str, int]]],
    output_dir: str | Path,
    device,
    crop_cache=None,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
) -> dict[str, dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, eval_transform = build_transforms(image_size)
    results = {}
    prediction_frames = []

    for dataset_name, samples in datasets.items():
        if crop_cache:
            crop_cache.ensure(samples)
        loader = DataLoader(
            XRayDataset(samples, eval_transform, crop_cache, return_path=True),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        labels, probabilities, paths = predict(model, loader, device, dataset_name)
        results[dataset_name] = binary_metrics(labels, probabilities)
        sample_ids = [hashlib.sha256(path.encode("utf-8")).hexdigest()[:16] for path in paths]
        prediction_frames.append(
            pd.DataFrame(
                {"dataset": dataset_name, "sample_id": sample_ids, "label": labels, "probability": probabilities}
            )
        )

    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    return results
