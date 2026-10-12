"""Evaluate a saved checkpoint on frozen RSNA test and Kermany external data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config
from .data import LungCropCache, discover_kermany, load_manifest
from .evaluation import evaluate_datasets
from .models.classifier import build_model
from .preflight import check_data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="path to the classifier .pt checkpoint")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="YAML config path (default: configs/default.yaml)")
    parser.add_argument("--head", choices=("fastkan", "linear", "mlp"), default=None,
                        help="head for legacy checkpoints without metadata (otherwise inferred)")
    parser.add_argument("--no-crop", action="store_true",
                        help="evaluate a legacy no-crop checkpoint without metadata (otherwise inferred)")
    parser.add_argument("--output-dir", default=None,
                        help="results folder (default: evaluation/ beside the checkpoint)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    metadata = checkpoint.get("metadata", {})
    model_config = dict(config["model"])
    head = args.head or metadata.get("head") or model_config.pop("head")
    if metadata.get("head") and args.head and args.head != metadata["head"]:
        raise ValueError("--head conflicts with the head stored in this checkpoint")
    model_config.pop("head", None)
    model = build_model(head=head, **model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    split_data = load_manifest(paths["manifest"], paths.get("rsna_root"))
    datasets = {"RSNA_test": split_data["test"]}
    if paths.get("kermany_root"):
        datasets["Kermany"] = discover_kermany(paths["kermany_root"])

    crop_cache = None
    checkpoint_crop = metadata.get("use_lung_crop")
    if checkpoint_crop is True and args.no_crop:
        raise ValueError("--no-crop conflicts with this cropped checkpoint")
    use_crop = checkpoint_crop if checkpoint_crop is not None else not args.no_crop
    check_data(paths, use_crop=use_crop)
    if use_crop:
        crop_cache = LungCropCache(paths["segmentation_checkpoint"], paths["crop_cache"], device)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / "evaluation"
    metrics = evaluate_datasets(
        model,
        datasets,
        output_dir,
        device,
        crop_cache=crop_cache,
        image_size=config["data"]["image_size"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
