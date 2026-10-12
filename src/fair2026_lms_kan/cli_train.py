"""Train a FastKAN, linear, or MLP head under the frozen RSNA split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config
from .data import LungCropCache, build_loaders, discover_kermany, load_manifest
from .evaluation import evaluate_datasets
from .models.classifier import build_model
from .preflight import check_data
from .training import fit, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml",
                        help="YAML config path (default: configs/default.yaml; use local.yaml for real data)")
    parser.add_argument("--head", choices=("fastkan", "linear", "mlp"), default=None,
                        help="classifier head (default: model.head in config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed (default: training.seed in config)")
    parser.add_argument("--no-crop", action="store_true",
                        help="train on full images without MobileUNet crop; default is lung crop")
    parser.add_argument("--resume", action="store_true",
                        help="continue from last completed epoch in the same run directory")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    model_config = dict(config["model"])
    training_config = config["training"]
    default_head = model_config.pop("head")
    head = args.head or default_head
    seed = args.seed if args.seed is not None else training_config["seed"]
    preflight = check_data(paths, use_crop=not args.no_crop)
    # Seed before construction: model parameters and DataLoader shuffling both
    # depend on the RNG state, not just the training loop.
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = Path(paths["manifest"])
    split_data = load_manifest(manifest_path, paths.get("rsna_root"))
    manifest_hash = preflight["manifest_sha256"]
    crop_cache = None
    if not args.no_crop:
        crop_cache = LungCropCache(paths["segmentation_checkpoint"], paths["crop_cache"], device)

    loaders = build_loaders(
        split_data,
        batch_size=training_config["batch_size"],
        seed=seed,
        image_size=config["data"]["image_size"],
        crop_cache=crop_cache,
        num_workers=training_config["num_workers"],
    )
    model = build_model(head=head, **model_config)
    run_name = f"{head}_{'no_crop' if args.no_crop else 'crop'}_seed{seed}"
    run_dir = Path(paths["output_dir"]) / run_name
    checkpoint_path = run_dir / "checkpoint.pt"
    metadata = {
        "run_name": run_name,
        "head": head,
        "seed": seed,
        "manifest_sha256": manifest_hash,
        "use_lung_crop": not args.no_crop,
        "model": model_config,
    }
    model, _ = fit(
        model,
        loaders,
        checkpoint_path,
        device,
        seed,
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        label_smoothing=training_config["label_smoothing"],
        max_epochs=training_config["max_epochs"],
        patience=training_config["patience"],
        metadata=metadata,
        resume=args.resume,
    )

    datasets = {"RSNA_test": split_data["test"]}
    if paths.get("kermany_root"):
        datasets["Kermany"] = discover_kermany(paths["kermany_root"])
    metrics = evaluate_datasets(
        model,
        datasets,
        run_dir,
        device,
        crop_cache=crop_cache,
        image_size=config["data"]["image_size"],
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
