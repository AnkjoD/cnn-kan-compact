"""Training and checkpoint selection for the controlled classifier experiments."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from .metrics import binary_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, description="epoch"):
    is_training = optimizer is not None
    model.train(is_training)
    labels, probabilities = [], []
    total_loss = 0.0
    amp_enabled = device.type == "cuda"

    for inputs, targets in tqdm(loader, desc=description, leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(is_training), autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(inputs)
            loss = criterion(logits, targets)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * targets.size(0)
        labels.extend(targets.detach().cpu().numpy())
        probabilities.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())

    metrics = binary_metrics(labels, probabilities)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def fit(
    model,
    loaders,
    output_path: str | Path,
    device,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    max_epochs: int = 15,
    patience: int = 5,
    metadata: dict | None = None,
    resume: bool = False,
):
    """Select the checkpoint with the best validation balanced accuracy."""
    seed_everything(seed)
    model = model.to(device)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    history = []
    best_score = -1.0
    bad_epochs = 0
    best_state = None
    last_path = output_path.with_name("last.pt")
    start_epoch = 0
    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"No resume state: {last_path}")
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved["metadata"] != (metadata or {}):
            raise ValueError("Resume metadata differs (head, seed, split or crop)")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        history = saved["history"]
        best_score = saved["best_score"]
        bad_epochs = saved["bad_epochs"]
        start_epoch = saved["epoch"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if saved["loader_rng"] is not None:
            loaders["train"].generator.set_state(saved["loader_rng"])
        if output_path.exists():
            best_state = torch.load(output_path, map_location=device, weights_only=False)["model_state"]
        if bad_epochs >= patience:
            start_epoch = max_epochs
    elif output_path.exists() or last_path.exists():
        raise FileExistsError(f"Run already exists at {output_path.parent}; use --resume or choose another run")

    for epoch in range(start_epoch, max_epochs):
        train_metrics = run_epoch(
            model, loaders["train"], criterion, device, optimizer, scaler, f"train {epoch + 1}/{max_epochs}"
        )
        validation_metrics = run_epoch(
            model, loaders["val"], criterion, device, description=f"val {epoch + 1}/{max_epochs}"
        )
        score = validation_metrics["balanced_accuracy"]
        history.append(
            {"epoch": epoch + 1, "train": train_metrics, "val": validation_metrics, "selection_score": score}
        )
        if score > best_score:
            best_score = score
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state": best_state,
                    "epoch": epoch + 1,
                    "seed": seed,
                    "selection_score": score,
                    "val_metrics": validation_metrics,
                    "history": history,
                    "metadata": metadata or {},
                },
                output_path,
            )
        else:
            bad_epochs += 1
        # Save after every complete epoch. Re-run an interrupted epoch after a
        # runtime failure; no partial epoch is treated as finished.
        temporary = last_path.with_suffix(".tmp")
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(), "metadata": metadata or {},
                    "history": history, "best_score": best_score, "bad_epochs": bad_epochs,
                    "epoch": epoch + 1, "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "loader_rng": loaders["train"].generator.get_state() if loaders["train"].generator else None}, temporary)
        temporary.replace(last_path)
        if bad_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    history_path = output_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, history
