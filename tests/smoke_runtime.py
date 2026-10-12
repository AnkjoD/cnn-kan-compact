"""Fast runtime check on local computers, servers, or Colab.

Run after ``pip install -e .``. Supply checkpoints from your own Drive to
verify that state_dict keys in this public package match the archived models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from fair2026_lms_kan.models.classifier import build_model
from fair2026_lms_kan.models.mobileunet import build_mobileunet


def check_classifier(head: str, device: torch.device, checkpoint_path: str | None) -> None:
    model = build_model(
        head=head,
        backbone_size="pico",
        fusion_dim=256,
        hidden_dim=64,
        grid_size=4,
        dropout=0.057746090438249055,
        num_classes=2,
        drop_path_rate=0.1,
    ).to(device)
    if checkpoint_path:
        # Only load checkpoints that you created or otherwise trust.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state, strict=True)
        print(f"[OK] {head}: original checkpoint keys matched ({len(state)} tensors)")

    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224, device=device))
    assert output.shape == (1, 2) and torch.isfinite(output).all()
    print(f"[OK] {head}: forward output shape {tuple(output.shape)}")

    if not checkpoint_path:
        with TemporaryDirectory() as temporary:
            filename = Path(temporary) / "roundtrip.pt"
            torch.save({"model_state": model.state_dict()}, filename)
            recovered = torch.load(filename, map_location="cpu", weights_only=True)
            duplicate = build_model(
                head=head, backbone_size="pico", grid_size=4,
                dropout=0.057746090438249055,
            )
            duplicate.load_state_dict(recovered["model_state"], strict=True)
        print(f"[OK] {head}: package checkpoint roundtrip")


def check_segmentation(device: torch.device, checkpoint_path: str | None) -> None:
    model = build_mobileunet().to(device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state, strict=True)
        print(f"[OK] MobileUNet: original checkpoint keys matched ({len(state)} tensors)")
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 256, 256, device=device))
    assert output.shape == (1, 1, 256, 256) and torch.isfinite(output).all()
    print(f"[OK] MobileUNet: forward output shape {tuple(output.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", choices=("fastkan", "linear", "mlp"), default="fastkan",
                        help="head type of --classifier-checkpoint (default: fastkan)")
    parser.add_argument("--classifier-checkpoint", default=None,
                        help="trusted classifier .pt checkpoint to strict-load (optional)")
    parser.add_argument("--segmentation-checkpoint", default=None,
                        help="trusted MobileUNet .pt checkpoint to strict-load (optional)")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch", torch.__version__, "device", device)
    heads = (args.head,) if args.classifier_checkpoint else ("fastkan", "linear", "mlp")
    for head in heads:
        check_classifier(head, device, args.classifier_checkpoint)
    check_segmentation(device, args.segmentation_checkpoint)
    print("[PASS] Smoke test finished")


if __name__ == "__main__":
    main()
