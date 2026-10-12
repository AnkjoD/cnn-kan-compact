# Reproducibility protocol

## Primary experiments

- Frozen patient-level RSNA train, validation, and test manifest.
- Seeds: 42, 123, and 2026 for cropped FastKAN, Linear, MLP, and
  no-crop FastKAN.
- Identical backbone, MobileFusion neck, preprocessing, split, optimization,
  and checkpoint selection for the FastKAN, linear, and MLP head comparison.
- Checkpoint selection uses validation balanced accuracy.
- External Kermany evaluation uses a fixed probability threshold of 0.5 and
  reports ROC-AUC, recall, specificity, and balanced accuracy.

## Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp configs/default.yaml configs/local.yaml
```

Edit only `configs/local.yaml`, then run:

```bash
python train.py --config configs/local.yaml --head fastkan --seed 42
python train.py --config configs/local.yaml --head fastkan --seed 123
python train.py --config configs/local.yaml --head fastkan --seed 2026
python train.py --config configs/local.yaml --head linear --seed 42
python train.py --config configs/local.yaml --head linear --seed 123
python train.py --config configs/local.yaml --head linear --seed 2026
python train.py --config configs/local.yaml --head mlp --seed 42
python train.py --config configs/local.yaml --head mlp --seed 123
python train.py --config configs/local.yaml --head mlp --seed 2026
python train.py --config configs/local.yaml --head fastkan --seed 42 --no-crop
python train.py --config configs/local.yaml --head fastkan --seed 123 --no-crop
python train.py --config configs/local.yaml --head fastkan --seed 2026 --no-crop
```

Evaluate an existing checkpoint:

```bash
python evaluate.py runs/fastkan_crop_seed42/checkpoint.pt \
  --config configs/local.yaml
```

## Reported but separated analyses

The original project also explored source probes, Grad-CAM, an older crop/no-crop,
rotation self-supervision, knowledge distillation, and domain-adversarial
adaptation. Those experiments had different completion states or protocols.
They should remain separate from the primary controlled comparison unless their
checkpoint, manifest, and metric provenance are verified.

The three-seed extension is based on completed Colab logs. The archival
prediction ZIP is not included here, so the tabulated extension metrics remain
marked `notebook_log_pending_archive_check`. The CLI can resume from the last completed epoch with `--resume`; see
[additional workflows](ADDITIONAL_WORKFLOWS.md).
