# Additional reproducible workflows

These commands run on local GPU, CPU, or Colab (from a terminal cell). Set your
own paths. They generate **new** output and cannot certify the paper's archived
numbers without its exact checkpoint and original data split.

## 1. Validate RSNA manifest before training

```bash
python audit_manifest.py /private/rsna_manifest.csv --data-root /private/rsna --check-files --output runs/manifest_audit.json
```

The CSV must contain `path,label,split` and ideally `patient_id`. With
`patient_id`, the command rejects patients shared across train/val/test. Without
it, patient separation **cannot be verified**. It never builds a replacement
split. `--patient-column NAME` changes the identifier column. `--check-files`
checks paths; `--output` saves count and SHA256 to JSON.

## 2. Train MobileUNet from real image–mask pairs (optional)

Create a separate **private** CSV `image,mask,split,patient_id` with explicit
train/val assignments. Relative paths use `--data-root`.

```bash
python train_segmentation.py /private/segmentation_pairs.csv --data-root /private/lung_masks --output runs/mobileunet_new.pt --epochs 15 --batch-size 16 --lr 0.001 --seed 42
```

Options: `--num-workers` defaults 0; `--epochs` 15; `--batch-size` 16; `--lr`
0.001; `--seed` 42. The validation set only uses real masks. The output
checkpoint includes the pair manifest hash, train/val counts and selected Dice.
For comparison with a recorded classifier run, use the same segmentation
checkpoint and image–mask split as that run. This command saves the Dice
obtained by the current segmentation training run.

## 3. Train a classifier and continue after Colab disconnects

```bash
python train.py --config configs/local.yaml --head fastkan --seed 42
python train.py --config configs/local.yaml --head fastkan --seed 42 --resume
```

Each complete epoch writes `last.pt` including optimizer, scaler and random
states; `checkpoint.pt` remains the best validation epoch. A partial epoch is
repeated after disconnect. Keep the run directory, crop cache, manifest,
configuration and seed. On a new account, copy the whole run directory and use
paths to the same data. `--resume` rejects a changed head, seed, crop mode or
manifest hash. You may increase `training.max_epochs` in the YAML. A fresh
invocation refuses to overwrite a run with existing checkpoints.

## 4. Analyze saved predictions

```bash
python analyze_predictions.py runs/fastkan_crop_seed42/predictions.csv --bins 10 --output-dir runs/fastkan_crop_seed42/analysis
```

Reports per-dataset ROC-AUC, average precision, Brier score, ECE and calibration
bins, and a precision–recall curve in `analysis.json`. ECE depends on the bin
count; confidence intervals need a separate sampling analysis. No threshold is
selected on the external test set. Compare the RSNA and Kermany results and
state that source, age, acquisition and labeling can all differ.

## 5. Grad-CAM and source probe (exploratory)

```bash
python gradcam.py runs/fastkan_crop_seed42/checkpoint.pt /private/image.png --config configs/local.yaml --output runs/example_gradcam.png
python source_probe.py runs/fastkan_crop_seed42/checkpoint.pt --config configs/local.yaml --seed 42 --output runs/source_probe.json
```

For a legacy checkpoint without metadata, add `--head fastkan`; add `--no-crop`
only if the legacy checkpoint used full images. New checkpoints infer both
from metadata and reject conflicting flags. Grad-CAM uses the final spatial
backbone stage and outputs one overlay; the displayed probability is the
pneumonia class. Only publish deidentified images with the applicable permission.

The probe draws equal counts from each of four `(source, disease label)` cells,
trains a scaled logistic regression on held-out classifier features, and reports
source ROC-AUC. It requires four examples per cell at minimum. A high AUC shows
that source is decodable; it **does not establish a causal shortcut** or explain
the RSNA→Kermany drop. Keep the probe split, balancing procedure, seed and classifier checkpoint
with the probe output so comparisons use the same protocol.

## Provenance boundary

The primary paper metrics require the original RSNA manifest, classifier and
MobileUNet checkpoints and per-image predictions. The cleaned repository does
not ship those private artifacts. Legacy DenseNet, ResNet, KD, DANN and rotation
SSL experiments have different or incomplete protocols and should not be
presented as reproduced primary findings. SRM code is present as an exploratory
module, but its training command and old result have not been validated here.

## 6. Compare two heads on precisely matched images

```bash
python compare_heads.py runs/mlp_crop_seed42/predictions.csv runs/fastkan_crop_seed42/predictions.csv --dataset Kermany --iterations 2000 --seed 42 --output runs/mlp_vs_fastkan.json
```

This reports MLP minus FastKAN ROC-AUC and a stratified paired image-level
bootstrap interval. It rejects mismatched image IDs or labels. The interval is
for these **two checkpoints**, not evidence that the head ranking holds across
seeds. Flags: `--dataset` defaults Kermany, `--iterations` defaults 2000,
`--seed` defaults 42, and `--output` defaults `head_comparison.json`.
