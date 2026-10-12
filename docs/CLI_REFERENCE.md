# Danh sách đối số của các script

Bảng lấy từ khai báo argparse trong mã nguồn. Dùng `python SCRIPT --help` để xem trợ giúp tại runtime.

## `analysis.py`

| Đối số/cờ | Khai báo / trợ giúp |
| --- | --- |
| `-h`, `--help` | In trợ giúp |
| `predictions` | help='CSV with dataset,sample_id,label,probability' |
| `--output-dir` | default=None |
| `--bins` | type=int; default=10 |

## `audit_manifest.py`

| Đối số/cờ | Khai báo / trợ giúp |
| --- | --- |
| `-h`, `--help` | In trợ giúp |
| `manifest` |  |
| `--data-root` |  |
| `--patient-column` | default='patient_id' |
| `--check-files` | action='store_true' |
| `--output` |  |

## `compare_heads.py`

| Đối số/cờ | Khai báo / trợ giúp |
| --- | --- |
| `-h`, `--help` | In trợ giúp |
| `left` | help='predictions.csv of the first head' |
| `right` | help='predictions.csv of the second head' |
| `--dataset` | default='Kermany' |
| `--seed` | type=int; default=42 |
| `--iterations` | type=int; default=2000 |
| `--output` | default='head_comparison.json' |

## `evaluate.py`

| Đối số/cờ | Khai báo / trợ giúp |
| --- | --- |
| `-h`, `--help` | In trợ giúp |
| `checkpoint` | help='path to the classifier .pt checkpoint' |
| `--config` | default='configs/default.yaml'; help='YAML config path (default: configs/default.yaml)' |
| `--head` | choices=('fastkan', 'linear', 'mlp'); default=None; help='head for legacy checkpoints without metadata (otherwise inferred)' |
| `--no-crop` | action='store_true'; help='evaluate a legacy no-crop checkpoint without metadata (otherwise inferred)' |
| `--output-dir` | default=None; help='results folder (default: evaluation/ beside the checkpoint)' |

## `gradcam.py`

| Đối số/cờ | Khai báo / trợ giúp |
| --- | --- |
| `-h`, `--help` | In trợ giúp |
| `checkpoint` |  |
| `image` |  |
| `--config` | default='configs/default.yaml' |
| `--head` | choices=['fastkan', 'linear', 'mlp']; help='for legacy checkpoints' |
| `--no-crop` | action='store_true'; help='for legacy checkpoints' |
| `--output` | default='gradcam.png' |

## `pipeline.py`

| Đối số/cờ | Khai báo / trợ giúp |
| --- | --- |
| `-h`, `--help` | In trợ giúp |
| `--config` | default='configs/default.yaml' |
| `--heads` | nargs='+'; choices=['fastkan', 'linear', 'mlp']; default=['fastkan', 'linear', 'mlp'] |
| `--seeds` | nargs='+'; type=int; help='default: training.seeds in YAML' |
| `--skip-no-crop` | action='store_true'; help='omit the three FastKAN no-crop jobs' |
| `--download-missing` | action='store_true'; help='download missing Kaggle data using your account' |
| `--rebuild-rsna` | action='store_true'; help='create a NEW split only if manifest is absent' |
| `--rebuild-seed` | type=int; default=42 |
| `--segmentation-manifest` | help='explicit image/mask train/val CSV when checkpoint is absent' |
| `--segmentation-root` | help='base directory for relative image/mask paths' |
| `--max-hours` | type=float; help='soft deadline; interrupt current job, resume later' |
| `--max-jobs` | type=int; help='run only the first N jobs from the priority list' |
| `--skip-analysis` | action='store_true' |
| `--dry-run` | action='store_true'; help='show job plan without reading or writing data' |

## `prepare_data.py`

| Đối số/cờ | Khai báo / trợ giúp |
| --- | --- |
| `-h`, `--help` | In trợ giúp |
| `--config` | default='configs/default.yaml' |
| `--sources` | nargs='+'; choices=SOURCES; default=['rsna', 'kermany']; help='default: rsna kermany; masks only when training MobileUNet' |
| `--kaggle-command` | default='kaggle'; help='Kaggle CLI executable' |

## `prepare_rsna.py`

| Đối số/cờ | Khai báo / trợ giúp |
| --- | --- |
| `-h`, `--help` | In trợ giúp |
| `--config` | default='configs/default.yaml' |
| `--seed` | type=int; default=42 |

## `source_probe.py`

| Đối số/cờ | Khai báo / trợ giúp |
| --- | --- |
| `-h`, `--help` | In trợ giúp |
| `checkpoint` |  |
| `--config` | default='configs/default.yaml' |
| `--head` | choices=['fastkan', 'linear', 'mlp']; help='for legacy checkpoints' |
| `--no-crop` | action='store_true'; help='for legacy checkpoints' |
| `--seed` | type=int; default=42 |
| `--output` | default='source_probe.json' |

## `train.py`

| Đối số/cờ | Khai báo / trợ giúp |
| --- | --- |
| `-h`, `--help` | In trợ giúp |
| `--config` | default='configs/default.yaml'; help='YAML config path (default: configs/default.yaml; use local.yaml for real data)' |
| `--head` | choices=('fastkan', 'linear', 'mlp'); default=None; help='classifier head (default: model.head in config)' |
| `--seed` | type=int; default=None; help='random seed (default: training.seed in config)' |
| `--no-crop` | action='store_true'; help='train on full images without MobileUNet crop; default is lung crop' |
| `--resume` | action='store_true'; help='continue from last completed epoch in the same run directory' |

## `train_segmentation.py`

| Đối số/cờ | Khai báo / trợ giúp |
| --- | --- |
| `-h`, `--help` | In trợ giúp |
| `manifest` | help='CSV image,mask,split[,patient_id]; train and val specified in advance' |
| `--data-root` |  |
| `--output` | default='runs/mobileunet_new.pt' |
| `--seed` | type=int; default=42 |
| `--epochs` | type=int; default=15 |
| `--batch-size` | type=int; default=16 |
| `--lr` | type=float; default=0.001 |
| `--num-workers` | type=int; default=0 |
