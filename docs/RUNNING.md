# Hướng dẫn chạy

## 1. Cài đặt

Khuyến nghị Python 3.11. Giải nén gói ZIP rồi mở terminal **trong thư mục có
`train.py` và `pyproject.toml`**.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Nếu PowerShell chặn kích hoạt, dùng trực tiếp `.venv\Scripts\python.exe`
thay cho `python` trong các lệnh sau.

### Linux / server

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**GPU NVIDIA:** lấy lệnh cài `torch` và `torchvision` phù hợp hệ điều hành và
CUDA từ [bộ chọn chính thức của PyTorch](https://pytorch.org/get-started/locally/).
Cài PyTorch trước khi cài package này.

**CPU:** có thể cài bằng:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Sau đó, trên cả Windows và Linux:

```bash
python -m pip install -e .
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python tests/smoke_runtime.py
```

Code tự chọn CUDA nếu PyTorch nhận GPU, ngược lại dùng CPU. Smoke test kiểm tra
forward của 3 head, MobileUNet và lưu/nạp trọng số trên dữ liệu giả. Nó không
train lại toàn bộ nghiên cứu. CPU train được nhưng sẽ chậm.

## 2. Cấu hình dữ liệu

Copy `configs/default.yaml` thành `configs/local.yaml` rồi sửa các đường dẫn:

```powershell
Copy-Item configs/default.yaml configs/local.yaml
```

Trên Linux dùng `cp configs/default.yaml configs/local.yaml`.
Ví dụ đường dẫn Windows trong YAML, dùng dấu `/`:

```yaml
paths:
  manifest: D:/xray/rsna_split_manifest.csv
  rsna_raw_root: D:/xray/rsna_raw
  rsna_root: D:/xray/rsna_png
  kermany_root: D:/xray/chest_xray
  lung_masks_root: D:/xray/lung_masks
  segmentation_checkpoint: D:/xray/mobileunet_final.pt
  crop_cache: D:/xray/lung_crop_cache
  output_dir: runs
```

Giữ các mục `data`, `model`, `training` của cấu hình. Hyperparameter được đọc
trực tiếp từ YAML. `data.image_size` mặc định 224. Nếu gặp lỗi multiprocessing
trên Windows/Colab, đặt `training.num_workers: 0`. Nếu hết VRAM, giảm
`training.batch_size` và lưu cấu hình của run mới.

Manifest RSNA cần các cột `path,label,split`, nên có thêm `patient_id` để kiểm
tra bệnh nhân không xuất hiện ở nhiều tập. Nhãn: 0 = Normal, 1 = Pneumonia;
split: `train`, `val`, `test`. Đường dẫn tương đối được tính từ `rsna_root`.
Kermany dùng cấu trúc `chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}`.
Toàn bộ các phần Kermany được gộp để đánh giá ngoài nguồn, không dùng train
classifier trong protocol này. Chi tiết: [docs/DATA.md](DATA.md).

### Chưa có dataset?

```bash
python prepare_data.py --config configs/local.yaml --sources rsna kermany
```

Cần xác thực tài khoản Kaggle và quyền truy cập nguồn dữ liệu. Hướng dẫn ở
[docs/DOWNLOAD_AND_PREFLIGHT.md](DOWNLOAD_AND_PREFLIGHT.md).
`requirements.txt`/`pyproject.toml` cài thư viện, không chứa ảnh hoặc tài khoản.
Nếu đã có dữ liệu, không cần tải lại.

Nếu chỉ có DICOM và cần tạo split mới:

```bash
python pipeline.py --config configs/local.yaml --download-missing --rebuild-rsna
```

Nếu dùng crop, cần checkpoint MobileUNet đã train. Nếu chưa có, train từ
manifest ảnh–mask bằng `train_segmentation.py`, theo
[docs/ADDITIONAL_WORKFLOWS.md](ADDITIONAL_WORKFLOWS.md).

## 3. Train một head / một seed

```bash
python train.py --config configs/local.yaml --head fastkan --seed 42
python train.py --config configs/local.yaml --head linear --seed 42
python train.py --config configs/local.yaml --head mlp --seed 42
python train.py --config configs/local.yaml --head fastkan --seed 42 --no-crop
```

Tiếp tục run bị gián đoạn:

```bash
python train.py --config configs/local.yaml --head fastkan --seed 42 --resume
```

`checkpoint.pt` giữ epoch có validation balanced accuracy tốt nhất. `last.pt`
giữ trạng thái ở epoch hoàn tất gần nhất để tiếp tục. Epoch đang dở cần chạy
lại. Run mới từ chối ghi đè checkpoint đã có.

## 4. Evaluate

```bash
python evaluate.py runs/fastkan_crop_seed42/checkpoint.pt --config configs/local.yaml
```

Lệnh lưu `metrics.json` và `predictions.csv`. Head/crop được đọc từ metadata
checkpoint do package tạo. Với checkpoint không lưu metadata, truyền head và
chế độ crop thực tế bằng cờ tương ứng. Chỉ nạp checkpoint từ nguồn bạn tin cậy.

## 5. Chạy pipeline lần lượt

Xem kế hoạch trước:

```bash
python pipeline.py --config configs/local.yaml --dry-run
```

Chạy toàn bộ kế hoạch mặc định:

```bash
python pipeline.py --config configs/local.yaml
```

Có 12 job: với mỗi seed 42, 123, 2026, chạy FastKAN crop, Linear crop, MLP crop,
FastKAN no-crop. Pipeline bỏ qua run đã đủ checkpoint/metrics/predictions,
tiếp tục run có `last.pt`, rồi phân tích dự đoán và so sánh head khi đủ file.
Không truyền cờ không có nghĩa bỏ qua việc chuẩn bị đường dẫn và dữ liệu.

Giới hạn mỗi phiên:

```bash
python pipeline.py --config configs/local.yaml --max-jobs 3 --max-hours 5.5
```

Chạy lại cùng lệnh để tiếp tục. `--max-jobs` đếm job chưa hoàn tất được chạy
trong phiên này. `--max-hours` giới hạn giai đoạn train/evaluate của classifier,
không bao gồm tải dữ liệu, train segmentation và phân tích sau đó. Nếu bị ngắt
trước khi epoch đầu hoàn tất thì chưa có `last.pt` để tiếp tục epoch đó.

## 6. Các cờ chính

### `train.py`

| Cờ | Ý nghĩa | Mặc định |
| --- | --- | --- |
| `--config PATH` | Cấu hình YAML | `configs/default.yaml` |
| `--head fastkan/linear/mlp` | Head phân loại | `model.head` trong YAML |
| `--seed N` | Seed của run | `training.seed` trong YAML |
| `--no-crop` | Bỏ crop phổi | Tắt |
| `--resume` | Tiếp tục từ `last.pt` | Tắt |
| `-h`, `--help` | Trợ giúp | — |

### `evaluate.py`

| Cờ/đối số | Ý nghĩa |
| --- | --- |
| `CHECKPOINT` | File trọng số bắt buộc |
| `--config PATH` | Cấu hình YAML |
| `--head fastkan/linear/mlp` | Head khi checkpoint thiếu metadata |
| `--no-crop` | Chế độ no-crop khi checkpoint thiếu metadata |
| `--output-dir PATH` | Nơi lưu kết quả, mặc định `evaluation/` cạnh checkpoint |
| `-h`, `--help` | Trợ giúp |

### `pipeline.py`

| Cờ | Ý nghĩa | Mặc định |
| --- | --- | --- |
| `--config PATH` | Cấu hình YAML | `configs/default.yaml` |
| `--heads HEAD ...` | Các head cần chạy | fastkan, linear, mlp |
| `--seeds N ...` | Các seed cần chạy | `training.seeds` |
| `--skip-no-crop` | Bỏ FastKAN no-crop | Tắt |
| `--download-missing` | Tải dataset còn thiếu qua Kaggle | Tắt |
| `--rebuild-rsna` | Tạo split mới nếu chưa có manifest | Tắt |
| `--rebuild-seed N` | Seed khi tạo split mới | 42 |
| `--segmentation-manifest PATH` | CSV ảnh–mask để train khi thiếu MobileUNet | Không có |
| `--segmentation-root PATH` | Thư mục gốc ảnh–mask | Theo YAML |
| `--max-hours N` | Giới hạn thời gian job classifier | Không giới hạn |
| `--max-jobs N` | Số job chưa xong tối đa mỗi phiên | Không giới hạn |
| `--skip-analysis` | Bỏ phân tích dự đoán/so sánh head | Tắt |
| `--dry-run` | Chỉ in kế hoạch | Tắt |
| `-h`, `--help` | Trợ giúp | — |

Danh sách đối số của **tất cả script**: [docs/CLI_REFERENCE.md](CLI_REFERENCE.md).
Các ví dụ Grad-CAM, probe, bootstrap, calibration và segmentation:
[docs/ADDITIONAL_WORKFLOWS.md](ADDITIONAL_WORKFLOWS.md).

## 7. Chạy trên Colab

Mở `notebooks/run_on_colab.ipynb` trong Colab, chọn runtime GPU và sửa `REPO_URL`
sau khi bạn đã push. Nếu chưa push, notebook nhận ZIP cùng cấu trúc thư mục.
Mount Drive, sửa `DRIVE_ROOT`/các đường dẫn và bật công tắc train/evaluate khi cần.
Notebook dùng cùng package Python như máy local. Kết quả/crop cache trên Drive
có thể giữ qua các phiên. Không cần máy cá nhân có GPU mạnh để dùng Colab.

