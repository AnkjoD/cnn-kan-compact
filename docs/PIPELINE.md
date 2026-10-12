# Pipeline tái lập các thí nghiệm chính

Từ thư mục gốc repo:

```bash
pip install -e .
cp configs/default.yaml configs/local.yaml
# Sửa tất cả đường dẫn thực tế trong configs/local.yaml.
python pipeline.py --config configs/local.yaml --dry-run
python pipeline.py --config configs/local.yaml
```

Pipeline mặc định chạy lần lượt theo seed 42, 123, 2026: FastKAN crop,
Linear crop, MLP crop, FastKAN không crop. Với mỗi job, `train.py` lưu
`last.pt` sau từng epoch, chọn `checkpoint.pt` theo validation balanced
accuracy, rồi đánh giá RSNA test và Kermany. Tiếp đó pipeline tính PR,
calibration và so sánh cặp AUC MLP/Linear với FastKAN nếu đủ file dự đoán.
Không tự chạy Grad-CAM/source probe vì cần chọn ảnh và diễn giải riêng.

Nếu dữ liệu Kaggle chưa có:

```bash
python pipeline.py --config configs/local.yaml --download-missing
```

Điều này **không** tự tạo manifest RSNA hay checkpoint MobileUNet từ không khí.
Dùng lại manifest và checkpoint gốc nếu muốn đối chiếu kết quả bài báo. Nếu
chỉ có DICOM thô và chấp nhận một **split mới**:

```bash
python pipeline.py --config configs/local.yaml --download-missing --rebuild-rsna
```

Nếu thiếu MobileUNet và bạn có manifest ảnh–mask train/val riêng:

```bash
python pipeline.py --config configs/local.yaml --download-missing \
  --segmentation-manifest /private/segmentation_pairs.csv \
  --segmentation-root /private/lung_masks
```

Checkpoint phân đoạn và split RSNA mới là **thí nghiệm mới**, không được ghi
thành kết quả cũ của bản thảo. Không dùng `--rebuild-rsna` nếu manifest đã có:
pipeline sẽ giữ file đó.

## Chạy chia nhiều phiên Colab

```bash
python pipeline.py --config configs/local.yaml --download-missing --max-hours 5.5 --max-jobs 3
```

Lặp lại **đúng lệnh này** ở phiên sau. Run hoàn tất được bỏ qua, run có
`last.pt` được tiếp tục từ epoch hoàn tất cuối. `--max-jobs 3` giới hạn ba
job **chưa hoàn tất trong mỗi lần gọi**. `--max-hours` kiểm soát giai đoạn
train job; download và bước chuẩn bị ban đầu có thể mất thêm thời gian. Tắt
runtime giữa epoch thì epoch đó được làm lại. Giữ nguyên manifest, config,
checkpoint phân đoạn, cache và thư mục `runs/` qua các phiên.

Các cờ khác: `--heads fastkan linear mlp`, `--seeds 42 123 2026`,
`--skip-no-crop`, `--skip-analysis`, `--rebuild-seed 42`, `--dry-run`,
`--config PATH`. Dùng `python pipeline.py --help` cho danh sách chính thức.

Pipeline so hash manifest và metadata của checkpoint trước khi bỏ qua một
run cũ; phát hiện bất đồng sẽ báo lỗi thay vì trộn kết quả. Tuy vậy manifest
và checkpoint của bản thảo chưa có trong repo, nên việc chạy từ Kaggle cho
ra số mới, không bảo đảm trùng con số đã công bố. Hướng dẫn tạo repo và push nằm trong README.
