# Kiểm tra gói code — 12/10/2026

Môi trường kiểm tra: Linux, Python 3.12, PyTorch 2.14.1+cpu,
torchvision 0.29.1+cpu. Không có GPU trong môi trường này.

- Phân tích cú pháp toàn bộ source Python: PASS.
- Pytest: 7 tests passed.
- Smoke runtime: FastKAN, Linear, MLP, MobileUNet forward; lưu/nạp checkpoint
  của các head: PASS.
- Tạo dữ liệu ảnh giả với manifest train/val/test và patient_id riêng biệt;
  FastKAN no-crop train 1 epoch, tiếp tục epoch 2 bằng `--resume`, rồi evaluate
  trên các ảnh giả RSNA/Kermany: PASS.
- 11 lệnh CLI `--help`: PASS.
- Notebook phát hành đã xóa output và execution_count.

Kiểm tra bằng dữ liệu giả nhằm xác nhận đường chạy code. Các chỉ số từ kiểm
tra này không phải số liệu bài báo. Không chạy lại bộ dữ liệu thật, không
nạp checkpoint của nghiên cứu, không đo hiệu năng GPU/Windows trong phiên này.

Cài đặt được kiểm tra bằng `pip install -e '.[dev]'` sau khi cài PyTorch CPU.
Trên GPU, cài bản PyTorch phù hợp bằng hướng dẫn chính thức rồi chạy smoke test.
