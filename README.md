# LMS-ConvNeXt + FastKAN cho phân loại X-quang ngực

Nghiên cứu kết hợp **CNN và FastKAN** để phân loại ảnh X-quang ngực, đồng thời
đánh giá tác động của crop vùng phổi và khả năng tổng quát hóa giữa các bộ dữ liệu.
Pipeline sử dụng LMS-ConvNeXt pico để trích xuất đặc trưng, MobileFusion để kết
hợp đặc trưng của 4 stage, và FastKAN làm head phân loại. Linear và MLP là các
head đối chứng trong cùng khung kiến trúc.

## Mục tiêu nghiên cứu

- Kiểm tra FastKAN có mang lại lợi ích so với Linear và MLP khi dùng chung CNN.
- Đánh giá hiệu quả của lung cropping bằng MobileUNet so với ảnh không crop.
- Đánh giá trên RSNA và kiểm tra ngoài nguồn trên Kermany, thay vì chỉ dựa vào
  hiệu năng trên dữ liệu cùng nguồn huấn luyện.
- Phân tích hành vi dự đoán và thông tin nguồn dữ liệu trong đặc trưng mô hình.

## Phương pháp

Ảnh X-quang được crop theo hộp bao phổi do MobileUNet dự đoán, sau đó tiền xử
lý bằng resize, CLAHE và chuẩn hóa. Ảnh 224×224×3 đi qua CNN, MobileFusion và
head phân loại để tạo dự đoán 2 lớp. Thí nghiệm không crop dùng ảnh toàn khung.

| Thành phần | Vai trò / cấu hình |
| --- | --- |
| MobileUNet | Dự đoán mask phổi để tạo crop, lề hộp bao 4% |
| Stem | Convolution 4×4, stride 4 |
| LMS-ConvNeXt pico | 4 stage, số kênh 32/64/128/256, mỗi stage gồm 2 LMSBlock |
| LMSBlock | 3 nhánh depthwise convolution: 3×3, 7×7, 7×7 dilation 3; concat và Conv 1×1; nhánh residual |
| MobileFusion | Chiếu 4 đầu ra về 256 kênh, GAP thành vector, học trọng số stage và cộng có trọng số |
| FastKAN | Head 256 → 64 → 2, dùng cơ sở Gaussian RBF, grid size 4 |
| Linear / MLP | Head đối chứng dùng chung backbone và fusion |

LMSBlock và MobileFusion là các thành phần thiết kế trong mô hình này. Các
thí nghiệm head/crop không phải đối chứng riêng để xác định đóng góp của từng
thành phần CNN. Kích thước 256 là lựa chọn thiết kế, chưa được khẳng định tối ưu.

## Dữ liệu và thiết kế đánh giá

RSNA là nguồn huấn luyện, validation và test nội bộ. Manifest CSV ghi rõ các
ảnh và tập tương ứng; kiểm tra patient-level separation khi có `patient_id`.
Kermany được dùng cho đánh giá ngoài nguồn, không dùng huấn luyện classifier
trong giao thức này.

Thí nghiệm đối chứng sử dụng seeds **42, 123, 2026**. Trong package, checkpoint
được chọn bằng validation balanced accuracy, các chỉ số tại ngưỡng 0,5 được
báo cáo cùng ROC-AUC. Mỗi run lưu checkpoint, lịch sử train, metrics và dự đoán
để phân tích tiếp. Các cấu hình huấn luyện nằm trong `configs/default.yaml`.

## Kết quả nghiên cứu

ROC-AUC trên Kermany, trung bình ± độ lệch chuẩn mẫu qua 3 seed, tính từ bảng
kết quả được lưu trong `results/paper_reported_metrics.csv`:

| Head / chế độ ảnh | ROC-AUC Kermany |
| --- | --- |
| FastKAN | 0.8089 ± 0.0534 |
| Linear | 0.8323 ± 0.0064 |
| MLP | 0.8415 ± 0.0181 |
| FastKAN không crop | 0.8507 ± 0.0234 |

Trong các kết quả này, FastKAN có crop chưa vượt Linear/MLP trên dữ liệu ngoài
nguồn. FastKAN không crop có AUC trung bình cao hơn FastKAN có crop, cho thấy
crop vùng phổi không tự bảo đảm tổng quát hóa tốt hơn. Phân tích source probe
kiểm tra khả năng nhận biết nguồn từ đặc trưng; AUC nguồn cao không tự chứng
minh một shortcut cụ thể gây ra suy giảm phân loại bệnh.

Các khác biệt RSNA/Kermany về quần thể, thu nhận ảnh, tỷ lệ lớp, tiền xử lý và
nhãn cùng tồn tại, nên không quy suy giảm hiệu năng cho một nguyên nhân duy nhất.
Số liệu tổng hợp là kết quả nghiên cứu đã lưu, không phải kết quả của smoke test.
Gói mã không kèm dataset, manifest thực tế hoặc checkpoint của các run.

## Sử dụng mã nguồn

Code hỗ trợ **GPU NVIDIA, CPU và Google Colab**, dùng chung một package Python.
Các lệnh chính chạy từ thư mục gốc repo:

```bash
python -m pip install -e .
python train.py --config configs/local.yaml --head fastkan --seed 42
python evaluate.py runs/fastkan_crop_seed42/checkpoint.pt --config configs/local.yaml
python pipeline.py --config configs/local.yaml
```

Cần cài PyTorch phù hợp máy và điền đường dẫn dataset/checkpoint trong
`configs/local.yaml` trước khi train. Khi cần tiếp tục run, thêm `--resume`.

- [Hướng dẫn cài đặt, dữ liệu, GPU/CPU, train, evaluate và Colab](docs/RUNNING.md)
- [Danh sách đối số của tất cả script](docs/CLI_REFERENCE.md)
- [Pipeline và giới hạn thời gian/job](docs/PIPELINE.md)
- [Chuẩn bị và tải dataset](docs/DOWNLOAD_AND_PREFLIGHT.md)
- [Grad-CAM, source probe, calibration, bootstrap và segmentation](docs/ADDITIONAL_WORKFLOWS.md)

## Cấu trúc repository

```text
configs/                        Cấu hình model, train và đường dẫn dữ liệu
src/fair2026_lms_kan/models/     LMS-ConvNeXt, MobileFusion, các head, MobileUNet
src/fair2026_lms_kan/            Dataset, train, evaluate và phân tích
notebooks/                      Chạy cùng package trên Colab
results/                        Kết quả nghiên cứu đã lưu
tests/                          Kiểm tra package
docs/                           Hướng dẫn sử dụng và thiết kế thí nghiệm
train.py / evaluate.py / pipeline.py  Điểm chạy chính
```

## Tái lập và kiểm tra

Đối chiếu một run cần đúng manifest, ảnh, preprocessing, cấu hình và checkpoint
liên quan. Chạy trên một split mới tạo ra thí nghiệm mới và có thể cho số khác.
Các kiểm tra package bằng dữ liệu giả xác nhận code chạy được, không thay thế
đánh giá trên ảnh thật. Chi tiết kiểm tra hiện tại:
[docs/RELEASE_CHECKS.md](docs/RELEASE_CHECKS.md).

Các bản CNN baseline trong nghiên cứu được đánh giá theo giao thức riêng và
không được dùng như đối chứng head trong bảng ở trên. Repo này cung cấp pipeline
LMS-ConvNeXt với FastKAN/Linear/MLP và các phân tích liên quan.

Dataset, checkpoint, cache và cấu hình local được giữ ngoài Git. Gói này chưa
kèm giấy phép sử dụng lại mã nguồn hoặc DOI của bài báo.
