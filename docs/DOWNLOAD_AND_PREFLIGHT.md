# Tải và kiểm tra dữ liệu trước khi train

Tạo `configs/local.yaml` từ `configs/default.yaml`, rồi đặt đường dẫn **riêng
trên máy của bạn**. Cài Kaggle CLI bằng `pip install kaggle`, đăng nhập theo
hướng dẫn của Kaggle (`kaggle auth login` hoặc API token). Với RSNA competition,
đăng nhập web và chấp nhận điều khoản dữ liệu trước. Không đưa token lên GitHub.

```bash
python prepare_data.py --config configs/local.yaml --sources rsna kermany
```

`--sources` nhận `rsna`, `kermany`, `masks` (có thể chọn một hoặc nhiều). Mặc
định tải RSNA và Kermany. Dùng `--sources masks` **chỉ khi** bạn muốn huấn luyện
MobileUNet mới từ mask. `--kaggle-command` chọn đường dẫn executable nếu cần.

Lệnh dùng `paths.rsna_raw_root` cho DICOM thô, `paths.kermany_root` cho ảnh
Kermany, `paths.lung_masks_root` cho mask. Nó kiểm tra cấu trúc dữ liệu trước;
nếu đã đủ thì in `[READY]` và bỏ qua. Nếu thiếu thì gọi Kaggle, giải nén, rồi
kiểm tra lại; tải dở hay Kaggle trả lỗi sẽ không được đánh dấu là đã đủ. Không
tự tải mỗi lần train để tránh tiêu tốn thời gian/dung lượng và tránh tự ý thay
nguồn dữ liệu đã dùng trong bài.

**Ưu tiên dùng lại manifest và ảnh RSNA đã train bản báo cáo.** Nếu chỉ có
DICOM thô và bạn chấp nhận một lần chia tập **mới**, chạy:

```bash
python prepare_rsna.py --config configs/local.yaml --seed 42
python audit_manifest.py /path/to/rsna_split_manifest.csv --data-root /path/to/rsna --check-files
```

`prepare_rsna.py` lấy `Normal` và `Lung Opacity`, bỏ lớp không tương ứng,
chuyển mỗi DICOM sang PNG bằng min–max từng ảnh, tạo split theo bệnh nhân
70/15/15 (làm tròn theo lớp) và lưu `patient_id` cùng file provenance. Nó
**không** ghi đè manifest đang tồn tại. Tập mới không được gọi là tập đã tạo
ra các số liệu trong bài. Danh sách ảnh và thứ tự chia ban đầu chưa đủ bằng
chứng để khẳng định trùng khớp tuyệt đối.

Sau đó:

```bash
python train.py --config configs/local.yaml --head fastkan --seed 42
```

`train.py` tự kiểm tra manifest, từng ảnh RSNA, khả năng trùng bệnh nhân nếu có
`patient_id`, Kermany nếu đã cấu hình, và checkpoint MobileUNet nếu bật crop.
Thiếu gì sẽ dừng trước bước GPU với thông báo tương ứng. Nó không train
MobileUNet hộ bạn: dùng checkpoint gốc để khớp bài báo hoặc chủ động chạy
`train_segmentation.py` cho **một thí nghiệm mới**. Với máy chỉ muốn thử train
không crop: thêm `--no-crop`.

Các tập được tải từ Kaggle vào máy của người chạy; repo GitHub không chứa ảnh,
DICOM, mask, checkpoint, manifest cá nhân hay thông tin đăng nhập.
