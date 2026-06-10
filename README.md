# RL-SAHI: Adaptive Slicing for Small Object Detection

Repo này thử nghiệm pipeline phát hiện vật thể nhỏ trên VisDrone bằng YOLO + SAHI-style slicing + DQN agent. Ý tưởng chính là chạy YOLO toàn ảnh để lấy trạng thái ban đầu, dùng DQN chọn vùng ROI khó/nhỏ cần phóng to, sau đó chạy YOLO trên crop và gộp kết quả.

## Thành phần chính

- `scripts/prepare_visdrone.py`: chuẩn bị dữ liệu VisDrone về cấu trúc `data/raw`.
- `scripts/detect.py`: chạy YOLO và sinh detection cache.
- `scripts/hard_region.py`: sinh hard-region cache dùng cho reward/training.
- `scripts/train.py`: train DQN chọn vùng slice.
- `scripts/infer.py`: chạy adaptive slicing inference và lưu prediction/visualization.
- `scripts/train_yolo.py`: fine-tune YOLO trên VisDrone.
- `configs/default.yaml`: workflow cũ dùng `yolo11s.pt` COCO.
- `configs/default_visdrone.yaml`: workflow VisDrone, dùng detector đã train cho 10 class VisDrone.

## Lưu ý trước khi upload GitHub

Repo đã có `.gitignore` để bỏ qua dữ liệu lớn và file ảnh:

- ảnh: `*.jpg`, `*.jpeg`, `*.png`, `*.bmp`, `*.gif`, `*.webp`, `*.tif`, `*.tiff`, `*.svg`;
- dữ liệu/cache/output: `archive/`, `data/raw/`, `data/cache/`, `data/cache_*/`, `outputs/`, `runs/`;
- checkpoint/model sinh ra: `*.pt`, `*.pth`, `*.onnx`, `*.engine`.

File `yolo11s.pt` đang được cho phép commit bằng rule `!yolo11s.pt` vì đây là weight mẫu nhỏ của workflow mặc định. Nếu không muốn đưa weight này lên GitHub, hãy xóa dòng `!yolo11s.pt` trong `.gitignore`.

Máy hiện tại đang nằm trong một Git repo cha ở `D:/`, vì vậy khi push GitHub nên tạo repo riêng hoặc chỉ add đúng thư mục `D:/RL-SAHI`.

## Cài đặt môi trường

Kích hoạt môi trường đã cấu hình:

```powershell
conda activate tvtuan
```

Cài dependency Python cơ bản nếu cần:

```powershell
pip install -r requirements.txt
```

Kiểm tra GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Kết quả mong đợi trên máy hiện tại:

```text
2.11.0+cu128
True
NVIDIA GeForce RTX 5060 Laptop GPU
```

## Chuẩn bị dữ liệu

Dữ liệu không được commit lên GitHub. Đặt hoặc giải nén VisDrone vào `archive/`, sau đó chạy:

```powershell
python scripts/prepare_visdrone.py --source archive --overwrite
```

Sau khi chuẩn bị, cấu trúc dữ liệu chính là:

```text
data/raw/images/train
data/raw/images/val
data/raw/labels/train
data/raw/labels/val
```

## Workflow VisDrone khuyến nghị

Workflow hiện tại nên dùng config VisDrone:

```powershell
configs/default_visdrone.yaml
```

Sinh detection cache:

```powershell
python scripts/detect.py --config configs/default_visdrone.yaml --split train
python scripts/detect.py --config configs/default_visdrone.yaml --split val
```

Sinh hard-region cache:

```powershell
python scripts/hard_region.py --config configs/default_visdrone.yaml --split train
python scripts/hard_region.py --config configs/default_visdrone.yaml --split val
```

Train DQN:

```powershell
python scripts/train.py --config configs/default_visdrone.yaml --split train --episodes 30000
```

Chạy inference thử trên validation:

```powershell
python scripts/infer.py --config configs/default_visdrone.yaml --split val --limit 10
```

Chạy full validation inference:

```powershell
python scripts/infer.py --config configs/default_visdrone.yaml --split val
```

Kết quả inference nằm trong:

```text
runs/infer_visdrone/detections
runs/infer_visdrone/metadata
runs/infer_visdrone/visualizations
```

## Workflow YOLO COCO mặc định

Config mặc định dùng `yolo11s.pt` pretrained COCO:

```powershell
python scripts/detect.py --split val --limit 5
python scripts/hard_region.py --split val --limit 5
python scripts/infer.py --split val --limit 5
```

Không nên dùng checkpoint DQN của workflow COCO chung với workflow VisDrone, vì detector khác nhau tạo feature/state dimension khác nhau.

## Gợi ý lệnh Git

Nếu muốn upload riêng project này lên GitHub, cách sạch nhất là tạo repo Git ngay trong `D:/RL-SAHI` hoặc copy project sang thư mục repo mới. Sau đó:

```powershell
git init
git add .gitignore README.md AGENTS.md CLAUDE.md GEMINI.md configs scripts src requirements.txt VisDrone.yaml yolo11s.pt
git commit -m "Initial RL-SAHI project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

Trước khi commit, kiểm tra file lớn hoặc dữ liệu ảnh không bị add nhầm:

```powershell
git status --short
```

Nếu thấy `data/raw`, `archive`, `runs`, `outputs` hoặc file ảnh trong danh sách staged, hãy gỡ khỏi stage trước khi commit.
