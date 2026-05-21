# Hướng dẫn Training & Chạy mô hình OVOD + OVRD

Tài liệu này mô tả quy trình huấn luyện **mô hình OVOD+OVRD** (Transformer end-to-end) trong thư mục gốc dự án, kết hợp dữ liệu chuẩn từ **Scene-Graph-Benchmark.pytorch** (VG150).

---

## 1. Hai codebase trong workspace

| Thành phần | Vai trò | Entry point |
|-----------|---------|-------------|
| **Thư mục gốc** (`main.py`, `models/`, `data/`) | Mô hình OVOD+OVRD — **1 lượt train** cho OD + Relation | `python main.py` |
| **`Scene-Graph-Benchmark.pytorch/`** | Benchmark SGG gốc (Mask R-CNN + relation head) — **2 giai đoạn** | `tools/detector_pretrain_net.py`, `tools/relation_train_net.py` |

> Bạn **không bắt buộc** cài/build Scene-Graph-Benchmark để train OVOD. Chỉ cần **ảnh VG** và file annotation **H5/JSON** từ benchmark đó (hoặc JSON đã convert).

### 1.1. Cấu trúc Scene-Graph-Benchmark.pytorch

```
Scene-Graph-Benchmark.pytorch/
├── README.md, INSTALL.md, DATASET.md, METRICS.md
├── setup.py                         # Build maskrcnn_benchmark (CUDA ext)
├── configs/
│   ├── e2e_relation_X_101_32_8_FPN_1x.yaml   # SGG chính (ResNeXt-101)
│   ├── e2e_relation_R_101_FPN_1x.yaml
│   └── e2e_relation_detector_*.yaml           # Pretrain Faster R-CNN
├── datasets/vg/
│   ├── image_data.json              # ✅ Có sẵn — metadata ảnh (id, w, h)
│   ├── VG-SGG-dicts-with-attri.json # ✅ Có sẵn — 150 object + 50 predicate
│   ├── VG-SGG-with-attri.h5         # ⚠️ Cần tải thêm (xem §3.2)
│   └── generate_attribute_labels.py # Script tạo H5 (nếu tự build)
├── maskrcnn_benchmark/              # Core: data, modeling, engine, solver
│   ├── config/                      # paths_catalog.py — đường dẫn dataset
│   ├── data/datasets/evaluation/vg/ # Loader & metrics VG
│   └── modeling/roi_heads/relation_head/  # Motif, VCTree, TDE, ...
├── tools/
│   ├── detector_pretrain_net.py     # Giai đoạn 1: Faster R-CNN trên VG
│   ├── relation_train_net.py        # Giai đoạn 2: SGG (PredCls/SGCls/SGDet)
│   └── relation_test_net.py         # Evaluation
└── visualization/                   # Notebook visualize kết quả
```

**Workflow benchmark gốc (tham khảo, không dùng cho OVOD):**

```
Bước A: Pretrain detector (MODEL.RELATION_ON False)
   → tools/detector_pretrain_net.py + configs/e2e_relation_detector_*.yaml
Bước B: Train relation head (PredCls / SGCls / SGDet)
   → tools/relation_train_net.py + configs/e2e_relation_*.yaml
   → Cần checkpoint Faster R-CNN pretrained
```

**Workflow OVOD (dùng cho dự án này):**

```
Chuẩn bị VG150 (H5 → JSON) → python main.py (end-to-end, 6 losses)
```

---

## 2. Đánh giá mức độ sẵn sàng (OVOD)

| Thành phần | Trạng thái | Ghi chú |
|------------|-----------|---------|
| Backbone (Swin/ResNet) | ✅ | Qua `timm`, pretrained ImageNet |
| FPN + Transformer | ✅ | Encoder + Decoder, aux loss |
| Object / Relation Head | ✅ | Closed-vocab hoặc CLIP embed |
| SSGA + SGOR | ✅ | Sparse attention + box refine |
| CLIP Text Encoder | ✅ | Frozen, `--clip_model` |
| Criterion (6 losses) | ✅ | ce, bbox, giou, rel, vl, vl_pred |
| Hungarian Matcher | ✅ | Object + Relation |
| Scene Graph JSON Loader | ✅ | `data/vg_json_dataset.py` |
| Convert H5 → JSON | ✅ | `convert.py` |
| Training Loop | ✅ | AdamW, LR backbone ×0.1, grad clip |

---

## 3. Đa máy: CPU / GPU / Mac (MPS)

Code tự chọn thiết bị và preset hyperparameter theo máy.

| `--profile` | Máy | Hành vi |
|-------------|-----|---------|
| `auto` (mặc định) | Không có CUDA | → preset **cpu** (batch=1, image 320, 3 layer, …) |
| `auto` | Có CUDA | → preset **gpu** |
| `cpu` | Máy yếu / không GPU | Train đầy đủ nhưng config nhẹ |
| `cpu_debug` | Test nhanh | 32 train + 16 val, 1 epoch, model nhỏ |
| `gpu` | Workstation / server | batch=2, image 640, aux loss |
| `none` | Tự cấu hình | Không ghi đè default argparse |

| `--device` | Ý nghĩa |
|------------|---------|
| `auto` | CUDA → MPS (Mac) → CPU |
| `cpu` | Bắt buộc CPU |
| `cuda` | GPU (fallback CPU nếu không có) |

**Train trên máy không có GPU (PowerShell):**

```powershell
python main.py `
    --profile auto `
    --dataset_file vg150 `
    --vg_img_root vg_data\images `
    --vg_train_ann vg_data\train.json `
    --vg_val_ann vg_data\val.json `
    --num_rel_predicates 50 `
    --output_dir output\cpu_test
```

**Chỉ thử pipeline (vài phút, 48 ảnh):**

```powershell
python main.py `
    --profile cpu_debug `
    --dataset_file vg150 `
    --vg_img_root vg_data\images `
    --vg_train_ann vg_data\train.json `
    --vg_val_ann vg_data\val.json `
    --num_rel_predicates 50 `
    --output_dir output\cpu_debug
```

**Train subset VG trên CPU (ví dụ 500 ảnh):**

```powershell
python main.py --profile cpu --max_train_samples 500 --max_val_samples 100 ...
```

> CPU train full VG150 (~57k ảnh) **rất chậm** (nhiều giờ/ngày). Dùng `cpu_debug` hoặc `--max_train_samples` trước; train full trên máy có GPU.

---

## 4. Cài đặt môi trường (OVOD)

### 3.1. Yêu cầu

- **Python**: 3.8+
- **GPU**: NVIDIA ≥ 8 GB VRAM (khuyến nghị ≥ 16 GB)
- **CUDA**: tương thích PyTorch (11.7+ / 12.x)
- **RAM**: ≥ 16 GB | **Disk**: ≥ 50 GB (ảnh VG + checkpoint)

### 3.2. Cài dependencies

```powershell
cd "c:\Users\sonb1\Downloads\source_code 2"
python -m venv .venv
.\.venv\Scripts\activate

# PyTorch — chọn bản khớp CUDA: https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

### 3.3. Kiểm tra

```powershell
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
from models import build_model
print('Model imports: OK')
"
```

---

## 5. Chuẩn bị dữ liệu VG150

### 5.1. Cấu trúc thư mục khuyến nghị

```
source_code 2/
├── Scene-Graph-Benchmark.pytorch/datasets/vg/
│   ├── image_data.json                    # có sẵn
│   ├── VG-SGG-dicts-with-attri.json       # có sẵn
│   └── VG-SGG-with-attri.h5               # tải thêm (§4.2)
├── vg_data/
│   ├── images/                            # ảnh VG (*.jpg)
│   ├── train.json                         # sau convert
│   └── val.json
└── convert.py
```

### 5.2. Tải ảnh và annotation H5

**Ảnh Visual Genome** (hai phần, ~15 GB):

- [Part 1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip)
- [Part 2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

Giải nén toàn bộ `*.jpg` vào `vg_data/images/` (tên file: `{image_id}.jpg`).

**Annotation H5** (chưa có trong repo — bắt buộc tải):

Theo `Scene-Graph-Benchmark.pytorch/DATASET.md`:

- [OneDrive — scene graphs](https://1drv.ms/u/s!AmRLLNf6bzcir8xf9oC3eNWlVMTRDw?e=63t7Ed) → đặt tại  
  `Scene-Graph-Benchmark.pytorch/datasets/vg/VG-SGG-with-attri.h5`
- Backup: [Baidu](https://pan.baidu.com/s/1oyPQBDHXMQ5Tsl0jy5OzgA) (mã: 1234), [Weiyun](https://share.weiyun.com/ViTWrFxG)

**Thông số VG150 (sau filter benchmark):**

| | Giá trị |
|--|--------|
| Object classes | **150** (`NUM_CLASSES: 151` trong config SGB = 150 + background) |
| Predicate classes | **50** |
| Split H5 `split` | 0=train, 1=val, 2=test |

### 5.3. Convert H5 → JSON (cho `main.py`)

```powershell
cd "c:\Users\sonb1\Downloads\source_code 2"

python -c "
from convert import convert_vg150_to_json
convert_vg150_to_json(
    h5_path=r'Scene-Graph-Benchmark.pytorch\datasets\vg\VG-SGG-with-attri.h5',
    dict_path=r'Scene-Graph-Benchmark.pytorch\datasets\vg\VG-SGG-dicts-with-attri.json',
    image_data_path=r'Scene-Graph-Benchmark.pytorch\datasets\vg\image_data.json',
    output_train=r'vg_data\train.json',
    output_val=r'vg_data\val.json',
)
"
```

Hoặc sửa đường dẫn trong `convert.py` rồi chạy `python convert.py`.

**Format mỗi record trong JSON:**

```json
{
  "file_name": "1.jpg",
  "width": 800,
  "height": 600,
  "boxes": [[x1, y1, x2, y2]],
  "labels": [0],
  "relations": [[sub_idx, obj_idx, predicate_id]]
}
```

- `boxes`: xyxy pixel; `labels`: 0..149; `relations`: index trong `boxes` cùng ảnh  
- `num_classes` / `num_rel_predicates` được **tự infer** từ JSON khi train

### 5.4. COCO (tùy chọn — chỉ Object Detection)

COCO **không có** relation annotations → `--num_rel_predicates 0` (tắt `L_rel`).

```
coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

---

## 6. Training end-to-end (OVOD)

> **Một lượt train duy nhất:** Object queries + Relation queries qua cùng Transformer Decoder; **6 losses** backprop đồng thời.

### 6.1. Luồng loss

```
Image → Backbone → FPN → Encoder → Decoder (obj + rel queries)
                    ↓
    L_ce + L_bbox + L_giou + L_rel + L_vl + L_vl_pred
                    ↓
         L_total = Σ λᵢ · Lᵢ  (1 backward)
```

| Loss | Mô tả | CLI weight |
|------|--------|------------|
| `loss_ce` | Focal / CE classification | `--w_ce` |
| `loss_bbox` | L1 box (cxcywh normalized) | `--w_bbox` |
| `loss_giou` | GIoU | `--w_giou` |
| `loss_rel` | Relation / predicate CE | `--w_rel` |
| `loss_vl` | CLIP contrastive (objects) | `--w_vl` |
| `loss_vl_pred` | CLIP contrastive (predicates) | `--w_vl_pred` |

Optimizer: **backbone LR = `lr × 0.1`**, phần còn lại = `lr` (AdamW).

### 6.2. Lệnh training chính — VG150 (Linux/bash)

```bash
python main.py \
    --dataset_file vg150 \
    --vg_img_root vg_data/images \
    --vg_train_ann vg_data/train.json \
    --vg_val_ann vg_data/val.json \
    --backbone resnet50 \
    --pretrained 1 \
    --hidden_dim 256 \
    --nheads 8 \
    --enc_layers 6 \
    --dec_layers 6 \
    --num_obj_queries 100 \
    --num_rel_queries 50 \
    --num_rel_predicates 50 \
    --batch_size 2 \
    --lr 1e-4 \
    --epochs 50 \
    --lr_drop 40 \
    --clip_max_norm 0.1 \
    --use_focal 1 \
    --use_aux_loss 1 \
    --w_ce 1.0 \
    --w_bbox 5.0 \
    --w_giou 2.0 \
    --w_rel 1.0 \
    --w_vl 1.0 \
    --w_vl_pred 1.0 \
    --clip_model openai/clip-vit-base-patch32 \
    --rel_recall_k 50 \
    --output_dir output/vg150_e2e \
    --device cuda
```

> `--dataset_file` chấp nhận: `vg_json`, `vg150`, `scene_graph` (cùng loader JSON).

### 6.3. Windows PowerShell

```powershell
python main.py `
    --dataset_file vg150 `
    --vg_img_root "c:\Users\sonb1\Downloads\source_code 2\vg_data\images" `
    --vg_train_ann "c:\Users\sonb1\Downloads\source_code 2\vg_data\train.json" `
    --vg_val_ann "c:\Users\sonb1\Downloads\source_code 2\vg_data\val.json" `
    --backbone resnet50 `
    --pretrained 1 `
    --num_obj_queries 100 `
    --num_rel_queries 50 `
    --num_rel_predicates 50 `
    --batch_size 2 `
    --epochs 50 `
    --output_dir output\vg150_e2e `
    --device cuda
```

### 6.4. Biến thể

**Swin backbone:**

```bash
python main.py \
    --dataset_file vg150 \
    --vg_img_root vg_data/images \
    --vg_train_ann vg_data/train.json \
    --vg_val_ann vg_data/val.json \
    --backbone swin_tiny_patch4_window7_224 \
    --pretrained 1 \
    --num_rel_predicates 50 \
    --output_dir output/swin_vg150
```

**Chỉ COCO detection (không relation):**

```bash
python main.py \
    --dataset_file coco \
    --coco_path /path/to/coco \
    --num_rel_queries 0 \
    --num_rel_predicates 0 \
    --output_dir output/coco_det
```

### 6.5. Debug — overfit test

```powershell
python test_overfit.py
```

Loss phải giảm dần qua ~20 step. Nếu NaN → xem §9.

---

## 7. Tham số CLI

### 7.1. Training

| Tham số | Mặc định | Mô tả |
|---------|---------|-------|
| `--lr` | 1e-4 | LR cho head/transformer (backbone = lr×0.1) |
| `--batch_size` | 2 | Batch/GPU |
| `--weight_decay` | 1e-4 | AdamW |
| `--epochs` | 50 | Số epoch |
| `--lr_drop` | 40 | StepLR: LR ×0.1 |
| `--clip_max_norm` | 0.1 | Gradient clipping |
| `--seed` | 42 | Random seed |

### 7.2. Model

| Tham số | Mặc định | Mô tả |
|---------|---------|-------|
| `--backbone` | swin_tiny_patch4_window7_224 | timm model name |
| `--pretrained` | 1 | ImageNet pretrained |
| `--hidden_dim` | 256 | Transformer dim |
| `--nheads` | 8 | Attention heads |
| `--enc_layers` / `--dec_layers` | 6 / 6 | Số layer |
| `--num_obj_queries` | 100 | Object queries |
| `--num_rel_queries` | 50 | Relation queries |
| `--num_rel_predicates` | 0 | 0=tắt rel loss; VG150=50 |
| `--clip_dim` | 512 | CLIP embedding dim |
| `--clip_model` | openai/clip-vit-base-patch32 | HuggingFace CLIP |
| `--fpn_level` | -1 | FPN level (-1=coarsest) |
| `--image_size` | 640 | Resize input |

### 7.3. Loss

| Tham số | Mặc định | Mô tả |
|---------|---------|-------|
| `--use_focal` | 1 | Sigmoid focal vs softmax CE |
| `--use_aux_loss` | 1 | Deep supervision decoder |
| `--w_ce` / `--w_bbox` / `--w_giou` | 1 / 5 / 2 | Object losses |
| `--w_rel` | 1 | Relation CE |
| `--w_vl` / `--w_vl_pred` | 1 / 1 | VL contrastive |

### 7.4. Dataset

| Tham số | Mặc định | Mô tả |
|---------|---------|-------|
| `--dataset_file` | coco | `coco` \| `vg_json` \| `vg150` \| `scene_graph` |
| `--coco_path` | — | Bắt buộc nếu `coco` |
| `--vg_img_root` | — | Thư mục ảnh VG |
| `--vg_train_ann` / `--vg_val_ann` | — | JSON train/val |
| `--num_workers` | 2 | DataLoader workers |

### 7.5. Hardware & eval

| Tham số | Mặc định | Mô tả |
|---------|---------|-------|
| `--device` | auto | auto / cpu / cuda / mps |
| `--profile` | auto | auto / cpu / cpu_debug / gpu / none |
| `--max_train_samples` | 0 | Giới hạn ảnh train (0=full) |
| `--max_val_samples` | 0 | Giới hạn ảnh val |

### 7.6. Eval & output

| Tham số | Mặc định | Mô tả |
|---------|---------|-------|
| `--eval_map` | 1 | mAP@0.5 trên val |
| `--rel_recall_k` | 50 | R@K cho relations |
| `--output_dir` | "" | Lưu `checkpoint{epoch:04}.pth` |
| `--print_freq` | 50 | Log mỗi N steps |

---

## 8. Output & checkpoint

```
output/vg150_e2e/
├── checkpoint0000.pth
├── checkpoint0001.pth
└── ...
```

```python
# Resume (thêm vào main.py nếu cần)
ckpt = torch.load("output/vg150_e2e/checkpoint0042.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
start_epoch = ckpt["epoch"] + 1
```

---

## 9. Log mẫu

```
Model parameters: 14,142,289
Using focal loss: True | Aux loss: True
Epoch 0 step 50/5000 loss 48.2341
Epoch 0 done in 320.5s | loss_ce 12.92 loss_bbox 0.75 loss_giou 1.21 loss_rel 5.59 loss_vl 0.00
Val epoch 0 | loss_ce 12.41 ... mAP50 0.0012 R@50 0.0000
```

| Epoch | loss_ce | loss_bbox | loss_giou | loss_rel |
|-------|---------|-----------|-----------|----------|
| 0 | ~12–15 | ~0.8–1.2 | ~1.2–1.5 | ~5–7 |
| 10 | ~5–8 | ~0.4–0.6 | ~0.8–1.0 | ~3–5 |
| 40 | ~2–4 | ~0.2–0.4 | ~0.5–0.7 | ~1–3 |

---

## 10. Ước tính tài nguyên

| Config | VRAM | ~Thời gian/epoch (VG, RTX 3090) |
|--------|------|--------------------------------|
| ResNet-50, B=2, 640px | 6–8 GB | ~20–40 phút (tùy số ảnh) |
| Swin-Tiny, B=2, 640px | 8–10 GB | ~25–50 phút |
| ResNet-50, B=1, 480px | 4–5 GB | Chậm hơn nhưng tiết kiệm VRAM |

Giảm VRAM: `--batch_size 1`, `--image_size 480`, `--num_obj_queries 50`, `--enc_layers 3 --dec_layers 3`, `--backbone resnet50`.

---

## 11. Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| `Loss is nan` | `--lr 5e-5`, kiểm tra JSON/boxes hợp lệ |
| CUDA OOM | `--batch_size 1`, `--image_size 480`, giảm queries/layers |
| Thiếu `VG-SGG-with-attri.h5` | Tải theo §4.2; không convert được JSON |
| `FileNotFoundError` ảnh | Đảm bảo `file_name` trong JSON khớp `vg_data/images/` |
| mAP = 0 sau nhiều epoch | Kiểm tra `loss_bbox`/`loss_giou` có giảm; đợi thêm epoch |
| R@K = 0 | `--num_rel_predicates 50`, JSON có `relations`, `loss_rel > 0` |
| `transformers` thiếu | `pip install transformers` (cho CLIP; nếu thiếu → embed ngẫu nhiên) |

---

## 12. Quy trình khuyến nghị (OVOD + VG150)

```
Bước 1: Cài môi trường (§3)
   ↓
Bước 2: python test_overfit.py  → loss giảm
   ↓
Bước 3: Tải ảnh VG + VG-SGG-with-attri.h5 (§4.2)
   ↓
Bước 4: Convert H5 → vg_data/train.json, val.json (§4.3)
   ↓
Bước 5: python main.py --dataset_file vg150 ... (§5.2)
   ↓
Bước 6: Val — mAP@0.5 > 0 và R@50 > 0
   ↓
Bước 7: Tinh chỉnh --w_* , --lr , backbone
```

**So sánh với Scene-Graph-Benchmark (nếu muốn baseline paper):**

| | OVOD (`main.py`) | Scene-Graph-Benchmark |
|--|------------------|------------------------|
| Kiến trúc | DETR-style Transformer | Faster R-CNN + relation ROI head |
| Giai đoạn | 1 stage | 2 stage (detector → relation) |
| Metric chuẩn | mAP@0.5, R@K (đơn giản) | R@K, mR@K, ng-mR@K (xem METRICS.md) |
| Cài đặt | `pip install -r requirements.txt` | `python setup.py build develop` + CUDA ext |

---

## 13. Cấu trúc mã nguồn OVOD

```
source_code 2/
├── main.py                    # CLI, build model, train loop
├── engine.py                  # train_one_epoch(), evaluate()
├── convert.py                 # VG H5 → JSON
├── test_overfit.py            # Sanity check gradient
├── requirements.txt
├── TRAINING_GUIDE.md          # File này
├── dataset_guide.md           # Chi tiết dataset & download links
├── Scene-Graph-Benchmark.pytorch/   # Benchmark gốc + VG annotations
│
├── models/
│   ├── ovod_model.py          # OVODModel
│   ├── backbone.py, fpn.py, transformer.py
│   ├── prediction_heads.py    # ObjectHead, RelationHead
│   ├── ssga.py, sgor.py, clip_text.py
│   └── criterion.py           # HungarianMatcher, SetCriterion
│
├── data/
│   ├── datasets.py            # build_dataset()
│   ├── vg_json_dataset.py     # SceneGraphJsonDataset
│   ├── data_loaders.py, transforms.py
│
└── utils/
    ├── device_utils.py          # auto CPU/GPU profile
    ├── box_ops.py, eval_metrics.py, misc.py
```

---

## 14. Tài liệu tham khảo thêm

- `dataset_guide.md` — so sánh VG150 / Open Images / PSG  
- `Scene-Graph-Benchmark.pytorch/DATASET.md` — download H5 chính thức  
- `Scene-Graph-Benchmark.pytorch/METRICS.md` — metric PredCls / SGCls / SGDet  
- `MATH_DOCUMENTATION.md` — công thức loss chi tiết
