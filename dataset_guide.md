# Dataset cho End-to-End Training (OD + Relation Detection)

---

## Yêu cầu: Dataset phải có CẢ HAI

Để train end-to-end cả Object Detection + Relation Detection trong 1 lượt,
dataset **bắt buộc** phải có:

1. ✅ **Bounding boxes** + class labels (cho Object Detection)
2. ✅ **Relations/Triplets** `(subject, predicate, object)` (cho Relation Detection)

> COCO **không đủ** — chỉ có boxes, không có relations.
> Phải dùng dataset Scene Graph Generation (SGG).

---

## 1. 🏆 Visual Genome VG150 (KHUYẾN NGHỊ)

**Đây là benchmark chuẩn** được dùng trong hầu hết các paper SGG
(RelTR, BGNN, OvSGTR, GPS-Net, Motif-Net, ...).

### Thông số

| Thuộc tính | Giá trị |
|-----------|---------|
| Số ảnh | 108,077 (train ~70%, val ~30%) |
| Số object classes | **150** (person, tree, building, ...) |
| Số predicate classes | **50** (on, has, wearing, near, ...) |
| Trung bình objects/ảnh | ~11.5 |
| Trung bình relations/ảnh | ~6.2 |
| Dung lượng ảnh | ~15 GB (2 parts) |
| Dung lượng annotations | ~600 MB |

### Download

#### Bước 1: Ảnh Visual Genome

```bash
# Part 1 (~9 GB)
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip

# Part 2 (~5.5 GB)  
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

# Giải nén vào cùng thư mục
mkdir -p vg_data/images
unzip images.zip -d vg_data/images/
unzip images2.zip -d vg_data/images/
# Nếu ảnh nằm trong subfolder VG_100K, VG_100K_2 thì move hết ra images/
```

Hoặc từ trang chính: **https://homes.cs.washington.edu/~ranjay/visualgenome/api.html**

#### Bước 2: VG150 Annotations (pre-processed)

Lấy từ repo Scene-Graph-Benchmark (chuẩn nhất):

```bash
# Clone repo lấy annotations
git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.git
# Annotations nằm trong:
# Scene-Graph-Benchmark/datasets/vg/
#   ├── VG-SGG-with-attri.h5
#   ├── VG-SGG-dicts-with-attri.json
#   └── image_data.json
```

Hoặc download trực tiếp:
- **VG-SGG.h5**: [Google Drive từ repo Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark)
- **VG-SGG-dicts.json**: cùng repo

#### Bước 3: Convert sang JSON format (cho hệ thống của bạn)

Hệ thống OVOD dùng JSON format, cần convert từ H5:

```python
"""Script convert VG150 H5 → JSON cho OVOD system."""
import h5py
import json
import numpy as np

def convert_vg150_to_json(h5_path, dict_path, image_data_path, output_train, output_val):
    # Load H5
    data = h5py.File(h5_path, 'r')
    # Load dicts
    with open(dict_path) as f:
        dicts = json.load(f)
    # Load image metadata
    with open(image_data_path) as f:
        image_data = json.load(f)
    
    # H5 structure:
    # - labels: (N_total_boxes,) — class labels
    # - boxes_1024: (N_total_boxes, 4) — boxes scaled to 1024
    # - img_to_first_box: (N_images,)
    # - img_to_last_box: (N_images,)
    # - relationships: (N_total_rels, 2) — (sub_idx, obj_idx) local to image
    # - predicates: (N_total_rels,) — predicate labels
    # - img_to_first_rel: (N_images,)
    # - img_to_last_rel: (N_images,)
    # - split: (N_images,) — 0=train, 1=val, 2=test
    
    labels = data['labels'][:, 0]
    boxes = data['boxes_1024'][:]
    img_to_first_box = data['img_to_first_box'][:]
    img_to_last_box = data['img_to_last_box'][:]
    
    relationships = data['relationships'][:]
    predicates = data['predicates'][:, 0]
    img_to_first_rel = data['img_to_first_rel'][:]
    img_to_last_rel = data['img_to_last_rel'][:]
    split = data['split'][:]
    
    train_records = []
    val_records = []
    
    for i in range(len(split)):
        # Image info
        img_info = image_data[i]
        w, h = img_info['width'], img_info['height']
        fname = f"{img_info['image_id']}.jpg"
        
        # Boxes for this image
        first_box = img_to_first_box[i]
        last_box = img_to_last_box[i]
        if first_box < 0:
            continue
        
        img_boxes = boxes[first_box:last_box+1].tolist()
        img_labels = labels[first_box:last_box+1].tolist()
        
        # Scale boxes from 1024 to actual pixel coords
        scale_x = w / 1024.0
        scale_y = h / 1024.0
        for b in range(len(img_boxes)):
            img_boxes[b] = [
                img_boxes[b][0] * scale_x,  # x1
                img_boxes[b][1] * scale_y,  # y1
                img_boxes[b][2] * scale_x,  # x2
                img_boxes[b][3] * scale_y,  # y2
            ]
        
        # Labels are 1-indexed in H5, convert to 0-indexed
        img_labels = [int(l) - 1 for l in img_labels]
        
        # Relations for this image
        first_rel = img_to_first_rel[i]
        last_rel = img_to_last_rel[i]
        rels = []
        if first_rel >= 0:
            for r in range(first_rel, last_rel + 1):
                sub_idx = int(relationships[r, 0]) - first_box
                obj_idx = int(relationships[r, 1]) - first_box
                pred_id = int(predicates[r]) - 1  # 1-indexed → 0-indexed
                if 0 <= sub_idx < len(img_boxes) and 0 <= obj_idx < len(img_boxes):
                    rels.append([sub_idx, obj_idx, pred_id])
        
        record = {
            "file_name": fname,
            "width": w,
            "height": h,
            "boxes": img_boxes,
            "labels": img_labels,
            "relations": rels,
        }
        
        if split[i] == 0:
            train_records.append(record)
        elif split[i] == 1:  # val
            val_records.append(record)
        # split=2 is test, skip
    
    with open(output_train, 'w') as f:
        json.dump(train_records, f)
    with open(output_val, 'w') as f:
        json.dump(val_records, f)
    
    print(f"Train: {len(train_records)} images")
    print(f"Val: {len(val_records)} images")
    print(f"Object classes: {len(dicts['idx_to_label'])}")
    print(f"Predicate classes: {len(dicts['idx_to_predicate'])}")


if __name__ == "__main__":
    convert_vg150_to_json(
        h5_path="VG-SGG-with-attri.h5",
        dict_path="VG-SGG-dicts-with-attri.json",
        image_data_path="image_data.json",
        output_train="vg_data/train.json",
        output_val="vg_data/val.json",
    )
```

#### Bước 4: Train

```bash
python main.py \
    --dataset_file vg_json \
    --vg_img_root vg_data/images \
    --vg_train_ann vg_data/train.json \
    --vg_val_ann vg_data/val.json \
    --num_obj_queries 100 \
    --num_rel_queries 50 \
    --num_rel_predicates 50 \
    --epochs 50 \
    --output_dir output/vg150_e2e
```

---

## 2. Open Images V6 — Visual Relationship Detection

### Thông số

| Thuộc tính | Giá trị |
|-----------|---------|
| Tổng số ảnh | ~9 triệu (nhưng chỉ ~100K có relations) |
| Object classes | 600 |
| Relationship types | 329 |
| Dung lượng | Rất lớn (~500GB full, có thể download subset) |

### Download (subset có relationships)

```python
# Dùng FiftyOne (đơn giản nhất)
pip install fiftyone

import fiftyone.zoo as foz
dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    label_types=["relationships"],
    max_samples=10000,  # giới hạn để test trước
)
```

### Nhận xét

- ✅ Rất lớn, đa dạng
- ❌ Format khác, cần viết converter riêng
- ❌ Không phải benchmark chuẩn cho SGG (khó so sánh kết quả)
- ❌ Dung lượng quá lớn cho thử nghiệm ban đầu

---

## 3. PSG — Panoptic Scene Graph Generation

### Thông số

| Thuộc tính | Giá trị |
|-----------|---------|
| Số ảnh | 49K (subset COCO) |
| Object classes | 133 |
| Predicate classes | 56 |
| Đặc biệt | Panoptic segmentation + relations |

### Download

```bash
# Từ repo chính
git clone https://github.com/Jingkang50/OpenPSG.git
# Dataset instructions trong README
```

Website: https://psg.cs.jhu.edu/

### Nhận xét

- ✅ Dựa trên COCO → quen thuộc
- ✅ Clean annotations
- ❌ Cần convert format
- ❌ Không phổ biến bằng VG150

---

## So sánh tổng quan

| Dataset | Objects | Relations | Ảnh | Benchmark chuẩn? | Khuyến nghị |
|---------|---------|-----------|-----|-------------------|-------------|
| **VG150** | 150 classes | 50 predicates | 108K | ✅ **Chuẩn nhất** | 🏆 **Dùng ngay** |
| Open Images V6 | 600 | 329 | 9M+ | ❌ | Quá lớn |
| PSG | 133 | 56 | 49K | Đang phát triển | Tùy chọn |
| COCO | 80 | ❌ Không có | 118K | Chỉ OD | Không đủ |

---

## Khuyến nghị cuối cùng

**Dùng Visual Genome VG150.** Lý do:

1. **Benchmark chuẩn** — tất cả paper SGG đều report trên VG150
2. **Kích thước vừa phải** — 108K ảnh, train được trên 1 GPU
3. **50 predicate classes** — khớp với `--num_rel_predicates 50` trong config
4. **Community support** — nhiều repo có sẵn pre-processed annotations
5. **Dễ so sánh** — kết quả R@50, R@100 có thể so trực tiếp với paper
