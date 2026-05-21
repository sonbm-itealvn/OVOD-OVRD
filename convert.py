"""Script convert VG150 H5 → JSON cho OVOD system."""
import h5py
import json
from pathlib import Path

import numpy as np


def _to_py(x):
    """Chuyển numpy scalar / array sang kiểu Python native (JSON-safe)."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def convert_vg150_to_json(
    h5_path,
    dict_path,
    image_data_path,
    output_train,
    output_val,
    num_val_im=5000,
    filter_empty_rels=True,
):
    """
    Convert VG-SGG H5 → JSON.

    Split trong H5 (giống Scene-Graph-Benchmark visual_genome.py):
      - split == 0: pool train + val (lấy num_val_im ảnh đầu làm val, phần còn lại là train)
      - split == 2: test (bỏ qua khi export train/val)
    """
    data = h5py.File(h5_path, "r")
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
    
    # Indices thuộc train+val pool (split==0), khớp maskrcnn_benchmark load_graphs()
    split_mask = split == 0
    split_mask &= img_to_first_box >= 0
    if filter_empty_rels:
        split_mask &= img_to_first_rel >= 0
    pool_indices = np.where(split_mask)[0]
    val_indices = set(int(i) for i in pool_indices[:num_val_im])
    train_indices = set(int(i) for i in pool_indices[num_val_im:])

    train_records = []
    val_records = []

    for i in range(len(split)):
        # Image info
        img_info = image_data[i]
        w, h = int(_to_py(img_info["width"])), int(_to_py(img_info["height"]))
        fname = f"{int(_to_py(img_info['image_id']))}.jpg"

        # Boxes for this image
        first_box = int(_to_py(img_to_first_box[i]))
        last_box = int(_to_py(img_to_last_box[i]))
        if first_box < 0:
            continue

        img_boxes = boxes[first_box : last_box + 1].tolist()
        img_labels = labels[first_box : last_box + 1].tolist()

        # Scale boxes from 1024 to actual pixel coords
        scale_x = w / 1024.0
        scale_y = h / 1024.0
        for b in range(len(img_boxes)):
            img_boxes[b] = [
                float(img_boxes[b][0]) * scale_x,
                float(img_boxes[b][1]) * scale_y,
                float(img_boxes[b][2]) * scale_x,
                float(img_boxes[b][3]) * scale_y,
            ]

        # Labels are 1-indexed in H5, convert to 0-indexed
        img_labels = [int(_to_py(l)) - 1 for l in img_labels]

        # Relations for this image
        first_rel = int(_to_py(img_to_first_rel[i]))
        last_rel = int(_to_py(img_to_last_rel[i]))
        rels = []
        if first_rel >= 0:
            for r in range(first_rel, last_rel + 1):
                sub_idx = int(_to_py(relationships[r, 0])) - first_box
                obj_idx = int(_to_py(relationships[r, 1])) - first_box
                pred_id = int(_to_py(predicates[r])) - 1
                if 0 <= sub_idx < len(img_boxes) and 0 <= obj_idx < len(img_boxes):
                    rels.append([int(sub_idx), int(obj_idx), int(pred_id)])

        record = {
            "file_name": fname,
            "width": w,
            "height": h,
            "boxes": img_boxes,
            "labels": img_labels,
            "relations": rels,
        }

        if i in val_indices:
            val_records.append(record)
        elif i in train_indices:
            train_records.append(record)

    Path(output_train).parent.mkdir(parents=True, exist_ok=True)
    with open(output_train, "w", encoding="utf-8") as f:
        json.dump(train_records, f, default=_json_default)
    with open(output_val, "w", encoding="utf-8") as f:
        json.dump(val_records, f, default=_json_default)
    
    data.close()

    print(f"Train: {len(train_records)} images")
    print(f"Val: {len(val_records)} images")
    print(f"Object classes: {len(dicts['idx_to_label'])}")
    print(f"Predicate classes: {len(dicts['idx_to_predicate'])}")


if __name__ == "__main__":
    _VG = "Scene-Graph-Benchmark.pytorch/datasets/vg"
    convert_vg150_to_json(
        h5_path=f"{_VG}/VG-SGG-with-attri.h5",
        dict_path=f"{_VG}/VG-SGG-dicts-with-attri.json",
        image_data_path=f"{_VG}/image_data.json",
        output_train="vg_data/train.json",
        output_val="vg_data/val.json",
    )