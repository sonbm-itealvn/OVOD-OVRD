"""
Scene-graph / VG-style dataset from JSON annotations.

Mỗi phần tử là một ảnh với box xyxy tuyệt đối (pixel), nhãn lớp liên tục 0..C-1,
và relations (sub_idx, obj_idx, predicate_id) chỉ số trong danh sách object của ảnh đó.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import make_coco_transforms


def _infer_counts(records: List[Dict[str, Any]]) -> tuple:
    max_cls = -1
    max_pred = -1
    for r in records:
        for lab in r.get("labels", []) or []:
            max_cls = max(max_cls, int(lab))
        for tri in r.get("relations", []) or []:
            if len(tri) >= 3:
                max_pred = max(max_pred, int(tri[2]))
    num_classes = max(max_cls + 1, 1)
    num_pred = max(max_pred + 1, 0)
    return num_classes, num_pred


class SceneGraphJsonDataset(Dataset):
    """
    Đọc JSON: list các dict {file_name, boxes, labels, relations?, width?, height?}
    """

    def __init__(self, img_root: Path, ann_file: Path, transforms, image_set: str):
        self.img_root = Path(img_root)
        self.transforms = transforms
        with ann_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "images" in data:
            self.records = self._from_coco_style(data)
        else:
            if not isinstance(data, list):
                raise ValueError("Annotation JSON phải là list các ảnh hoặc dict có key 'images'.")
            self.records = data
        if len(self.records) == 0:
            raise ValueError(f"Không có mẫu nào trong {ann_file}")

    @staticmethod
    def _from_coco_style(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Hỗ trợ format gần COCO: images + annotations + relations (optional)."""
        id_to_file = {im["id"]: im["file_name"] for im in data["images"]}
        id_to_size = {im["id"]: (im["height"], im["width"]) for im in data["images"]}
        anns_by_img: Dict[int, List] = {}
        for a in data.get("annotations", []):
            anns_by_img.setdefault(a["image_id"], []).append(a)
        rels_by_img: Dict[int, List] = {}
        for rel in data.get("relations", []):
            rels_by_img.setdefault(rel["image_id"], []).append(rel)

        cat_ids = sorted([c["id"] for c in data.get("categories", [])])
        cat2i = {cid: i for i, cid in enumerate(cat_ids)} if cat_ids else {}

        pred_cat_ids = sorted([c["id"] for c in data.get("predicate_categories", [])])
        pred2i = {cid: i for i, cid in enumerate(pred_cat_ids)} if pred_cat_ids else {}

        records = []
        for im in data["images"]:
            iid = im["id"]
            h, w = id_to_size[iid]
            objs = sorted(anns_by_img.get(iid, []), key=lambda x: x.get("id", 0))
            boxes, labels = [], []
            ann_id_to_idx = {}
            idx = 0
            for o in objs:
                x, y, bw, bh = o["bbox"]
                if bw <= 1 or bh <= 1:
                    continue
                aid = o.get("id", idx)
                ann_id_to_idx[aid] = idx
                boxes.append([x, y, x + bw, y + bh])
                cid = o["category_id"]
                labels.append(cat2i.get(cid, int(cid)) if cat2i else int(cid))
                idx += 1

            relations = []
            for rel in rels_by_img.get(iid, []):
                sid = ann_id_to_idx.get(rel.get("subject_id"), rel.get("subject_idx"))
                oid = ann_id_to_idx.get(rel.get("object_id"), rel.get("object_idx"))
                pid = rel["predicate_id"]
                if pred2i:
                    pid = pred2i.get(pid, int(pid))
                else:
                    pid = int(pid)
                if sid is None or oid is None:
                    continue
                relations.append([int(sid), int(oid), int(pid)])

            records.append(
                {
                    "file_name": id_to_file[iid],
                    "width": w,
                    "height": h,
                    "boxes": boxes,
                    "labels": labels,
                    "relations": relations,
                }
            )
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        r = self.records[index]
        path = self.img_root / r["file_name"]
        image = Image.open(path).convert("RGB")
        w, h = image.size

        boxes = r.get("boxes") or []
        labels = r.get("labels") or []
        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)

        rels = r.get("relations") or []
        if len(rels) == 0:
            rel_t = torch.zeros((0, 3), dtype=torch.int64)
        else:
            rel_t = torch.as_tensor(rels, dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "relations": rel_t,
            "image_id": torch.as_tensor([index], dtype=torch.int64),
            "orig_size": torch.as_tensor([int(h), int(w)], dtype=torch.int64),
            "size": torch.as_tensor([int(h), int(w)], dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def build_scene_graph_dataset(image_set: str, config: Dict[str, Any]):
    img_root = Path(config["vg_img_root"])
    if image_set == "train":
        ann = Path(config["vg_train_ann"])
    else:
        ann = Path(config["vg_val_ann"])
    if not img_root.is_dir():
        raise FileNotFoundError(f"vg_img_root không tồn tại: {img_root}")
    if not ann.is_file():
        raise FileNotFoundError(f"File annotation không tồn tại: {ann}")

    with ann.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    records_preview = (
        SceneGraphJsonDataset._from_coco_style(raw) if isinstance(raw, dict) and "images" in raw else raw
    )
    num_classes, num_pred = _infer_counts(records_preview)
    config["num_classes"] = num_classes
    if int(config.get("num_rel_predicates", 0) or 0) <= 0 and num_pred > 0:
        config["num_rel_predicates"] = num_pred

    transforms = make_coco_transforms(image_set, image_size=int(config.get("image_size", 640)))
    return SceneGraphJsonDataset(img_root, ann, transforms, image_set)
