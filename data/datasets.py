import json
from pathlib import Path

import torch
from torchvision.datasets import CocoDetection

from .transforms import make_coco_transforms


def _load_coco_category_map(ann_path: Path):
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cats = sorted(data["categories"], key=lambda c: c["id"])
    cat2contiguous = {c["id"]: i for i, c in enumerate(cats)}
    return cat2contiguous, len(cats)


class CocoDetectionOVOD(CocoDetection):
    """COCO detection with boxes/labels tensors and optional relations."""

    def __init__(self, img_folder, ann_file, transforms, cat2contiguous):
        super().__init__(root=str(img_folder), annFile=str(ann_file))
        self._transforms = transforms
        self.cat2contiguous = cat2contiguous

    def __getitem__(self, index):
        image, anns = super().__getitem__(index)
        w, h = image.size
        boxes = []
        labels = []
        for obj in anns:
            if obj.get("iscrowd", 0) == 1:
                continue
            cid = obj["category_id"]
            if cid not in self.cat2contiguous:
                continue
            x, y, bw, bh = obj["bbox"]
            if bw <= 1 or bh <= 1:
                continue
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat2contiguous[cid])

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.as_tensor([self.ids[index]], dtype=torch.int64),
            "orig_size": torch.as_tensor([int(h), int(w)], dtype=torch.int64),
            "size": torch.as_tensor([int(h), int(w)], dtype=torch.int64),
            "relations": torch.zeros((0, 3), dtype=torch.int64),
        }

        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target


def _infer_metadata(ann_path: Path):
    """Read metadata (num_classes) from annotation file without building a dataset."""
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "categories" in data:
        return len(data["categories"])
    return None


def build_dataset(image_set, config):
    dataset_file = config.get("dataset_file", "coco").lower()
    if dataset_file == "coco":
        return _build_coco_dataset(image_set, config)
    if dataset_file in ("vg_json", "vg150", "scene_graph"):
        from .vg_json_dataset import build_scene_graph_dataset
        return build_scene_graph_dataset(image_set, config)
    raise NotImplementedError(f"Dataset '{dataset_file}' is not implemented.")


def _build_coco_dataset(image_set, config):
    coco_path = Path(config["coco_path"])
    if not coco_path.is_dir():
        raise FileNotFoundError(f"coco_path is not a directory: {coco_path}")

    if image_set == "train":
        img_folder = coco_path / "train2017"
        ann_file = coco_path / "annotations" / "instances_train2017.json"
    else:
        img_folder = coco_path / "val2017"
        ann_file = coco_path / "annotations" / "instances_val2017.json"

    if not img_folder.is_dir():
        raise FileNotFoundError(f"Missing image folder: {img_folder}")
    if not ann_file.is_file():
        raise FileNotFoundError(f"Missing annotation file: {ann_file}")

    cat2contiguous, num_classes = _load_coco_category_map(ann_file)

    # Set num_classes only if not already set (avoid overwriting between train/val)
    if "num_classes" not in config:
        config["num_classes"] = num_classes

    transforms = make_coco_transforms(image_set, image_size=int(config.get("image_size", 640)))
    return CocoDetectionOVOD(img_folder, ann_file, transforms, cat2contiguous)
