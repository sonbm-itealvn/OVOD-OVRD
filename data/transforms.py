import random
import torch
import torchvision.transforms.functional as F

from utils.box_ops import box_xyxy_to_cxcywh


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, size):
        self.size = size if isinstance(size, (list, tuple)) else (size, size)

    def __call__(self, image, target):
        old_w, old_h = image.size
        new_h, new_w = int(self.size[0]), int(self.size[1])
        image = F.resize(image, (new_h, new_w))
        if "boxes" in target and target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            sx = new_w / float(old_w)
            sy = new_h / float(old_h)
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes[:, 0] /= new_w
            boxes[:, 1] /= new_h
            boxes[:, 2] /= new_w
            boxes[:, 3] /= new_h
            boxes = boxes.clamp(0.0, 1.0)
            target["boxes"] = boxes
        target["size"] = torch.as_tensor([new_h, new_w], dtype=torch.int64)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        image = F.hflip(image)
        if "boxes" in target and target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            # boxes are cxcywh normalised: flip cx
            boxes[:, 0] = 1.0 - boxes[:, 0]
            target["boxes"] = boxes
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class EnsureCxcywhNormalized(object):
    """
    Safety transform: verify boxes are in cxcywh normalised [0,1] format.
    If boxes appear to be in absolute xyxy (values > 1), convert them.
    Applied as the last geometric transform before ToTensor.
    """

    def __call__(self, image, target):
        if "boxes" not in target or target["boxes"].numel() == 0:
            return image, target
        boxes = target["boxes"]
        # Heuristic: if any coordinate > 1.0 the boxes are probably absolute xyxy
        if boxes.max() > 1.0:
            if hasattr(image, "size"):
                w, h = image.size  # PIL
            else:
                h, w = image.shape[-2:]  # tensor
            # Assume xyxy absolute → convert
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes[:, 0] /= w
            boxes[:, 1] /= h
            boxes[:, 2] /= w
            boxes[:, 3] /= h
            boxes = boxes.clamp(0.0, 1.0)
            target["boxes"] = boxes
        return image, target


def make_coco_transforms(image_set, image_size=640):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = Normalize(mean=mean, std=std)
    size = (image_size, image_size)

    if image_set == "train":
        return Compose(
            [
                Resize(size),
                RandomHorizontalFlip(),
                EnsureCxcywhNormalized(),
                ToTensor(),
                normalize,
            ]
        )
    if image_set in ("val", "test"):
        return Compose([Resize(size), EnsureCxcywhNormalized(), ToTensor(), normalize])
    raise ValueError(f"unknown image_set {image_set}")
