"""
Đánh giá detection (mAP@0.5 đơn giản) và quan hệ (R@K) trên tập val.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from utils.box_ops import box_cxcywh_to_xyxy, box_iou


def _xyxy_norm_from_cxcywh(cxcywh: torch.Tensor) -> torch.Tensor:
    """cxcywh đã chuẩn hóa [0,1] -> xyxy cùng không gian."""
    return box_cxcywh_to_xyxy(cxcywh.clamp(0.0, 1.0))


def average_precision_from_scores(scores: List[float], tps: List[int], num_gt: int) -> float:
    """AP kiểu VOC: sort theo score giảm, tích lũy precision/recall."""
    if num_gt == 0 or len(scores) == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp_cum = 0
    fp_cum = 0
    precisions = []
    recalls = []
    for i in order:
        if tps[i]:
            tp_cum += 1
        else:
            fp_cum += 1
        prec = tp_cum / max(tp_cum + fp_cum, 1)
        rec = tp_cum / float(num_gt)
        precisions.append(prec)
        recalls.append(rec)
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * max(0.0, r - prev_r)
        prev_r = r
    return float(ap)


class DetectionEvalAccumulator:
    """Gom dự đoán theo lớp để tính mAP@iou (mặc định 0.5)."""

    def __init__(self, num_classes: int, iou_thresh: float = 0.5, score_thresh: float = 0.05):
        self.num_classes = int(num_classes)
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self._reset()

    def _reset(self):
        self._per_class: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
        self._gt_count = defaultdict(int)

    def add_batch(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        prob = outputs["pred_logits"].softmax(-1)
        fg = prob[:, :, :-1]
        scores, labels = fg.max(-1)
        pred_boxes = outputs["pred_boxes"]
        bsz = pred_boxes.shape[0]

        for b in range(bsz):
            tgt = targets[b]
            gt_boxes_c = tgt["boxes"]
            gt_labs = tgt["labels"]
            if gt_labs.numel() == 0:
                continue
            gt_xy = _xyxy_norm_from_cxcywh(gt_boxes_c)

            for c in range(self.num_classes):
                mask_gt = gt_labs == c
                num_gtc = int(mask_gt.sum().item())
                if num_gtc == 0:
                    continue
                self._gt_count[c] += num_gtc

                gt_b = gt_xy[mask_gt]
                gt_used = torch.zeros(gt_b.shape[0], dtype=torch.bool, device=gt_b.device)

                sc = scores[b]
                pl = labels[b]
                pb = pred_boxes[b]
                pred_mask = (pl == c) & (sc >= self.score_thresh)
                if not pred_mask.any():
                    continue
                idxs = torch.where(pred_mask)[0]
                idxs = idxs[torch.argsort(sc[idxs], descending=True)]

                for pi in idxs:
                    box_p = _xyxy_norm_from_cxcywh(pb[pi : pi + 1])
                    ious, _ = box_iou(box_p, gt_b)
                    best_j = int(ious.argmax())
                    if float(ious[0, best_j]) >= self.iou_thresh and not gt_used[best_j]:
                        gt_used[best_j] = True
                        self._per_class[c].append((float(sc[pi].item()), 1))
                    else:
                        self._per_class[c].append((float(sc[pi].item()), 0))

    def compute_mean_ap(self) -> float:
        aps = []
        for c in range(self.num_classes):
            dets = self._per_class[c]
            num_gt = self._gt_count[c]
            if num_gt == 0:
                continue
            scores = [d[0] for d in dets]
            tps = [d[1] for d in dets]
            aps.append(average_precision_from_scores(scores, tps, num_gt))
        if len(aps) == 0:
            return 0.0
        return float(sum(aps) / len(aps))


@torch.no_grad()
def relation_recall_counts_batch(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    k: int = 50,
    iou_thresh: float = 0.5,
    num_rel_predicates: int = 0,
) -> Tuple[int, int]:
    """
    Trả về (hits, total_gt_relations) trên batch để cộng dồn micro R@K toàn tập val.
    """
    if num_rel_predicates <= 0:
        return 0, 0
    num_cls = outputs["rel_logits"].shape[-1]
    if num_cls <= 1:
        return 0, 0

    rel_logits = outputs["rel_logits"]
    sub_a = outputs["sub_assignment"]
    obj_a = outputs["obj_assignment"]
    boxes = outputs["pred_boxes"]
    bsz = boxes.shape[0]

    hits = 0
    total = 0

    for b in range(bsz):
        tgt = targets[b]
        rels = tgt.get("relations")
        if rels is None or rels.numel() == 0:
            continue
        gt_boxes = _xyxy_norm_from_cxcywh(tgt["boxes"])
        if gt_boxes.numel() == 0:
            continue

        rl = rel_logits[b]
        ps = sub_a[b].softmax(-1)
        po = obj_a[b].softmax(-1)
        pc = rl.softmax(-1)
        nq = rl.shape[0]
        nobj = boxes.shape[1]
        if nobj <= 1:
            continue

        triple_scores = []
        for qi in range(nq):
            si = int(ps[qi].argmax().item())
            oi_scores = po[qi].clone()
            oi_scores[si] = -1e9
            oi = int(oi_scores.argmax().item())
            pid = int(pc[qi].argmax().item())
            s = float(pc[qi, pid] * ps[qi, si] * po[qi, oi])
            triple_scores.append((s, qi, si, oi, pid))

        triple_scores.sort(key=lambda x: -x[0])
        top = triple_scores[:k]

        matched_gt = torch.zeros(rels.shape[0], dtype=torch.bool, device=rels.device)
        for gt_i in range(rels.shape[0]):
            if matched_gt[gt_i]:
                continue
            si_gt, oi_gt, pid_gt = rels[gt_i].tolist()
            if si_gt >= gt_boxes.shape[0] or oi_gt >= gt_boxes.shape[0]:
                continue
            b_sub = gt_boxes[si_gt : si_gt + 1]
            b_obj = gt_boxes[oi_gt : oi_gt + 1]
            for _, _, si, oi, pid in top:
                if int(pid) != int(pid_gt):
                    continue
                bs_p = _xyxy_norm_from_cxcywh(boxes[b, si : si + 1])
                bo_p = _xyxy_norm_from_cxcywh(boxes[b, oi : oi + 1])
                iou_s, _ = box_iou(bs_p, b_sub)
                iou_o, _ = box_iou(bo_p, b_obj)
                if float(iou_s[0, 0]) >= iou_thresh and float(iou_o[0, 0]) >= iou_thresh:
                    matched_gt[gt_i] = True
                    break

        # FIX BUG-2: moved OUTSIDE the for-gt_i loop (was incorrectly indented inside)
        hits += int(matched_gt.sum().item())
        total += int(rels.shape[0])

    return hits, total
