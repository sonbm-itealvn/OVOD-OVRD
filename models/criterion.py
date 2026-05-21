"""
Loss functions and Hungarian Matcher for OVOD + OVRD.

Includes:
  - HungarianMatcher (DETR-style bipartite matching)
  - SetCriterion with:
      * Focal Loss for object classification
      * L1 + GIoU for bounding box regression
      * Relation matching + predicate CE + pointer CE
      * InfoNCE / CLIP contrastive alignment loss
      * Auxiliary (deep supervision) losses for intermediate decoder layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from utils.misc import get_world_size, is_dist_avail_and_initialized


# ---------------------------------------------------------------------------
#  Focal Loss
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Sigmoid focal loss (binary per-class, as in Deformable-DETR / DINO).

    Args:
        inputs:  (N, C) raw logits
        targets: (N, C) one-hot or multi-hot float targets
    """
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _tgt_to_pred_map(pred_idx: torch.Tensor, tgt_idx: torch.Tensor, num_tgt: int, device):
    m = torch.full((num_tgt,), -1, dtype=torch.long, device=device)
    if pred_idx.numel() > 0:
        m[tgt_idx] = pred_idx
    return m


# ---------------------------------------------------------------------------
#  Hungarian Matcher
# ---------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    """Bipartite matching between object queries and GT boxes (DETR-style)."""

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0,
                 use_focal: bool = True, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        out_logits = outputs["pred_logits"]
        out_bbox = outputs["pred_boxes"]
        bs = out_bbox.shape[0]

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]
            if tgt_ids.numel() == 0:
                empty = torch.as_tensor([], dtype=torch.long, device=out_bbox.device)
                indices.append((empty, empty))
                continue

            if self.use_focal:
                # Focal-loss-based cost (sigmoid)
                out_prob = out_logits[b].sigmoid()
                neg_cost_class = (1 - self.focal_alpha) * (out_prob ** self.focal_gamma) * \
                                 (-(1 - out_prob + 1e-8).log())
                pos_cost_class = self.focal_alpha * ((1 - out_prob) ** self.focal_gamma) * \
                                 (-(out_prob + 1e-8).log())
                num_classes = out_logits.shape[-1]
                tgt_one_hot = F.one_hot(tgt_ids.long(), num_classes).float()  # (M, C)
                cost_class = torch.matmul(pos_cost_class, tgt_one_hot.T) + \
                             torch.matmul(neg_cost_class, (1 - tgt_one_hot).T)
            else:
                out_prob = out_logits[b].softmax(-1)
                cost_class = -out_prob[:, tgt_ids.long()]

            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[b]),
                box_cxcywh_to_xyxy(tgt_bbox),
            )

            C = (self.cost_bbox * cost_bbox +
                 self.cost_class * cost_class +
                 self.cost_giou * cost_giou)
            C = C.detach().cpu().numpy()
            row, col = linear_sum_assignment(C)
            indices.append(
                (
                    torch.as_tensor(row, dtype=torch.long, device=out_bbox.device),
                    torch.as_tensor(col, dtype=torch.long, device=out_bbox.device),
                )
            )
        return indices


# ---------------------------------------------------------------------------
#  Set Criterion
# ---------------------------------------------------------------------------

class SetCriterion(nn.Module):
    """
    Object detection + relation detection losses with optional auxiliary
    deep-supervision from intermediate decoder layers.
    """

    def __init__(self, num_classes, matcher, weight_dict, losses,
                 eos_coef: float = 0.1,
                 num_rel_predicates: int = 0,
                 use_focal: bool = True,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.eos_coef = eos_coef
        self.num_rel_predicates = int(num_rel_predicates or 0)
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Class weight for softmax CE (only used when use_focal=False)
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    # ---- Classification ----

    def loss_labels(self, outputs, targets, indices, num_boxes):
        logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)], dim=0
        )

        if self.use_focal:
            # Sigmoid Focal Loss
            target_one_hot = torch.zeros(
                logits.shape[0], logits.shape[1], logits.shape[2],
                dtype=logits.dtype, device=logits.device,
            )
            if target_classes.numel() > 0:
                target_one_hot[idx[0], idx[1], target_classes.long()] = 1.0

            loss_ce = sigmoid_focal_loss(
                logits.flatten(0, 1),
                target_one_hot.flatten(0, 1),
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction="sum",
            ) / num_boxes.clamp(min=1.0)
        else:
            # Softmax CE with class weights
            target_classes_full = torch.full(
                logits.shape[:2], self.num_classes, dtype=torch.int64, device=logits.device
            )
            if target_classes.numel() > 0:
                target_classes_full[idx] = target_classes
            loss_ce = F.cross_entropy(
                logits.flatten(0, 1),
                target_classes_full.flatten(),
                weight=self.empty_weight,
                reduction="mean",
            )

        return {"loss_ce": loss_ce}

    # ---- Bounding boxes ----

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            z = outputs["pred_boxes"].sum() * 0.0
            return {"loss_bbox": z, "loss_giou": z}

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes.clamp(min=1.0)

        giou = torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        loss_giou = (1.0 - giou).sum() / num_boxes.clamp(min=1.0)
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    # ---- Relations ----

    def loss_relations(self, outputs, targets, indices, num_boxes):
        device = outputs["pred_boxes"].device
        rel_logits = outputs["rel_logits"]
        sub_attn = outputs["sub_assignment"]
        obj_attn = outputs["obj_assignment"]

        if self.num_rel_predicates <= 0:
            return {"loss_rel": torch.zeros((), device=device)}

        num_rel_classes = rel_logits.shape[-1]
        if num_rel_classes <= 1:
            return {"loss_rel": torch.zeros((), device=device)}

        loss_p = torch.zeros((), device=device)
        loss_ptr = torch.zeros((), device=device)
        n_terms = 0

        for b, tgt in enumerate(targets):
            rels = tgt.get("relations")
            if rels is None or rels.numel() == 0:
                continue
            pred_idx, tgt_idx = indices[b]
            tgt_to_pred = _tgt_to_pred_map(
                pred_idx, tgt_idx, tgt["boxes"].shape[0], device
            )

            # Filter relations: keep only those whose BOTH subject AND object
            # have a matched prediction from object matching.
            # (If N_obj < N_gt, some GT objects will be unmatched → tgt_to_pred = -1)
            sub_ids_all = tgt_to_pred[rels[:, 0]]
            obj_ids_all = tgt_to_pred[rels[:, 1]]
            valid_mask = (sub_ids_all >= 0) & (obj_ids_all >= 0)
            if not valid_mask.any():
                continue

            valid_rels = rels[valid_mask]    # (R', 3)
            sub_ids = sub_ids_all[valid_mask]
            obj_ids = obj_ids_all[valid_mask]
            R = valid_rels.shape[0]
            Qr = rel_logits.shape[1]

            # Relation Hungarian matching
            with torch.no_grad():
                log_p = rel_logits[b].log_softmax(-1)
                log_sub = sub_attn[b].log_softmax(-1)
                log_obj = obj_attn[b].log_softmax(-1)
                cost = torch.zeros(Qr, R, device=device)
                for r in range(R):
                    ps, po, pid = sub_ids[r], obj_ids[r], valid_rels[r, 2].long()
                    if int(pid.item()) < 0 or int(pid.item()) >= num_rel_classes:
                        cost[:, r] = 1e8
                        continue
                    cost[:, r] = -log_p[:, pid] - log_sub[:, ps] - log_obj[:, po]
                cost = cost.detach().cpu().numpy()
                row, col = linear_sum_assignment(cost)

            row = torch.as_tensor(row, device=device, dtype=torch.long)
            col = torch.as_tensor(col, device=device, dtype=torch.long)
            if row.numel() == 0:
                continue

            matched = rel_logits[b][row]
            tgt_pred = valid_rels[col, 2].long()
            loss_p = loss_p + F.cross_entropy(matched, tgt_pred, reduction="sum")

            for k in range(row.shape[0]):
                r = int(col[k].item())
                q = int(row[k].item())
                ps, po = int(sub_ids[r].item()), int(obj_ids[r].item())
                pid = valid_rels[r, 2].long()
                if pid.item() < 0 or pid.item() >= num_rel_classes:
                    continue
                loss_ptr = loss_ptr + F.cross_entropy(
                    sub_attn[b, q : q + 1],
                    torch.tensor([ps], device=device),
                )
                loss_ptr = loss_ptr + F.cross_entropy(
                    obj_attn[b, q : q + 1],
                    torch.tensor([po], device=device),
                )
            n_terms += int(row.numel())

        if n_terms == 0:
            return {"loss_rel": torch.zeros((), device=device)}
        return {"loss_rel": (loss_p + loss_ptr) / float(n_terms)}

    # ---- Vision-Language contrastive alignment (InfoNCE / CLIP loss) ----

    def loss_vl_align(self, outputs, targets, indices, num_boxes):
        """
        InfoNCE contrastive loss between visual object embeddings and
        GT text embeddings (pre-computed from CLIP text encoder).

        Negative strategy: **all-class negatives**.
        Each visual embedding is contrasted against ALL class text embeddings
        (not just in-batch GT classes). This provides C negatives per sample
        regardless of batch size, avoiding the weak-negative problem when B is small.

        Requires targets to contain 'text_embed' key with shape
        (num_classes, clip_dim) — the L2-normalised CLIP text embeddings
        for all category names.
        """
        device = outputs["pred_logits"].device

        # Collect text embeddings from targets (all-class matrix)
        text_embed = None
        for t in targets:
            if "text_embed" in t:
                text_embed = t["text_embed"]
                break

        if text_embed is None:
            return {"loss_vl": torch.zeros((), device=device)}

        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return {"loss_vl": torch.zeros((), device=device)}

        # Dimension check: VL loss only applies in open-vocab mode
        # where pred_logits dim == clip_dim. In closed-vocab mode
        # pred_logits is (B, N, num_classes+1) — skip.
        vis_dim = outputs["pred_logits"].shape[-1]
        txt_dim = text_embed.shape[-1]
        if vis_dim != txt_dim:
            return {"loss_vl": torch.zeros((), device=device)}

        # Visual embeddings of matched predictions
        vis_embeds = outputs["pred_logits"][idx]          # (M, clip_dim)
        vis_embeds = F.normalize(vis_embeds, dim=-1)

        # All class text embeddings as negatives
        all_text = F.normalize(text_embed, dim=-1)         # (C, clip_dim)

        # GT labels for positive matching
        gt_labels = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)], dim=0
        ).long()  # (M,)

        # Get temperature
        temperature = outputs.get("temperature", torch.tensor(0.07, device=device))

        # Similarity: each visual embed vs ALL class text embeds → (M, C)
        logits = torch.matmul(vis_embeds, all_text.T) / temperature

        # Each matched prediction's GT class is the positive; rest are negatives
        loss_vl = F.cross_entropy(logits, gt_labels)

        return {"loss_vl": loss_vl}

    # ---- Predicate Vision-Language contrastive loss (open-vocab OVRD) ----

    def loss_vl_pred(self, outputs, targets, indices, num_boxes):
        """
        Contrastive loss for predicate embeddings vs CLIP predicate text embeddings.
        Mirrors loss_vl_align but for the relation branch.

        In open-vocab mode, rel_logits are (B, N_rel, clip_dim) embeddings.
        Each matched predicate embedding is contrasted against ALL P predicate
        text embeddings to learn alignment with CLIP text space.

        Requires targets to contain 'pred_text_embed' key with shape
        (P, clip_dim) — L2-normalised CLIP text embeddings for predicate names.
        """
        device = outputs["rel_logits"].device

        # Collect predicate text embeddings
        pred_text_embed = None
        for t in targets:
            if "pred_text_embed" in t:
                pred_text_embed = t["pred_text_embed"]
                break

        if pred_text_embed is None:
            return {"loss_vl_pred": torch.zeros((), device=device)}

        # Dimension check: only in open-vocab mode (rel_logits dim == clip_dim)
        rel_dim = outputs["rel_logits"].shape[-1]
        txt_dim = pred_text_embed.shape[-1]
        if rel_dim != txt_dim:
            return {"loss_vl_pred": torch.zeros((), device=device)}

        # Get matched relation queries from relation matching
        # We need to find which relation queries are matched to GT predicates
        # Re-run relation matching logic (lightweight)
        rel_logits = outputs["rel_logits"]
        sub_attn = outputs["sub_assignment"]
        obj_attn = outputs["obj_assignment"]

        all_pred_embeds = []
        all_pred_labels = []

        for b, tgt in enumerate(targets):
            rels = tgt.get("relations")
            if rels is None or rels.numel() == 0:
                continue

            pred_idx, tgt_idx = indices[b]
            tgt_to_pred = _tgt_to_pred_map(
                pred_idx, tgt_idx, tgt["boxes"].shape[0], device
            )

            sub_ids_all = tgt_to_pred[rels[:, 0]]
            obj_ids_all = tgt_to_pred[rels[:, 1]]
            valid_mask = (sub_ids_all >= 0) & (obj_ids_all >= 0)
            if not valid_mask.any():
                continue

            valid_rels = rels[valid_mask]
            sub_ids = sub_ids_all[valid_mask]
            obj_ids = obj_ids_all[valid_mask]
            R = valid_rels.shape[0]
            Qr = rel_logits.shape[1]

            # Quick matching (reuse logic from loss_relations)
            with torch.no_grad():
                num_pred_classes = pred_text_embed.shape[0]
                log_sub = sub_attn[b].log_softmax(-1)
                log_obj = obj_attn[b].log_softmax(-1)
                cost = torch.zeros(Qr, R, device=device)
                for r in range(R):
                    ps, po = sub_ids[r], obj_ids[r]
                    cost[:, r] = -log_sub[:, ps] - log_obj[:, po]
                row, col = linear_sum_assignment(cost.detach().cpu().numpy())

            row = torch.as_tensor(row, device=device, dtype=torch.long)
            col = torch.as_tensor(col, device=device, dtype=torch.long)
            if row.numel() == 0:
                continue

            matched_embeds = rel_logits[b][row]  # (K, clip_dim)
            matched_labels = valid_rels[col, 2].long()  # (K,)

            all_pred_embeds.append(matched_embeds)
            all_pred_labels.append(matched_labels)

        if len(all_pred_embeds) == 0:
            return {"loss_vl_pred": torch.zeros((), device=device)}

        vis_embeds = F.normalize(torch.cat(all_pred_embeds, dim=0), dim=-1)
        gt_labels = torch.cat(all_pred_labels, dim=0)
        all_text = F.normalize(pred_text_embed, dim=-1)

        # Temperature from RelationHead
        temperature = outputs.get("rel_temperature", torch.tensor(0.07, device=device))

        # Contrastive: each predicate embed vs ALL predicate text embeds → (K, P)
        logits = torch.matmul(vis_embeds, all_text.T) / temperature
        loss_vl_pred = F.cross_entropy(logits, gt_labels)

        return {"loss_vl_pred": loss_vl_pred}

    # ---- Helpers ----

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
            "relations": self.loss_relations,
            "vl": self.loss_vl_align,
            "vl_pred": self.loss_vl_pred,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        # --- Main losses on final decoder layer ---
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=outputs["pred_logits"].device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
            num_boxes = num_boxes / float(get_world_size())
        num_boxes = torch.clamp(num_boxes / max(len(targets), 1), min=1.0)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # --- Auxiliary losses on intermediate decoder layers ---
        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux, targets, aux_indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
