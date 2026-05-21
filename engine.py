import math
import sys
from typing import Dict, Iterable, Optional

import torch

from utils.eval_metrics import DetectionEvalAccumulator, relation_recall_counts_batch


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0.0,
    print_freq: int = 50,
    scaler: torch.amp.GradScaler = None,
) -> Dict[str, float]:
    model.train()
    criterion.train()

    use_amp = device.type == "cuda"
    if scaler is None and use_amp:
        scaler = torch.amp.GradScaler("cuda")

    loss_meters: Dict[str, float] = {}
    n_batches = 0

    for step, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(
                loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
            )

        if not math.isfinite(losses.item()):
            print("Loss is {}, stopping training".format(losses.item()))
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(losses).backward()
            if max_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if max_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        n_batches += 1
        for k, v in loss_dict.items():
            loss_meters[k] = loss_meters.get(k, 0.0) + float(v.detach().cpu())

        if print_freq and (step + 1) % print_freq == 0:
            print(f"Epoch {epoch} step {step+1}/{len(data_loader)} loss {losses.item():.4f}")

    if n_batches > 0:
        for k in list(loss_meters.keys()):
            loss_meters[k] /= float(n_batches)
    return loss_meters


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessors,
    data_loader: Iterable,
    device: torch.device,
    output_dir: str,
    print_freq: int = 100,
    compute_map: bool = True,
    num_classes: int = 80,
    num_rel_predicates: int = 0,
    rel_recall_k: int = 50,
) -> Dict[str, float]:
    model.eval()
    criterion.eval()

    loss_meters: Dict[str, float] = {}
    n_batches = 0

    det_accum: Optional[DetectionEvalAccumulator] = None
    if compute_map:
        det_accum = DetectionEvalAccumulator(num_classes=num_classes)
    rel_hits = 0
    rel_total = 0

    use_amp = device.type == "cuda"

    for step, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(
                loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
            )

        n_batches += 1
        for k, v in loss_dict.items():
            loss_meters[k] = loss_meters.get(k, 0.0) + float(v.detach().cpu())

        if det_accum is not None:
            det_accum.add_batch(outputs, targets)
        h, t = relation_recall_counts_batch(
            outputs, targets, k=rel_recall_k, num_rel_predicates=num_rel_predicates
        )
        rel_hits += h
        rel_total += t

        if print_freq and (step + 1) % print_freq == 0:
            print(f"Val step {step+1}/{len(data_loader)} loss {losses.item():.4f}")

    if n_batches > 0:
        for k in list(loss_meters.keys()):
            loss_meters[k] /= float(n_batches)

    if det_accum is not None:
        loss_meters["mAP50"] = det_accum.compute_mean_ap()
    if rel_total > 0:
        loss_meters[f"R@{rel_recall_k}"] = rel_hits / float(rel_total)

    return loss_meters
