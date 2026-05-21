import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler, Subset

from data.datasets import build_dataset
from data.data_loaders import collate_fn
from engine import evaluate, train_one_epoch
from models import build_model
from models.criterion import HungarianMatcher, SetCriterion
from utils.device_utils import apply_profile, describe_hardware, dataloader_kwargs, resolve_device


def get_args_parser():
    parser = argparse.ArgumentParser("OVOD + OVRD training", add_help=True)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr_drop", default=40, type=int, help="StepLR step_size (epochs).")
    parser.add_argument("--clip_max_norm", default=0.1, type=float, help="Gradient clipping max norm.")

    parser.add_argument("--backbone", default="swin_tiny_patch4_window7_224", type=str)
    parser.add_argument("--pretrained", default=1, type=int, choices=[0, 1], help="1 loads timm pretrained weights.")
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--nheads", default=8, type=int)
    parser.add_argument("--enc_layers", default=6, type=int)
    parser.add_argument("--dec_layers", default=6, type=int)
    parser.add_argument("--dim_feedforward", default=1024, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--num_obj_queries", default=100, type=int)
    parser.add_argument("--num_rel_queries", default=50, type=int)
    parser.add_argument("--num_rel_predicates", default=0, type=int, help="0 disables supervised relation CE.")
    parser.add_argument("--clip_dim", default=512, type=int)
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32", type=str,
                        help="HuggingFace CLIP model name for text encoder.")

    # Focal Loss
    parser.add_argument("--use_focal", default=1, type=int, choices=[0, 1],
                        help="1: use sigmoid focal loss; 0: softmax CE.")
    parser.add_argument("--focal_alpha", default=0.25, type=float)
    parser.add_argument("--focal_gamma", default=2.0, type=float)

    # Auxiliary losses
    parser.add_argument("--use_aux_loss", default=1, type=int, choices=[0, 1],
                        help="1: deep supervision from intermediate decoder layers.")

    # FPN level
    parser.add_argument("--fpn_level", default=-1, type=int,
                        help="Which FPN level to use as transformer input (-1=coarsest).")

    parser.add_argument(
        "--dataset_file",
        default="coco",
        type=str,
        choices=["coco", "vg_json", "vg150", "scene_graph"],
        help="coco | vg_json / vg150 / scene_graph (JSON scene graph, cùng loader).",
    )
    parser.add_argument(
        "--coco_path",
        type=str,
        default=None,
        help="Thư mục gốc COCO (train2017, val2017, annotations/). Bắt buộc nếu dataset_file=coco.",
    )
    parser.add_argument("--vg_img_root", type=str, default=None, help="Thư mục gốc ảnh cho scene_graph / VG JSON.")
    parser.add_argument("--vg_train_ann", type=str, default=None, help="JSON train (list hoặc COCO-style + relations).")
    parser.add_argument("--vg_val_ann", type=str, default=None, help="JSON val.")
    parser.add_argument("--image_size", default=640, type=int)
    parser.add_argument("--num_workers", default=2, type=int)

    parser.add_argument("--eval_map", default=1, type=int, choices=[0, 1], help="1: tính mAP50 + R@K trên val.")
    parser.add_argument("--rel_recall_k", default=50, type=int, help="K cho R@K (quan hệ).")

    # Loss weights
    parser.add_argument("--w_ce", default=1.0, type=float, help="Weight for classification loss.")
    parser.add_argument("--w_bbox", default=5.0, type=float, help="Weight for L1 bbox loss.")
    parser.add_argument("--w_giou", default=2.0, type=float, help="Weight for GIoU loss.")
    parser.add_argument("--w_rel", default=1.0, type=float, help="Weight for relation loss.")
    parser.add_argument("--w_vl", default=1.0, type=float, help="Weight for VL contrastive loss (objects).")
    parser.add_argument("--w_vl_pred", default=1.0, type=float, help="Weight for VL contrastive loss (predicates).")

    parser.add_argument("--output_dir", default="", help="Where to save checkpoints; empty skips saving.")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | cpu | cuda | mps — auto: CUDA → MPS → CPU.",
    )
    parser.add_argument(
        "--profile",
        default="auto",
        choices=["auto", "none", "cpu", "cpu_debug", "gpu"],
        help="Preset hyperparams theo máy: auto=cpu hoặc gpu; none=không đổi default.",
    )
    parser.add_argument(
        "--max_train_samples",
        default=0,
        type=int,
        help="0=tất cả; >0 chỉ lấy N ảnh train (debug CPU).",
    )
    parser.add_argument(
        "--max_val_samples",
        default=0,
        type=int,
        help="0=tất cả; >0 chỉ lấy N ảnh val (debug CPU).",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--print_freq", default=50, type=int)
    return parser


def validate_dataset_args(args):
    d = str(args.dataset_file).lower()
    if d == "coco":
        if not args.coco_path:
            raise SystemExit("Thiếu --coco_path (bắt buộc khi --dataset_file coco).")
    elif d in ("vg_json", "vg150", "scene_graph"):
        if not args.vg_img_root or not args.vg_train_ann or not args.vg_val_ann:
            raise SystemExit(
                "Scene graph / VG: cần đủ --vg_img_root, --vg_train_ann, --vg_val_ann "
                f"(dataset_file={args.dataset_file})."
            )
    else:
        raise SystemExit(f"dataset_file không hợp lệ: {args.dataset_file}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_subset(dataset, max_samples: int):
    if max_samples and max_samples > 0:
        n = min(max_samples, len(dataset))
        return Subset(dataset, list(range(n)))
    return dataset


def main(args):
    validate_dataset_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.device = str(device)

    config = dict(vars(args))
    config["pretrained"] = bool(args.pretrained)
    config["use_focal"] = bool(args.use_focal)
    config["use_aux_loss"] = bool(args.use_aux_loss)

    dataset_train = _maybe_subset(build_dataset("train", config), int(args.max_train_samples))
    dataset_val = _maybe_subset(build_dataset("val", config), int(args.max_val_samples))

    model = build_model(config)
    model.to(device)

    # --- Matcher & Criterion ---
    use_focal = bool(args.use_focal)
    matcher = HungarianMatcher(use_focal=use_focal,
                               focal_alpha=args.focal_alpha,
                               focal_gamma=args.focal_gamma)

    weight_dict = {
        "loss_ce": args.w_ce,
        "loss_bbox": args.w_bbox,
        "loss_giou": args.w_giou,
        "loss_rel": args.w_rel,
        "loss_vl": args.w_vl,
        "loss_vl_pred": args.w_vl_pred,
    }

    # Auxiliary loss weights (same as main, for each intermediate layer)
    if bool(args.use_aux_loss):
        num_aux = args.dec_layers - 1
        for i in range(num_aux):
            for k, v in list(weight_dict.items()):
                weight_dict[f"{k}_{i}"] = v

    losses = ["labels", "boxes", "relations", "vl", "vl_pred"]
    criterion = SetCriterion(
        num_classes=int(config["num_classes"]),
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        eos_coef=0.1,
        num_rel_predicates=int(config.get("num_rel_predicates", 0) or 0),
        use_focal=use_focal,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )
    criterion.to(device)

    # --- Optimiser ---
    # Use different LR for backbone (lower) vs rest
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    param_groups = [
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": other_params, "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)

    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    batch_sampler_train = BatchSampler(sampler_train, args.batch_size, drop_last=True)

    dl_kw = dataloader_kwargs(device, args.num_workers)
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=collate_fn,
        **dl_kw,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=collate_fn,
        **dl_kw,
    )

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(describe_hardware(device, getattr(args, "_profile_name", args.profile)))
    print(f"Train images: {len(dataset_train)} | Val images: {len(dataset_val)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Using focal loss: {use_focal} | Aux loss: {bool(args.use_aux_loss)}")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            max_norm=args.clip_max_norm,
            print_freq=args.print_freq,
        )
        lr_scheduler.step()
        dt = time.time() - t0
        if train_stats:
            ce = train_stats.get("loss_ce", 0.0)
            bbox = train_stats.get("loss_bbox", 0.0)
            giou = train_stats.get("loss_giou", 0.0)
            rel = train_stats.get("loss_rel", 0.0)
            vl = train_stats.get("loss_vl", 0.0)
            print(
                f"Epoch {epoch} done in {dt:.1f}s | "
                f"loss_ce {ce:.4f} loss_bbox {bbox:.4f} loss_giou {giou:.4f} "
                f"loss_rel {rel:.4f} loss_vl {vl:.4f}"
            )

        if args.output_dir:
            checkpoint_path = Path(args.output_dir) / f"checkpoint{epoch:04}.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": dict(vars(args)),
                    "num_classes": int(config["num_classes"]),
                },
                checkpoint_path,
            )

        val_stats = evaluate(
            model,
            criterion,
            None,
            data_loader_val,
            device,
            args.output_dir,
            print_freq=max(args.print_freq, 100),
            compute_map=bool(args.eval_map),
            num_classes=int(config["num_classes"]),
            num_rel_predicates=int(config.get("num_rel_predicates", 0) or 0),
            rel_recall_k=int(args.rel_recall_k),
        )
        if val_stats:
            extra = ""
            if "mAP50" in val_stats:
                extra += f" mAP50 {val_stats['mAP50']:.4f}"
            rk = f"R@{args.rel_recall_k}"
            if rk in val_stats:
                extra += f" {rk} {val_stats[rk]:.4f}"
            print(
                f"Val epoch {epoch} | loss_ce {val_stats.get('loss_ce', 0):.4f} "
                f"loss_bbox {val_stats.get('loss_bbox', 0):.4f} loss_giou {val_stats.get('loss_giou', 0):.4f} "
                f"loss_rel {val_stats.get('loss_rel', 0):.4f}{extra}"
            )


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    args._profile_name = apply_profile(args, parser)
    main(args)
