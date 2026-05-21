"""
Chọn device và profile training theo phần cứng (CPU / CUDA / MPS).
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

import torch


# Các preset — chỉ ghi đè tham số khi user giữ giá trị mặc định của argparse
PROFILE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "cpu": {
        "device": "cpu",
        "backbone": "resnet50",
        "batch_size": 1,
        "image_size": 320,
        "num_obj_queries": 30,
        "num_rel_queries": 15,
        "enc_layers": 3,
        "dec_layers": 3,
        "dim_feedforward": 512,
        "num_workers": 0,
        "use_aux_loss": 0,
        "pretrained": 1,
        "epochs": 5,
        "print_freq": 10,
    },
    "cpu_debug": {
        "device": "cpu",
        "backbone": "resnet50",
        "pretrained": 0,
        "batch_size": 1,
        "image_size": 224,
        "num_obj_queries": 10,
        "num_rel_queries": 5,
        "enc_layers": 2,
        "dec_layers": 2,
        "dim_feedforward": 512,
        "num_workers": 0,
        "use_aux_loss": 0,
        "epochs": 1,
        "eval_map": 0,
        "max_train_samples": 32,
        "max_val_samples": 16,
        "print_freq": 1,
    },
    "gpu": {
        "device": "cuda",
        "batch_size": 2,
        "image_size": 640,
        "num_workers": 2,
        "use_aux_loss": 1,
    },
}


def cuda_available() -> bool:
    return torch.cuda.is_available()


def mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def resolve_device(device: str) -> torch.device:
    """
    device: 'auto' | 'cpu' | 'cuda' | 'cuda:0' | 'mps'
    """
    d = (device or "auto").strip().lower()
    if d == "auto":
        if cuda_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if d == "cuda" and not cuda_available():
        print("[device] CUDA not available, using CPU.")
        return torch.device("cpu")
    if d == "mps" and not mps_available():
        print("[device] MPS not available, using CPU.")
        return torch.device("cpu")
    return torch.device(device)


def resolve_profile_name(profile: str, device: torch.device) -> str:
    p = (profile or "auto").strip().lower()
    if p != "auto":
        return p
    if device.type == "cuda":
        return "gpu"
    return "cpu"


def parser_defaults(parser: argparse.ArgumentParser) -> Dict[str, Any]:
    return {
        a.dest: a.default
        for a in parser._actions
        if a.dest not in ("help", "options")
    }


def apply_profile(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """
    Áp preset theo --profile; không ghi đè tham số user đã đổi trên CLI.
    Trả về tên profile thực tế (cpu / cpu_debug / gpu / none).
    """
    defaults = parser_defaults(parser)
    name = resolve_profile_name(getattr(args, "profile", "auto"), resolve_device(args.device))
    if name in ("none", "off"):
        return "none"

    overrides = PROFILE_OVERRIDES.get(name, {})
    for key, value in overrides.items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) == defaults.get(key):
            setattr(args, key, value)

    # Đồng bộ device sau preset
    args.device = str(resolve_device(args.device))
    return name


def describe_hardware(device: torch.device, profile: str) -> str:
    lines = [f"Device: {device} (type={device.type})"]
    if device.type == "cuda":
        lines.append(f"GPU: {torch.cuda.get_device_name(device)}")
    lines.append(f"Profile: {profile}")
    if profile.startswith("cpu"):
        lines.append(
            "CPU hint: use --profile cpu_debug or --max_train_samples 500 for a quick run."
        )
    return "\n".join(lines)


def dataloader_kwargs(device: torch.device, num_workers: int) -> Dict[str, Any]:
    """pin_memory chỉ bật trên CUDA; Windows CPU thường dùng num_workers=0."""
    kw = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return kw
