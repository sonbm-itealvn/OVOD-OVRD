import torch
from torch.utils.data import DataLoader, DistributedSampler

from .datasets import build_dataset


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, targets


def build_data_loader(dataset, batch_size, is_train=True, distributed=False, num_workers=2):
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=is_train)
    else:
        sampler = torch.utils.data.RandomSampler(dataset) if is_train else torch.utils.data.SequentialSampler(dataset)

    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=is_train)

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_dataloaders_from_config(config, distributed=False):
    train_ds = build_dataset("train", config)
    val_ds = build_dataset("val", config)
    train_loader = build_data_loader(
        train_ds,
        int(config["batch_size"]),
        is_train=True,
        distributed=distributed,
        num_workers=int(config.get("num_workers", 2)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=int(config.get("num_workers", 2)),
        pin_memory=True,
    )
    return train_loader, val_loader
