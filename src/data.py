import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class ImageDataset(Dataset[tuple[torch.Tensor, int]]):
    VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".jfif")

    def __init__(
        self,
        root: str | Path,
        transform: transforms.Compose | None = None,
        class_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted(d for d in os.listdir(root) if (self.root / d).is_dir())
        self.class_to_idx = class_to_idx or {c: i for i, c in enumerate(self.classes)}

        self.paths: list[Path] = []
        self.labels: list[int] = []
        for cls in self.classes:
            cls_dir = self.root / cls
            for f in os.listdir(cls_dir):
                if f.lower().endswith(self.VALID_EXT):
                    self.paths.append(cls_dir / f)
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, self.labels[idx]

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def get_train_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_val_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.from_numpy(weights)


def create_dataloaders(
    data_root: str | Path,
    batch_size: int = 32,
    train_split: float = 0.9,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    img_size: int = 224,
) -> tuple[
    DataLoader[tuple[torch.Tensor, int]], DataLoader[tuple[torch.Tensor, int]], dict[str, Any]
]:
    full = ImageDataset(data_root, transform=get_val_transform(img_size))
    labels = full.labels

    indices = np.arange(len(full))
    train_idx, val_idx = train_test_split(
        indices, train_size=train_split, stratify=labels, random_state=seed
    )

    train_ds = ImageDataset(
        data_root, transform=get_train_transform(img_size), class_to_idx=full.class_to_idx
    )
    val_ds = ImageDataset(
        data_root, transform=get_val_transform(img_size), class_to_idx=full.class_to_idx
    )

    persistent = num_workers > 0
    train_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        Subset(train_ds, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent,
    )
    val_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        Subset(val_ds, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    train_labels = [labels[i] for i in train_idx]
    class_weights = compute_class_weights(train_labels, full.num_classes)

    return (
        train_loader,
        val_loader,
        {
            "num_classes": full.num_classes,
            "classes": full.classes,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "class_weights": class_weights,
        },
    )
