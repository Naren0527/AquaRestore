"""Dataset classes and transformations for the aquatic image classifier."""

import os
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from aquarestore.classification.config import ClassifierConfig


def get_train_transforms(config: ClassifierConfig | None = None) -> Callable:
    """Build training data augmentation pipeline."""
    cfg = config or ClassifierConfig()
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(
                cfg.image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )


def get_eval_transforms(config: ClassifierConfig | None = None) -> Callable:
    """Build evaluation / inference preprocessing pipeline."""
    cfg = config or ClassifierConfig()
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )


class AquaticDataset(Dataset):
    """PyTorch Dataset for aquatic vs non-aquatic image classification."""

    def __init__(
        self,
        samples: Sequence[tuple[str | Path, int]],
        transform: Callable | None = None,
    ) -> None:
        self.samples = [(Path(p), int(label)) for p, label in samples]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                image_rgb = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Corrupt or unreadable image at {path}: {e}")

        if self.transform is not None:
            tensor = self.transform(image_rgb)
        else:
            tensor = transforms.ToTensor()(image_rgb)

        return tensor, label, str(path)

    @classmethod
    def from_directory(
        cls,
        split_dir: Path | str,
        transform: Callable | None = None,
    ) -> "AquaticDataset":
        """Load dataset from standard directory layout: <split_dir>/non_aquatic and <split_dir>/aquatic."""
        split_path = Path(split_dir)
        samples = []

        class_mapping = {"non_aquatic": 0, "aquatic": 1}
        for class_name, label in class_mapping.items():
            class_folder = split_path / class_name
            if not class_folder.exists():
                continue
            for fname in os.listdir(class_folder):
                fpath = class_folder / fname
                if fpath.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    samples.append((fpath, label))

        if not samples:
            raise ValueError(f"No valid images found in {split_dir}")

        return cls(samples=samples, transform=transform)

