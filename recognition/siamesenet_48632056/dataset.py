from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REQUIRED_COLUMNS: set[str] = {
    "image_name",
    "patient_id",
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
    "diagnosis",
    "benign_malignant",
    "target",
    "split",
}


def build_transforms() -> Tuple[A.Compose, A.Compose]:
    """Return train/validation Albumentations pipelines."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = A.Compose(
        [
            # Basic flips and rotations to cover view changes.
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    val_tf = A.Compose(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return train_tf, val_tf


class ISICDataset(Dataset):
    """Simple Dataset wrapper around the ISIC metadata CSV."""

    def __init__(self, csv_path: str, images_dir: str, transform: A.Compose | None) -> None:
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.transform = transform

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        if not self.images_dir.is_dir():
            raise NotADirectoryError(f"Images directory not found: {self.images_dir}")
        if self.transform is None:
            raise ValueError("transform must be provided for ISICDataset.")

        df = pd.read_csv(self.csv_path)
        self._ensure_required_columns(df.columns)
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, str]:
        row = self.df.iloc[index]

        # Accept bare stems or filenames with suffix.
        image_name = str(row["image_name"])
        image_path = self._resolve_image_path(image_name)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Albumentations expects dict input, returns tensor.
        transformed = self.transform(image=image)
        image_tensor = transformed["image"]

        target = int(row["target"])
        patient_id = str(row["patient_id"])
        return image_tensor, target, patient_id, image_name

    def _resolve_image_path(self, image_name: str) -> Path:
        stem = Path(image_name)
        if stem.suffix:
            return self.images_dir / stem.name
        return self.images_dir / f"{stem.name}.jpg"

    @staticmethod
    def _ensure_required_columns(columns: Iterable[str]) -> None:
        missing = REQUIRED_COLUMNS.difference(columns)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"CSV is missing required columns: {missing_str}")


def make_dataloader(
    csv_path: str,
    images_dir: str,
    transform: A.Compose,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ISICDataset(csv_path=csv_path, images_dir=images_dir, transform=transform)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )
