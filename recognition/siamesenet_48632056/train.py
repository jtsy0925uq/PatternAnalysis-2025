from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from dataset import build_transforms, make_dataloader
from module import SiameseBackbone, batch_hard_triplet_loss

REQUIRED_COLUMNS: set[str] = {
    "image_name",
    "patient_id",
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
    "diagnosis",
    "benign_malignant",
    "target",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Utilities for Siamese network training workflow."
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Run data sanity checks and create patient-wise train/val splits.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Load a single training batch to verify the data pipeline.",
    )
    parser.add_argument(
        "--model_smoke",
        action="store_true",
        help="Run a forward pass through the Siamese backbone and report diagnostics.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data/train"),
        help="Directory containing the train CSV and where split CSVs will be saved.",
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=Path("data/train/train-img"),
        help="Directory containing training images.",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path("data/train/train.csv"),
        help="Path to the training metadata CSV.",
    )
    return parser.parse_args()


def ensure_required_columns(columns: Iterable[str], csv_path: Path) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV at {csv_path} is missing required columns: {missing}")


def print_sanity_stats(df: pd.DataFrame, images_dir: Path) -> None:
    total_rows = len(df)
    unique_patients = df["patient_id"].nunique(dropna=True)
    target_counts = df["target"].value_counts().sort_index()
    target_ratio = df["target"].value_counts(normalize=True).sort_index()
    missing_values = df.isna().sum()

    print(f"Total rows: {total_rows}")
    print(f"Unique patients: {unique_patients}")
    print(f"Target class counts: {target_counts.to_dict()}")
    print(
        "Target class ratio (fraction of total rows per class): "
        f"{target_ratio.round(4).to_dict()}"
    )
    print(f"Missing values per column: {missing_values.to_dict()}")

    missing_images = find_missing_images(df, images_dir)
    print(f"Missing image files: {len(missing_images)}")
    if missing_images:
        sample = ", ".join(missing_images[:5])
        raise FileNotFoundError(
            f"Found {len(missing_images)} missing images under {images_dir}. "
            f"Sample: {sample}"
        )


def find_missing_images(df: pd.DataFrame, images_dir: Path) -> list[str]:
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found: {images_dir}")

    available_stems = {
        image_path.stem
        for image_path in images_dir.iterdir()
        if image_path.is_file()
    }

    missing = []
    for image_name in df["image_name"].astype(str):
        if image_name not in available_stems:
            missing.append(image_name)
    return missing


def run_split(data_root: Path, images_dir: Path, csv_path: Path) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    ensure_required_columns(df.columns, csv_path)

    if df["patient_id"].isna().any():
        raise ValueError("patient_id column contains missing values; cannot group split.")

    print_sanity_stats(df, images_dir)

    splitter = GroupShuffleSplit(test_size=0.2, random_state=42, n_splits=1)
    train_idx, val_idx = next(splitter.split(df, groups=df["patient_id"]))

    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    df_train["split"] = "train"
    df_val["split"] = "val"

    train_output_path = data_root / "train_split.csv"
    val_output_path = data_root / "val_split.csv"

    df_train.to_csv(train_output_path, index=False)
    df_val.to_csv(val_output_path, index=False)

    print(f"Saved train split to {train_output_path}")
    print(f"Saved validation split to {val_output_path}")


def run_smoke(data_root: Path, images_dir: Path) -> None:
    csv_path = data_root / "train_split.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Expected split CSV at {csv_path}. Run --split before invoking --smoke."
        )

    train_tf, _ = build_transforms()
    dataloader = make_dataloader(
        csv_path=str(csv_path),
        images_dir=str(images_dir),
        transform=train_tf,
        batch_size=8,
        shuffle=False,
    )

    try:
        images, targets, patient_ids, image_names = next(iter(dataloader))
    except StopIteration as exc:
        raise ValueError(f"No samples available in {csv_path}") from exc

    positives = int((targets == 1).sum().item())
    negatives = int((targets == 0).sum().item())

    print(f"Smoke batch shape: {tuple(images.shape)}")
    print(f"Smoke batch dtype: {images.dtype}")
    print(f"Smoke batch range: min {images.min().item():.4f}, max {images.max().item():.4f}")
    print(
        "Smoke batch class balance: "
        f"positives={positives}, negatives={negatives}, total={targets.numel()}"
    )


def run_model_smoke(data_root: Path, images_dir: Path) -> None:
    """
    Forward one batch through the Siamese backbone to validate the model stack.
    """
    csv_path = data_root / "train_split.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Expected split CSV at {csv_path}. Run --split before invoking --model_smoke."
        )

    _, val_tf = build_transforms()
    dataloader = make_dataloader(
        csv_path=str(csv_path),
        images_dir=str(images_dir),
        transform=val_tf,
        batch_size=8,
        shuffle=False,
    )

    try:
        images, targets, _, _ = next(iter(dataloader))
    except StopIteration as exc:
        raise ValueError(f"No samples available in {csv_path}") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseBackbone().to(device)
    model.eval()

    images = images.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        embeddings = model(images)

    norms = embeddings.norm(p=2, dim=-1)
    print(f"Model smoke embeddings shape: {tuple(embeddings.shape)}")
    print(
        "Model smoke embedding L2 norms: "
        f"mean={norms.mean().item():.4f}, std={norms.std(unbiased=False).item():.4f}"
    )

    if targets.unique().numel() >= 2:
        loss = batch_hard_triplet_loss(embeddings, targets)
        print(f"Model smoke triplet loss: {loss.item():.4f}")
    else:
        print("Model smoke triplet loss skipped: batch lacks both classes.")


def main() -> None:
    args = parse_args()

    action_taken = False

    if args.split:
        run_split(args.data_root, args.images_dir, args.csv_path)
        action_taken = True

    if args.smoke:
        run_smoke(args.data_root, args.images_dir)
        action_taken = True

    if args.model_smoke:
        run_model_smoke(args.data_root, args.images_dir)
        action_taken = True

    if not action_taken:
        print("No action specified. Use --split, --smoke, or --model_smoke to run a utility routine.")


if __name__ == "__main__":
    main()
