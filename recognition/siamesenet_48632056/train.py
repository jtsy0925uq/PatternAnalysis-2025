from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

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


def main() -> None:
    args = parse_args()

    if args.split:
        run_split(args.data_root, args.images_dir, args.csv_path)
    else:
        print("No action specified. Use --split to run the data sanity and split routine.")


if __name__ == "__main__":
    main()
