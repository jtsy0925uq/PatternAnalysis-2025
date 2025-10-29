from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import ISICDataset, build_transforms, make_dataloader
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
        "--train_run",
        action="store_true",
        help="Execute the training and validation loop.",
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.2,
        help="Margin used in batch hard triplet loss.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="tf_efficientnet_b0_ns",
        help="Backbone name for the Siamese encoder.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory where checkpoints will be saved.",
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


def run_train(args: argparse.Namespace) -> None:
    """
    Minimal training + validation loop for the Siamese backbone.
    """
    csv_train = args.data_root / "train_split.csv"
    csv_val = args.data_root / "val_split.csv"

    for required_path in (csv_train, csv_val):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Required split CSV not found: {required_path}. Run --split first."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_tf, val_tf = build_transforms()
    train_dataset = ISICDataset(
        csv_path=str(csv_train), images_dir=str(args.images_dir), transform=train_tf
    )
    val_dataset = ISICDataset(
        csv_path=str(csv_val), images_dir=str(args.images_dir), transform=val_tf
    )

    train_targets = torch.as_tensor(
        train_dataset.df["target"].to_numpy(), dtype=torch.long
    )
    if train_targets.numel() == 0:
        raise ValueError("Training split is empty.")

    class_counts = torch.bincount(train_targets)
    if class_counts.numel() < 2 or (class_counts == 0).any():
        raise ValueError("Training split must contain samples for both classes.")

    class_weights = class_counts.float().reciprocal()
    sample_weights = class_weights[train_targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = SiameseBackbone(backbone=args.backbone, out_dim=512).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=device.type == "cuda")
    amp_enabled = device.type == "cuda"

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_batches = 0

        for images, targets, _, _ in train_loader:
            images = images.to(device, non_blocking=pin_memory)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                embeddings = model(images)
                try:
                    loss = batch_hard_triplet_loss(
                        embeddings, targets, margin=args.margin
                    )
                except ValueError:
                    # Skip batches without both classes.
                    continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_batches += 1

        train_loss = running_loss / train_batches if train_batches else float("nan")

        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, targets, _, _ in val_loader:
                images = images.to(device, non_blocking=pin_memory)
                targets = targets.to(device)
                with autocast(enabled=amp_enabled):
                    embeddings = model(images)
                    try:
                        loss = batch_hard_triplet_loss(
                            embeddings, targets, margin=args.margin
                        )
                    except ValueError:
                        continue
                val_loss_total += loss.item()
                val_batches += 1

        val_loss = val_loss_total / val_batches if val_batches else float("nan")
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "model_last.pt"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": vars(args),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


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

    if args.train_run:
        run_train(args)
        action_taken = True

    if not action_taken:
        print(
            "No action specified. Use --split, --smoke, --model_smoke, or --train_run "
            "to run a utility routine."
        )


if __name__ == "__main__":
    main()
