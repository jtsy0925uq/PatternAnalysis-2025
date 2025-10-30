from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import ISICDataset, build_transforms
from module import SiameseBackbone, build_prototypes, score_by_prototypes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference and evaluation for Siamese network.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data/train"),
        help="Directory containing train CSVs (train_split.csv) and training images.",
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=Path("data/test/test-img"),
        help="Directory with images to score.",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path("data/test/test.csv"),
        help="Metadata CSV for inference containing required columns.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best.pt"),
        help="Checkpoint file containing trained Siamese backbone weights.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("predictions.csv"),
        help="Path where predictions will be written.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="tf_efficientnet_b0.ns_jft_in1k",
        help="Backbone identifier for the Siamese model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size used for prototype embedding and inference.",
    )
    return parser.parse_args()


def load_checkpoint(backbone: str, checkpoint_path: Path, device: torch.device) -> SiameseBackbone:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state") if isinstance(checkpoint, dict) else checkpoint
    if state_dict is None:
        raise KeyError("Checkpoint does not contain 'model_state'.")

    model = SiameseBackbone(backbone=backbone)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def compute_prototypes(
    model: SiameseBackbone,
    transform,
    data_root: Path,
    batch_size: int,
    device: torch.device,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    train_split_path = data_root / "train_split.csv"
    train_images_dir = data_root / "train-img"

    if not train_split_path.is_file():
        print(f"WARNING: train split CSV not found at {train_split_path}. Skipping prototypes.")
        return None
    if not train_images_dir.is_dir():
        print(f"WARNING: train images directory not found at {train_images_dir}. Skipping prototypes.")
        return None

    dataset = ISICDataset(csv_path=train_split_path, images_dir=train_images_dir, transform=transform)
    mask = dataset.df["split"].astype(str).str.lower() == "train"
    indices = np.nonzero(mask.to_numpy())[0].tolist()
    if not indices:
        print("WARNING: No training samples found in train_split.csv with split=='train'. Skipping prototypes.")
        return None

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    embeddings: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="Building prototypes"):
        images, ys, _, _ = batch
        images = images.to(device)
        embs = model(images)
        embeddings.append(embs.cpu())
        targets.append(ys.cpu())

    if not embeddings:
        print("WARNING: Unable to compute prototypes - no embeddings collected.")
        return None

    embs_tensor = torch.cat(embeddings, dim=0).to(device)
    ys_tensor = torch.cat(targets, dim=0).to(device)

    try:
        proto0, proto1 = build_prototypes(embs_tensor, ys_tensor)
    except ValueError as exc:
        print(f"WARNING: {exc}. Skipping prototypes.")
        return None

    return proto0, proto1


@torch.no_grad()
def run_inference(
    model: SiameseBackbone,
    proto0: torch.Tensor,
    proto1: torch.Tensor,
    csv_path: Path,
    images_dir: Path,
    transform,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    dataset = ISICDataset(csv_path=csv_path, images_dir=images_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    image_names: list[str] = []
    probabilities: list[float] = []

    for batch in tqdm(loader, desc="Predicting"):
        images, _, _, names = batch
        images = images.to(device)
        embeddings = model(images)
        probs = score_by_prototypes(embeddings, proto0, proto1).cpu().numpy()

        image_names.extend(names)
        probabilities.extend(probs.tolist())

    return pd.DataFrame(
        {
            "image_name": image_names,
            "probability_melanoma": np.array(probabilities, dtype=np.float32),
        }
    )


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_transform = build_transforms()

    model = load_checkpoint(args.backbone, args.checkpoint, device)

    prototypes = compute_prototypes(model, val_transform, args.data_root, args.batch_size, device)
    if prototypes is None:
        print("Prototypes unavailable; aborting inference.")
        return
    proto0, proto1 = prototypes

    predictions = run_inference(
        model=model,
        proto0=proto0,
        proto1=proto1,
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        transform=val_transform,
        batch_size=args.batch_size,
        device=device,
    )

    output_path = args.output_csv
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.sort_values("image_name").to_csv(output_path, index=False)
    print(f"Wrote predictions for {len(predictions)} images to {output_path}.")


if __name__ == "__main__":
    main()
