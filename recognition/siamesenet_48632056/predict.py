from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import auc as sklearn_auc

from dataset import ISICDataset, build_transforms
from module import SiameseBackbone, build_prototypes, score_by_prototypes

#arguments
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

#loads the checkpoint saved under checkpoints/best.pt
def load_checkpoint(backbone: str, checkpoint_path: Path, device: torch.device) -> SiameseBackbone:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Explicitly disable weights_only safeguard to load checkpoints containing pathlib objects.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    transform: object,
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
    transform: object,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, Optional[np.ndarray]]:
    dataset = ISICDataset(csv_path=csv_path, images_dir=images_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    image_names: list[str] = []
    probabilities: list[float] = []
    targets: list[int] = []

    has_targets = "target" in dataset.df.columns

    for batch in tqdm(loader, desc="Predicting"):
        images, targets_tensor, _, names = batch
        images = images.to(device)
        embeddings = model(images)
        probs = score_by_prototypes(embeddings, proto0, proto1).cpu().numpy()

        image_names.extend(names)
        probabilities.extend(probs.tolist())
        if has_targets:
            targets.extend(targets_tensor.cpu().numpy().tolist())

    pred_df = pd.DataFrame(
        {
            "image_name": image_names,
            "probability_melanoma": np.array(probabilities, dtype=np.float32),
        }
    )
    target_array = np.array(targets, dtype=np.int64) if has_targets and targets else None
    return pred_df, target_array


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_transform = build_transforms()

    model = load_checkpoint(args.backbone, args.checkpoint, device)

    # Build class prototypes from the training split for nearest-prototype scoring.
    prototypes = compute_prototypes(model, val_transform, args.data_root, args.batch_size, device)
    if prototypes is None:
        print("Prototypes unavailable; aborting inference.")
        return
    proto0, proto1 = prototypes

    # Run inference and capture optional ground-truth labels for evaluation.
    predictions, targets = run_inference(
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
    predictions = predictions[["image_name", "probability_melanoma"]]
    predictions.to_csv(output_path, index=False, float_format="%.4f")
    print(f"Wrote predictions for {len(predictions)} images to {output_path}.")

    if targets is not None and targets.size:
        probs = predictions["probability_melanoma"].to_numpy()
        preds_binary = (probs >= 0.5).astype(int)
        accuracy = accuracy_score(targets, preds_binary)
        try:
            auc = roc_auc_score(targets, probs)
            print(f"Accuracy: {accuracy:.4f} | ROC AUC: {auc:.4f}")
        except ValueError as exc:
            print(f"Accuracy: {accuracy:.4f} | ROC AUC skipped: {exc}")
    else:
        print("No ground-truth labels, skipping metrics.")

    # Generate evaluation figures for documentation.
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    probs = predictions["probability_melanoma"].to_numpy()

    if targets is not None and targets.size:
        try:
            fpr, tpr, _ = roc_curve(targets, probs)
            roc_auc = sklearn_auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], "k--", lw=1)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve - ISIC 2020 Siamese Model")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(fig_dir / "roc_curve.png", dpi=150)
            plt.close()
        except ValueError as exc:
            print(f"Skipping ROC curve: {exc}")

        cm = confusion_matrix(targets, (probs >= 0.5).astype(int))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Melanoma"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix - ISIC 2020 Siamese Model")
        plt.tight_layout()
        plt.savefig(fig_dir / "confusion_matrix.png", dpi=150)
        plt.close()
    else:
        print("Skipping ROC curve and confusion matrix: no ground-truth targets.")

    if len(predictions):
        sample_indices = np.random.choice(len(predictions), size=min(9, len(predictions)), replace=False)
        plt.figure(figsize=(8, 8))
        plotted = False
        for position, idx in enumerate(sample_indices, 1):
            img_name = predictions.iloc[idx]["image_name"]
            img_path = Path(args.images_dir) / f"{img_name}.jpg"
            if not img_path.is_file():
                print(f"Missing image for prediction grid: {img_path}")
                continue
            img = plt.imread(img_path)
            plt.subplot(3, 3, position)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"p(melanoma)={predictions.iloc[idx]['probability_melanoma']:.2f}")
            plotted = True
        if plotted:
            plt.tight_layout()
            plt.savefig(fig_dir / "pred_examples.png", dpi=150)
            plt.close()
        else:
            plt.close()
            print("Skipping prediction grid: sampled images not found on disk.")
    else:
        print("Skipping prediction grid: no predictions available.")


if __name__ == "__main__":
    main()
