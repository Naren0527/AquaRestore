"""Evaluation script for the trained AquaRestore aquatic image classifier."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from aquarestore.classification.config import ClassifierConfig
from aquarestore.classification.dataset import (
    AquaticDataset,
    get_eval_transforms,
)
from aquarestore.classification.model import AquaticClassifier


def evaluate_across_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float],
) -> list[dict]:
    """Analyze precision, recall, accuracy, and false positive rates across various confidence thresholds."""
    results = []
    for th in thresholds:
        # For aquatic class (1), accepted if probability >= th
        y_pred = (y_prob >= th).astype(int)
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        results.append(
            {
                "threshold": float(th),
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "false_positive_rate": fpr,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AquaRestore Aquatic Classifier"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/aquatic_classifier.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/dataset/test",
        help="Path to test set directory",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="models/detailed_test_evaluation.json",
        help="Path to save detailed evaluation metrics",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    config = ClassifierConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {model_path} (Device: {device})")

    checkpoint = torch.load(model_path, map_location=device)
    model = AquaticClassifier(
        num_classes=config.num_classes, pretrained=False
    )
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    test_dataset = AquaticDataset.from_directory(
        args.data_dir, transform=get_eval_transforms(config)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    print(f"Evaluating on {len(test_dataset)} held-out test samples...")

    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probabilities.extend(probs[:, 1].cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_prob = np.array(all_probabilities)

    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    threshold_eval = evaluate_across_thresholds(y_true, y_prob, thresholds)

    roc_auc = float(roc_auc_score(y_true, y_prob))

    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP & PERFORMANCE ANALYSIS (TEST SET)")
    print("=" * 70)
    print(
        f"{'Threshold':<11}{'Accuracy':<11}{'Precision':<12}{'Recall':<10}{'F1-Score':<11}{'FPR (Non-Aq Leak)':<15}"
    )
    print("-" * 70)
    for row in threshold_eval:
        marker = " <-- [CHOSEN]" if abs(row["threshold"] - 0.85) < 1e-3 else ""
        print(
            f"{row['threshold']:<11.2f}{row['accuracy']:<11.2%}{row['precision']:<12.2%}{row['recall']:<10.2%}{row['f1_score']:<11.4f}{row['false_positive_rate']:<15.2%}{marker}"
        )
    print("=" * 70)
    print(f"Area Under ROC Curve (ROC-AUC): {roc_auc:.4f}\n")

    output_data = {
        "model_path": str(model_path),
        "test_dataset_size": len(test_dataset),
        "roc_auc": roc_auc,
        "threshold_analysis": threshold_eval,
    }

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Detailed threshold analysis saved to: {out_p}")


if __name__ == "__main__":
    main()
