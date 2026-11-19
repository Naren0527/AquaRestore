"""Script to train the AquaRestore aquatic image classifier."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aquarestore.classification.config import ClassifierConfig
from aquarestore.classification.train import train_classifier


def main():
    parser = argparse.ArgumentParser(
        description="Train AquaRestore Aquatic Classifier"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/dataset",
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--patience", type=int, default=4, help="Early stopping patience"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for acceptance",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = ClassifierConfig(
        dataset_dir=Path(args.data_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        confidence_threshold=args.threshold,
        random_seed=args.seed,
    )

    print("============================================================")
    print("  AquaRestore Aquatic Image Classifier: Training Pipeline  ")
    print("============================================================")
    print(f"Dataset:       {config.dataset_dir}")
    print(f"Model Arch:    {config.model_name} (Transfer Learning)")
    print(f"Max Epochs:    {config.epochs}")
    print(f"Batch Size:    {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Random Seed:   {config.random_seed}")
    print(f"Threshold:     {config.confidence_threshold}")
    print("============================================================\n")

    metrics = train_classifier(config=config, dataset_dir=config.dataset_dir)
    print("\nTraining and Evaluation finished successfully!")
    print(f"Test Accuracy:  {metrics['test_accuracy']:.2%}")
    print(f"Test Precision: {metrics['test_precision']:.2%}")
    print(f"Test Recall:    {metrics['test_recall']:.2%}")
    print(f"Test F1-score:  {metrics['test_f1_score']:.4f}")
    print(f"Test ROC-AUC:   {metrics['test_roc_auc']:.4f}")
    print(f"Model saved to: {config.model_path}")


if __name__ == "__main__":
    main()
