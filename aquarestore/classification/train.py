"""Model training, evaluation, and checkpointing for AquaRestore aquatic classifier."""

import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from aquarestore.classification.config import ClassifierConfig
from aquarestore.classification.dataset import (
    AquaticDataset,
    get_eval_transforms,
    get_train_transforms,
)
from aquarestore.classification.model import AquaticClassifier, build_model


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate classifier performance and compute comprehensive metrics."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[float] = []

    with torch.no_grad():
        for inputs, targets, _ in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_predictions.extend(preds.cpu().numpy().tolist())
            # Probability for class 1 (aquatic)
            all_probabilities.extend(probs[:, 1].cpu().numpy().tolist())

    n_samples = len(all_targets)
    avg_loss = total_loss / max(1, n_samples)

    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "loss": float(avg_loss),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
    }


def plot_training_history(history: dict[str, list[float]], save_path: Path) -> None:
    """Plot and save loss and accuracy progression curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#e5322d")
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#3f6fd1")
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy curve
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Accuracy", color="#e5322d")
    ax2.plot(epochs, history["val_acc"], "s--", label="Val Accuracy", color="#3f6fd1")
    ax2.set_title("Training & Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix_and_roc(
    cm: list[list[int]],
    y_true: list[int],
    y_prob: list[float],
    cm_path: Path,
    roc_path: Path,
) -> None:
    """Plot and save confusion matrix heatmap and ROC curve."""
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_arr = np.array(cm)
    cax = ax.matshow(cm_arr, cmap="Blues", alpha=0.85)
    fig.colorbar(cax)

    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j,
                i,
                str(cm_arr[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black",
            )

    labels = ["Non-Aquatic", "Aquatic"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix (Test Set)", fontsize=13, pad=15)
    plt.tight_layout()
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # 2. ROC Curve
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(
            fpr,
            tpr,
            color="#e5322d",
            lw=2.5,
            label=f"MobileNetV3 (AUC = {auc_score:.4f})",
        )
        ax.plot([0, 1], [0, 1], color="#8a8a94", linestyle="--", lw=1.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
        ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=13)
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(roc_path, dpi=200)
        plt.close()
    except Exception as e:
        print(f"Could not generate ROC curve: {e}")


def train_classifier(
    config: ClassifierConfig | None = None,
    dataset_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Train the aquatic classifier using transfer learning, with early stopping and evaluation."""
    cfg = config or ClassifierConfig()
    set_seed(cfg.random_seed)

    base_dir = Path(dataset_dir or cfg.dataset_dir)
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    test_dir = base_dir / "test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build Datasets
    train_dataset = AquaticDataset.from_directory(
        train_dir, transform=get_train_transforms(cfg)
    )
    val_dataset = AquaticDataset.from_directory(
        val_dir, transform=get_eval_transforms(cfg)
    )
    test_dataset = AquaticDataset.from_directory(
        test_dir, transform=get_eval_transforms(cfg)
    )

    print(
        f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    # Initialize model with ImageNet transfer learning
    model = build_model(num_classes=cfg.num_classes, pretrained=True, device=device)

    # Stage 1: Train classification head with frozen backbone features
    model.freeze_backbone(freeze=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0
    best_weights = None
    epochs_no_improve = 0
    start_time = time.time()

    print("\n--- Phase A: Warm-up Classifier Head (Frozen Backbone) ---")
    for epoch in range(1, 4):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_train_loss = train_loss / total
        epoch_train_acc = correct / total

        val_metrics = evaluate_model(model, val_loader, device)
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_score"])

        print(
            f"Epoch {epoch:02d} [Head]: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1_score']:.4f}"
        )

        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print("\n--- Phase B: Fine-Tuning Top Backbone Blocks ---")
    model.unfreeze_top_blocks(num_blocks=4)
    # Re-initialize optimizer with lower learning rate for backbone fine-tuning
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate * 0.4,
        weight_decay=cfg.weight_decay,
    )

    for epoch in range(4, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_train_loss = train_loss / total
        epoch_train_acc = correct / total

        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step(val_metrics["f1_score"])

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_score"])

        print(
            f"Epoch {epoch:02d} [Fine-tune]: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1_score']:.4f}"
        )

        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    training_duration = time.time() - start_time
    print(f"\nTraining completed in {training_duration:.2f} seconds.")

    # Load best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)
        model = model.to(device)

    # Save best model checkpoint
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_name": cfg.model_name,
                "num_classes": cfg.num_classes,
                "class_names": list(cfg.class_names),
                "image_size": cfg.image_size,
                "confidence_threshold": cfg.confidence_threshold,
            },
            "best_val_f1": best_val_f1,
            "training_duration": training_duration,
        },
        cfg.model_path,
    )
    print(f"Best model checkpoint saved to: {cfg.model_path}")

    # Evaluate on held-out test set
    print("\n--- Evaluating Best Model on Held-out Test Set ---")
    test_metrics = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1-score:  {test_metrics['f1_score']:.4f}")
    print(f"Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix:\n{np.array(test_metrics['confusion_matrix'])}")

    # Plot and save artifacts
    plot_training_history(history, cfg.loss_curve_path)
    plot_confusion_matrix_and_roc(
        test_metrics["confusion_matrix"],
        test_metrics["y_true"],
        test_metrics["y_prob"],
        cfg.confusion_matrix_path,
        cfg.roc_curve_path,
    )

    # Save metrics and history json
    clean_test_metrics = {
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1_score": test_metrics["f1_score"],
        "test_roc_auc": test_metrics["roc_auc"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "best_val_f1": best_val_f1,
        "training_duration_seconds": round(training_duration, 2),
        "total_epochs_trained": len(history["train_loss"]),
    }
    with open(cfg.metrics_path, "w") as f:
        json.dump(clean_test_metrics, f, indent=2)

    with open(cfg.history_path, "w") as f:
        json.dump(history, f, indent=2)

    return clean_test_metrics

