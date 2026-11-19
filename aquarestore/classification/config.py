"""Configuration settings for the AquaRestore aquatic image classifier."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class ClassifierConfig:
    """Configuration parameters for model training and inference."""

    # Architecture
    model_name: str = "mobilenet_v3_small"
    num_classes: int = 2
    class_names: Sequence[str] = field(
        default_factory=lambda: ["non_aquatic", "aquatic"]
    )
    class_mapping: dict[int, str] = field(
        default_factory=lambda: {0: "non_aquatic", 1: "aquatic"}
    )

    # Preprocessing
    image_size: int = 224
    mean: Sequence[float] = field(
        default_factory=lambda: [0.485, 0.456, 0.406]
    )
    std: Sequence[float] = field(
        default_factory=lambda: [0.229, 0.224, 0.225]
    )

    # Training Hyperparameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 15
    patience: int = 4
    lr_factor: float = 0.5
    lr_patience: int = 2
    random_seed: int = 42

    # Inference & Validation
    confidence_threshold: float = 0.85
    min_resolution: tuple[int, int] = (64, 64)

    # File paths
    model_dir: Path = Path("models")
    model_path: Path = Path("models/aquatic_classifier.pth")
    metrics_path: Path = Path("models/classifier_metrics.json")
    history_path: Path = Path("models/training_history.json")
    confusion_matrix_path: Path = Path("models/confusion_matrix.png")
    roc_curve_path: Path = Path("models/roc_curve.png")
    loss_curve_path: Path = Path("models/training_curves.png")
    dataset_dir: Path = Path("data/dataset")

