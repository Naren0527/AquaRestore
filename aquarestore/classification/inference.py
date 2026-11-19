"""Inference engine for AquaRestore aquatic image classification."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms

from aquarestore.classification.config import ClassifierConfig
from aquarestore.classification.model import AquaticClassifier


@dataclass(frozen=True)
class ClassificationResult:
    """Outcome of aquatic image validation and classification."""

    is_aquatic: bool
    confidence: float
    aquatic_probability: float
    non_aquatic_probability: float
    predicted_class: str
    accepted: bool
    threshold: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert classification result to a serializable dictionary."""
        return {
            "is_aquatic": self.is_aquatic,
            "confidence": self.confidence,
            "aquatic_probability": self.aquatic_probability,
            "non_aquatic_probability": self.non_aquatic_probability,
            "predicted_class": self.predicted_class,
            "accepted": self.accepted,
            "threshold": self.threshold,
            "message": self.message,
        }


class AquaticInference:
    """High-performance classifier inference wrapper supporting CPU and GPU."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        threshold: float = 0.85,
        device: str | None = None,
    ) -> None:
        self.config = ClassifierConfig()
        self.threshold = float(threshold)

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.mean, std=self.config.std
                ),
            ]
        )

        resolved_model_path = Path(model_path or self.config.model_path)
        self.model = AquaticClassifier(
            num_classes=self.config.num_classes, pretrained=False
        )

        if resolved_model_path.exists():
            checkpoint = torch.load(
                resolved_model_path, map_location=self.device
            )
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            # If checkpoint does not yet exist, initialize with pretrained weights
            self.model = AquaticClassifier(
                num_classes=self.config.num_classes, pretrained=True
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: np.ndarray | PILImage.Image) -> ClassificationResult:
        """Classify an image as aquatic or non-aquatic and verify threshold acceptance."""
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("image must have shape (H, W, 3) in RGB order")
            if image.dtype != np.uint8:
                raise TypeError("image must have dtype uint8")
            if image.shape[0] == 0 or image.shape[1] == 0:
                raise ValueError("image must not be empty")
            pil_img = PILImage.fromarray(image)
        elif isinstance(image, PILImage.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError("image must be a NumPy array or PIL.Image.Image")

        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.model.predict_proba(input_tensor)[0]
            non_aq_prob = float(probs[0].item())
            aq_prob = float(probs[1].item())

        is_aquatic = aq_prob > non_aq_prob
        predicted_class = "aquatic" if is_aquatic else "non_aquatic"
        confidence = aq_prob if is_aquatic else non_aq_prob

        # Apply acceptance criteria
        if not is_aquatic:
            accepted = False
            message = (
                "Unsupported Image: AquaRestore is designed specifically for "
                "aquatic and water-containing images. Please upload an image "
                "containing a recognizable water environment."
            )
        elif aq_prob < self.threshold:
            accepted = False
            message = (
                f"Uncertain Image Content: Classifier confidence ({aq_prob:.1%}) "
                f"is below the required threshold ({self.threshold:.1%}). "
                "Please upload a clearer aquatic image."
            )
        else:
            accepted = True
            message = (
                f"Aquatic Image Detected. Confidence: {aq_prob:.1%}. "
                "Proceeding with AquaRestore enhancement."
            )

        return ClassificationResult(
            is_aquatic=is_aquatic,
            confidence=confidence,
            aquatic_probability=aq_prob,
            non_aquatic_probability=non_aq_prob,
            predicted_class=predicted_class,
            accepted=accepted,
            threshold=self.threshold,
            message=message,
        )

