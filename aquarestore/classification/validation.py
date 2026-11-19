"""Validation layer integrating aquatic classification into AquaRestore."""

from pathlib import Path
from typing import Any

import numpy as np

from aquarestore.classification.inference import (
    AquaticInference,
    ClassificationResult,
)

_DEFAULT_INFERENCE_INSTANCE: AquaticInference | None = None


def get_inference_engine(
    model_path: Path | str | None = None,
    threshold: float = 0.85,
) -> AquaticInference:
    """Return a cached or newly initialized AquaticInference engine."""
    global _DEFAULT_INFERENCE_INSTANCE
    if (
        _DEFAULT_INFERENCE_INSTANCE is None
        or _DEFAULT_INFERENCE_INSTANCE.threshold != threshold
    ):
        _DEFAULT_INFERENCE_INSTANCE = AquaticInference(
            model_path=model_path, threshold=threshold
        )
    return _DEFAULT_INFERENCE_INSTANCE


def validate_aquatic_image(
    image: np.ndarray,
    threshold: float = 0.85,
    model_path: Path | str | None = None,
) -> ClassificationResult:
    """
    Validate that an input image is recognizable aquatic / water content.

    Runs BEFORE any enhancement algorithms are invoked.

    Raises:
        TypeError: If input is not a uint8 NumPy array.
        ValueError: If image shape is not (H, W, 3) or image is empty.

    Returns:
        ClassificationResult with is_aquatic, accepted, confidence, and status message.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3) in RGB order")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")

    engine = get_inference_engine(model_path=model_path, threshold=threshold)
    return engine.predict(image)

