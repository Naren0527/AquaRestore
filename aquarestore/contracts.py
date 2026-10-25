
from dataclasses import dataclass
from typing import Any, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray


Image: TypeAlias = NDArray[np.uint8]


@dataclass(frozen=True)
class QualityMetrics:
    mean_luminance: float
    luminance_contrast: float
    mean_saturation: float
    edge_strength: float
    uiqm: float | None = None
    uciqe: float | None = None


@dataclass(frozen=True)
class EnhancementResult:
    image: Image
    metrics_before: QualityMetrics
    metrics_after: QualityMetrics
    parameters_used: dict[str, Any]
    method: str


def _validate_image(image: Image, name: str = "image") -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")

    if image.dtype != np.uint8:
        raise TypeError(f"{name} must have dtype uint8")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"{name} must have shape (H, W, 3) in RGB order"
        )

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")


def correct_color(image: Image) -> Image:
    """Apply the configured color correction to an RGB uint8 image."""
    from aquarestore.enhancement.color_correction import (
        correct_color as implementation,
    )

    return implementation(image)


def enhance_contrast(
    image: Image,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> Image:
    """Enhance luminance contrast using CLAHE."""
    from aquarestore.enhancement.contrast import (
        enhance_contrast as implementation,
    )

    return implementation(
        image,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )


def denoise_image(
    image: Image,
    diameter: int = 9,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
) -> Image:
    """Apply edge-preserving bilateral denoising to an RGB image."""
    _validate_image(image)

    if isinstance(diameter, bool) or not isinstance(
        diameter, (int, np.integer)
    ):
        raise TypeError("diameter must be an integer")

    if diameter <= 0:
        raise ValueError("diameter must be positive")

    for name, value in (
        ("sigma_color", sigma_color),
        ("sigma_space", sigma_space),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a real number")

        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    # OpenCV's bilateral filter operates on the supplied channel order;
    # RGB input and output are preserved.
    result = cv2.bilateralFilter(
        image,
        d=int(diameter),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    )

    return result


def calculate_quality_metrics(image: Image) -> QualityMetrics:
    """
    Calculate basic image diagnostics.

    UIQM and UCIQE remain None until their implementations and
    formulas are verified.
    """
    _validate_image(image)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0].astype(np.float32) / 255.0

    mean_luminance = float(np.mean(luminance))
    luminance_contrast = float(np.std(luminance))

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]) / 255.0)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32) / 255.0

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = float(
        np.mean(np.sqrt(grad_x**2 + grad_y**2))
    )

    return QualityMetrics(
        mean_luminance=mean_luminance,
        luminance_contrast=luminance_contrast,
        mean_saturation=mean_saturation,
        edge_strength=edge_strength,
        uiqm=None,
        uciqe=None,
    )


def enhance_image(
    image: Image,
    adaptive: bool = True,
) -> EnhancementResult:
    """
    Run AquaRestore and return the enhanced image with diagnostics.

    The returned metrics are basic image statistics, not a claim
    of perceptual quality or reference-based accuracy.
    """
    _validate_image(image)

    if not isinstance(adaptive, (bool, np.bool_)):
        raise TypeError("adaptive must be a boolean")

    from aquarestore.enhancement.adaptive import (
        measure_luminance_contrast,
        select_clip_limit,
    )
    from aquarestore.enhancement.color_correction import correct_color as color_impl
    from aquarestore.enhancement.pipeline import (
        enhance_image as pipeline_impl,
    )

    original = image.copy()
    metrics_before = calculate_quality_metrics(original)

    # Mirror the pipeline's current parameter-selection rule so the
    # result records the selected CLAHE clip limit.
    corrected = color_impl(original)

    if adaptive:
        measured_contrast = measure_luminance_contrast(corrected)
        clip_limit = select_clip_limit(
            measured_contrast,
            minimum=1.0,
            maximum=2.0,
        )
        method = "adaptive"
    else:
        clip_limit = 1.5
        method = "fixed"

    enhanced = pipeline_impl(original, adaptive=adaptive)
    metrics_after = calculate_quality_metrics(enhanced)

    parameters_used = {
        "adaptive": adaptive,
        "color_correction_strength": 0.35,
        "clip_limit": float(clip_limit),
        "tile_grid_size": (8, 8),
    }

    return EnhancementResult(
        image=enhanced,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        parameters_used=parameters_used,
        method=method,
    )


def compare_images(
    original: Image,
    enhanced: Image,
) -> dict[str, Any]:
    """Compare two same-sized RGB images using basic diagnostics."""
    _validate_image(original, "original")
    _validate_image(enhanced, "enhanced")

    if original.shape != enhanced.shape:
        raise ValueError(
            "original and enhanced images must have the same shape"
        )

    before = calculate_quality_metrics(original)
    after = calculate_quality_metrics(enhanced)

    metric_names = (
        "mean_luminance",
        "luminance_contrast",
        "mean_saturation",
        "edge_strength",
    )

    before_values = {
        name: getattr(before, name) for name in metric_names
    }
    after_values = {
        name: getattr(after, name) for name in metric_names
    }
    differences = {
        name: after_values[name] - before_values[name]
        for name in metric_names
    }

    return {
        "metrics_before": before_values,
        "metrics_after": after_values,
        "differences": differences,
        "uiqm": None,
        "uciqe": None,
    }
