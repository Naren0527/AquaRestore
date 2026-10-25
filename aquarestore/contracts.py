
from dataclasses import dataclass
from typing import Any, TypeAlias

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



def correct_color(image: Image) -> Image:
    """Return a color-corrected RGB uint8 image."""
    from aquarestore.enhancement.color_correction import (
        correct_color as _correct_color,
    )

    return _correct_color(image)


def enhance_contrast(
    image: Image,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> Image:
    """Enhance luminance contrast using CLAHE."""
    from aquarestore.enhancement.contrast import (
        enhance_contrast as _enhance_contrast,
    )

    return _enhance_contrast(
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
    """Apply edge-preserving denoising."""
    raise NotImplementedError(
        "Denoising is pending implementation by Person 2"
    )


def calculate_quality_metrics(image: Image) -> QualityMetrics:
    """Calculate the agreed image-quality measurements."""
    raise NotImplementedError(
        "Quality metrics are pending implementation by Person 2"
    )


def enhance_image(
    image: Image,
    adaptive: bool = True,
) -> EnhancementResult:
    """Run the complete AquaRestore enhancement pipeline."""
    raise NotImplementedError(
        "The EnhancementResult API will be connected after metrics "
        "and adaptive enhancement are implemented"
    )


def compare_images(
    original: Image,
    enhanced: Image,
) -> dict[str, Any]:
    """Return a quantitative comparison of two images."""
    raise NotImplementedError(
        "Image comparison is pending implementation by Person 2"
    )
