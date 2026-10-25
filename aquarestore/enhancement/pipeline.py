
import numpy as np

from aquarestore.contracts import Image
from aquarestore.enhancement.color_correction import correct_color
from aquarestore.enhancement.contrast import enhance_contrast
from aquarestore.enhancement.adaptive import (
    measure_luminance_contrast,
    select_clip_limit,
)


def enhance_image(
    image: Image,
    adaptive: bool = False,
) -> Image:
    """
    Run the AquaRestore enhancement pipeline.

    Fixed mode uses CLAHE clip limit 2.0.
    Adaptive mode selects a clip limit based on luminance contrast.

    Args:
        image: RGB uint8 image with shape (H, W, 3).
        adaptive: Whether to adapt the CLAHE clip limit.

    Returns:
        Enhanced RGB uint8 image with the same spatial dimensions.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3) in RGB order")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")

    if not isinstance(adaptive, (bool, np.bool_)):
        raise TypeError("adaptive must be a boolean")

    corrected = correct_color(image)

    if adaptive:
        contrast = measure_luminance_contrast(corrected)
        clip_limit = select_clip_limit(
            contrast,
            minimum=1.0,
            maximum=2.0,
        )
    else:
        clip_limit = 2.0

    return enhance_contrast(
        corrected,
        clip_limit=clip_limit,
        tile_grid_size=(8, 8),
    )
