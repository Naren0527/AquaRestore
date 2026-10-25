
import cv2
import numpy as np

from aquarestore.contracts import Image


def enhance_contrast(
    image: Image,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> Image:
    """
    Enhance local luminance contrast using CLAHE.

    Args:
        image: RGB uint8 image with shape (H, W, 3).
        clip_limit: Positive CLAHE contrast-limiting parameter.
        tile_grid_size: Positive (columns, rows) tile-grid dimensions.

    Returns:
        RGB uint8 image with the same spatial dimensions.

    Raises:
        TypeError: For unsupported input or parameter types.
        ValueError: For invalid image dimensions or parameter values.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3) in RGB order")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")

    if isinstance(clip_limit, bool) or not isinstance(
        clip_limit, (int, float, np.integer, np.floating)
    ):
        raise TypeError("clip_limit must be a real number")

    if not np.isfinite(clip_limit) or clip_limit <= 0:
        raise ValueError("clip_limit must be finite and positive")

    if not isinstance(tile_grid_size, tuple) or len(tile_grid_size) != 2:
        raise TypeError("tile_grid_size must be a tuple of two integers")

    for value in tile_grid_size:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("tile_grid_size values must be integers")
        if value <= 0:
            raise ValueError("tile_grid_size values must be positive")

    # Convert RGB to LAB so contrast is adjusted on luminance alone.
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tile_grid_size,
    )

    enhanced_luminance = clahe.apply(luminance)

    enhanced_lab = cv2.merge(
        (enhanced_luminance, channel_a, channel_b)
    )

    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    return result
