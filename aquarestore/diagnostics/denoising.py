
"""Edge-preserving denoising for AquaRestore."""

import cv2
import numpy as np

from aquarestore.contracts import Image


def denoise_image(
    image: Image,
    diameter: int = 9,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
) -> Image:
    """
    Reduce image noise while preserving important edges.

    Args:
        image: RGB uint8 image with shape (H, W, 3).
        diameter: Diameter of the pixel neighborhood.
        sigma_color: Filter strength for color differences.
        sigma_space: Filter strength for spatial distances.

    Returns:
        A denoised RGB uint8 image with unchanged dimensions.

    Raises:
        TypeError: If the image or parameters have invalid types.
        ValueError: If the image or parameters are invalid.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3) in RGB order")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")

    if isinstance(diameter, (bool, np.bool_)) or not isinstance(
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

    # OpenCV processes the supplied channels without changing their order.
    # Therefore, RGB input produces RGB output.
    result = cv2.bilateralFilter(
        image,
        d=int(diameter),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    )

    return result

