
import cv2
import numpy as np

from aquarestore.types import Image
from aquarestore.enhancement import contrast



def measure_luminance_contrast(image: Image) -> float:
    """
    Measure normalized luminance contrast using the standard
    deviation of the CIE Lab L channel.

    Returns a value between 0 and 1.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3) in RGB order")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0].astype(np.float32) / 255.0

    contrast = float(np.std(luminance))

    # Remove negligible floating-point residuals.
    if np.isclose(contrast, 0.0, atol=1e-7):
        return 0.0

    return contrast


def select_clip_limit(
    contrast: float,
    minimum: float = 1.0,
    maximum: float = 4.0,
) -> float:
    """
    Select a CLAHE clip limit from measured luminance contrast.

    Low contrast -> higher clip limit.
    High contrast -> lower clip limit.

    The mapping is deterministic and bounded.
    """
    if isinstance(contrast, bool) or not isinstance(
        contrast, (int, float, np.integer, np.floating)
    ):
        raise TypeError("contrast must be a real number")

    if not np.isfinite(contrast) or not 0.0 <= contrast <= 1.0:
        raise ValueError("contrast must be finite and between 0 and 1")

    if isinstance(minimum, bool) or isinstance(maximum, bool):
        raise TypeError("minimum and maximum must be real numbers")

    if not isinstance(minimum, (int, float, np.integer, np.floating)):
        raise TypeError("minimum must be a real number")

    if not isinstance(maximum, (int, float, np.integer, np.floating)):
        raise TypeError("maximum must be a real number")

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("minimum and maximum must be finite")

    if minimum <= 0 or maximum < minimum:
        raise ValueError("require 0 < minimum <= maximum")

    # Linear mapping: contrast 0 -> maximum, contrast 1 -> minimum.
    clip_limit = maximum - contrast * (maximum - minimum)

    return float(np.clip(clip_limit, minimum, maximum))
