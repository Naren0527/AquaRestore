
import cv2
import numpy as np

from aquarestore.contracts import Image


def correct_color(image: Image, strength: float = 0.35) -> Image:
    """
    Apply conservative, blended Gray World white balancing.

    strength=0.0 returns the original image.
    strength=1.0 applies the full Gray World correction.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3) in RGB order")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")

    if isinstance(strength, (bool, np.bool_)) or not isinstance(
        strength, (int, float, np.integer, np.floating)
    ):
        raise TypeError("strength must be a real number")

    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be finite and between 0 and 1")

    if strength == 0:
        return image.copy()

    source = image.astype(np.float32)
    channel_means = source.mean(axis=(0, 1))
    target_mean = float(channel_means.mean())

    safe_means = np.maximum(channel_means, 1.0)
    gains = target_mean / safe_means

    # Limit correction to avoid extreme channel amplification.
    gains = np.clip(gains, 0.7, 1.5)

    balanced = source * gains.reshape(1, 1, 3)

    # Blend corrected and original pixels for a gentler result.
    corrected = (
        (1.0 - strength) * source
        + strength * balanced
    )

    return np.clip(corrected, 0, 255).astype(np.uint8)
