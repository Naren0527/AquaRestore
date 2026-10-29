
"""Image-quality diagnostics for AquaRestore."""

import cv2
import numpy as np

from aquarestore.types import Image
from aquarestore.contracts import QualityMetrics



def calculate_quality_metrics(image: Image) -> QualityMetrics:
    """
    Calculate basic quality measurements for an RGB uint8 image.

    Measurements:
        - Mean luminance
        - Luminance contrast
        - Mean saturation
        - Edge strength

    UIQM and UCIQE are left as None until verified implementations
    are integrated.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3) in RGB order")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")

    # Convert RGB to LAB and normalize luminance.
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0].astype(np.float32) / 255.0

    mean_luminance = float(np.mean(luminance))
    luminance_contrast = float(np.std(luminance))

    # Calculate mean saturation in HSV color space.
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]) / 255.0)

    # Calculate average gradient magnitude as edge strength.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32) / 255.0

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    edge_strength = float(np.mean(gradient_magnitude))

    return QualityMetrics(
        mean_luminance=mean_luminance,
        luminance_contrast=luminance_contrast,
        mean_saturation=mean_saturation,
        edge_strength=edge_strength,
        uiqm=None,
        uciqe=None,
    )

