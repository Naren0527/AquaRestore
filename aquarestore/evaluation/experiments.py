
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from aquarestore.enhancement.pipeline import enhance_image


def calculate_basic_metrics(image: np.ndarray) -> dict[str, float]:
    """Calculate basic diagnostic metrics for an RGB uint8 image."""
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        raise ValueError("image must be a non-empty RGB uint8 array")

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0].astype(np.float32) / 255.0
    luminance_contrast = float(np.std(luminance))

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]) / 255.0)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_float = gray.astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))

    return {
        "luminance_contrast": luminance_contrast,
        "mean_saturation": mean_saturation,
        "edge_strength": edge_strength,
    }


def run_experiments(
    input_dir: str | Path = "data/samples",
    output_dir: str | Path = "outputs/evaluation",
) -> Path:
    """
    Evaluate fixed and adaptive enhancement on the same input images.

    Outputs:
      outputs/evaluation/fixed/
      outputs/evaluation/adaptive/
      outputs/evaluation/results.csv

    Returns the path to the CSV results file.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    image_paths = sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    if not image_paths:
        raise ValueError(f"No supported images found in {input_path}")

    fixed_dir = output_path / "fixed"
    adaptive_dir = output_path / "adaptive"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    adaptive_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_path / "results.csv"
    fieldnames = [
        "filename",
        "mode",
        "processing_time_seconds",
        "luminance_contrast",
        "mean_saturation",
        "edge_strength",
        "status",
        "error",
    ]

    with results_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, image_path in enumerate(image_paths, start=1):
            try:
                with PILImage.open(image_path) as pil_image:
                    original = np.asarray(
                        pil_image.convert("RGB"), dtype=np.uint8
                    ).copy()

                for mode, adaptive in (
                    ("fixed", False),
                    ("adaptive", True),
                ):
                    start = time.perf_counter()
                    enhanced = enhance_image(original, adaptive=adaptive)
                    elapsed = time.perf_counter() - start

                    metrics = calculate_basic_metrics(enhanced)

                    destination = (
                        fixed_dir if mode == "fixed" else adaptive_dir
                    ) / f"{image_path.stem}.png"

                    PILImage.fromarray(enhanced, mode="RGB").save(destination)

                    writer.writerow({
                        "filename": image_path.name,
                        "mode": mode,
                        "processing_time_seconds": elapsed,
                        **metrics,
                        "status": "ok",
                        "error": "",
                    })

                if index % 25 == 0 or index == len(image_paths):
                    print(f"Processed {index}/{len(image_paths)} images")

            except Exception as exc:
                # Record failures instead of silently dropping images.
                for mode in ("fixed", "adaptive"):
                    writer.writerow({
                        "filename": image_path.name,
                        "mode": mode,
                        "processing_time_seconds": "",
                        "luminance_contrast": "",
                        "mean_saturation": "",
                        "edge_strength": "",
                        "status": "error",
                        "error": str(exc),
                    })
                print(f"Failed: {image_path.name}: {exc}")

    print(f"Results saved to: {results_path.resolve()}")
    return results_path


if __name__ == "__main__":
    run_experiments()
