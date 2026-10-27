
"""Reproducible fixed-versus-adaptive enhancement experiments."""

import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from aquarestore.contracts import calculate_quality_metrics
from aquarestore.enhancement.pipeline import enhance_image


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

METRIC_NAMES = (
    "mean_luminance",
    "luminance_contrast",
    "mean_saturation",
    "edge_strength",
    "uiqm",
    "uciqe",
)

CSV_FIELDS = [
    "filename",
    "mode",
    "processing_time_seconds",
    *METRIC_NAMES,
    "status",
    "error",
]


def calculate_basic_metrics(image: np.ndarray) -> dict[str, float | None]:
    """Calculate the project's shared image-quality metrics."""
    metrics = calculate_quality_metrics(image)

    return {
        "mean_luminance": metrics.mean_luminance,
        "luminance_contrast": metrics.luminance_contrast,
        "mean_saturation": metrics.mean_saturation,
        "edge_strength": metrics.edge_strength,
        "uiqm": metrics.uiqm,
        "uciqe": metrics.uciqe,
    }


def _load_rgb_image(image_path: Path) -> np.ndarray:
    """Load an image as a validated RGB uint8 NumPy array."""
    with PILImage.open(image_path) as pil_image:
        image = np.asarray(pil_image.convert("RGB"), dtype=np.uint8).copy()

    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        raise ValueError(f"Invalid RGB image: {image_path.name}")

    return image


def _metrics_for_csv(image: np.ndarray) -> dict[str, float | None]:
    """Return metrics with values suitable for CSV serialization."""
    return calculate_basic_metrics(image)


def run_experiments(
    input_dir: str | Path = "data/samples",
    output_dir: str | Path = "outputs/evaluation",
) -> Path:
    """
    Compare original images with fixed and adaptive enhancement.

    For each input image, record:
      - Original image quality metrics.
      - Fixed enhancement quality metrics and runtime.
      - Adaptive enhancement quality metrics and runtime.

    Outputs:
      output_dir/
        fixed/
        adaptive/
        results.csv

    Returns the path to the results CSV file.

    Failed processing modes are recorded separately so that one mode's
    failure does not automatically discard a successful result from the
    other mode.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {input_path.resolve()}. "
            "Create the directory and add sample images before running "
            "the full experiment."
        )

    image_paths = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(
            f"No supported images found in {input_path.resolve()}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    fixed_dir = output_path / "fixed"
    adaptive_dir = output_path / "adaptive"

    fixed_dir.mkdir(parents=True, exist_ok=True)
    adaptive_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_path / "results.csv"
    total_rows = 0

    with results_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for index, image_path in enumerate(image_paths, start=1):
            try:
                original = _load_rgb_image(image_path)
            except Exception as exc:
                # The original could not be loaded, so neither enhancement
                # mode can be evaluated for this file.
                error_message = f"{type(exc).__name__}: {exc}"

                for mode in ("original", "fixed", "adaptive"):
                    row = {
                        "filename": image_path.name,
                        "mode": mode,
                        "processing_time_seconds": "",
                        **{name: "" for name in METRIC_NAMES},
                        "status": "error",
                        "error": error_message,
                    }
                    writer.writerow(row)
                    total_rows += 1

                print(f"Failed to load {image_path.name}: {error_message}")
                continue

            # Evaluate the original once, without treating it as ground truth.
            try:
                original_metrics = _metrics_for_csv(original)
                writer.writerow({
                    "filename": image_path.name,
                    "mode": "original",
                    "processing_time_seconds": 0.0,
                    **original_metrics,
                    "status": "ok",
                    "error": "",
                })
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                writer.writerow({
                    "filename": image_path.name,
                    "mode": "original",
                    "processing_time_seconds": "",
                    **{name: "" for name in METRIC_NAMES},
                    "status": "error",
                    "error": error_message,
                })
                print(f"Original metrics failed for {image_path.name}: {error_message}")

            total_rows += 1

            # Each mode is isolated: one failure must not prevent the other
            # mode from being attempted.
            for mode, adaptive in (
                ("fixed", False),
                ("adaptive", True),
            ):
                start = time.perf_counter()

                try:
                    enhanced = enhance_image(original, adaptive=adaptive)
                    elapsed = time.perf_counter() - start

                    if (
                        not isinstance(enhanced, np.ndarray)
                        or enhanced.dtype != np.uint8
                        or enhanced.ndim != 3
                        or enhanced.shape != original.shape
                    ):
                        raise ValueError(
                            "Enhancement must return an RGB uint8 array "
                            "with the same dimensions as the input."
                        )

                    metrics = _metrics_for_csv(enhanced)

                    destination_dir = (
                        fixed_dir if mode == "fixed" else adaptive_dir
                    )
                    destination = destination_dir / f"{image_path.stem}.png"

                    PILImage.fromarray(enhanced).save(destination)

                    writer.writerow({
                        "filename": image_path.name,
                        "mode": mode,
                        "processing_time_seconds": elapsed,
                        **metrics,
                        "status": "ok",
                        "error": "",
                    })

                except Exception as exc:
                    elapsed = time.perf_counter() - start
                    error_message = f"{type(exc).__name__}: {exc}"

                    writer.writerow({
                        "filename": image_path.name,
                        "mode": mode,
                        "processing_time_seconds": elapsed,
                        **{name: "" for name in METRIC_NAMES},
                        "status": "error",
                        "error": error_message,
                    })

                    print(f"{mode} failed for {image_path.name}: {error_message}")

                total_rows += 1

            if index % 25 == 0 or index == len(image_paths):
                print(f"Processed {index}/{len(image_paths)} images")

    print(f"Processed input images: {len(image_paths)}")
    print(f"Result rows written: {total_rows}")
    print(f"Results saved to: {results_path.resolve()}")

    return results_path


if __name__ == "__main__":
    run_experiments()

