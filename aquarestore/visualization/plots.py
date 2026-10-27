
"""Generate plots for AquaRestore enhancement experiments."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


QUALITY_METRICS = (
    "mean_luminance",
    "luminance_contrast",
    "mean_saturation",
    "edge_strength",
)


def load_results(csv_path: str | Path) -> list[dict]:
    """Load successful experiment rows from a results CSV."""
    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(f"Results CSV not found: {path.resolve()}")

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"filename", "mode", "status", *QUALITY_METRICS,
                    "processing_time_seconds"}
        available = set(reader.fieldnames or [])
        missing = required - available

        if missing:
            raise ValueError(
                "Results CSV is missing columns: "
                + ", ".join(sorted(missing))
            )

        rows = [
            row for row in reader
            if row.get("status") == "ok"
            and row.get("mode") in {"original", "fixed", "adaptive"}
        ]

    if not rows:
        raise ValueError("The results CSV contains no successful rows.")

    return rows


def _finite_value(value: str) -> float | None:
    """Return a finite numeric value, or None for missing/invalid values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if np.isfinite(number) else None


def plot_quality_comparison(
    rows: list[dict],
    output_dir: str | Path,
) -> Path:
    """Plot mean quality metrics for original, fixed, and adaptive images."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    modes = ("original", "fixed", "adaptive")
    labels = ("Original", "Fixed", "Adaptive")
    means: dict[str, list[float]] = {metric: [] for metric in QUALITY_METRICS}

    for metric in QUALITY_METRICS:
        for mode in modes:
            values = [
                value
                for row in rows
                if row["mode"] == mode
                and (value := _finite_value(row.get(metric, ""))) is not None
            ]
            means[metric].append(float(np.mean(values)) if values else np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.ravel()

    for ax, metric in zip(axes_flat, QUALITY_METRICS):
        values = means[metric]
        ax.bar(labels, values)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Mean value")
        ax.grid(axis="y", alpha=0.25)

        if all(not np.isfinite(value) for value in values):
            ax.text(
                0.5, 0.5, "No valid data",
                transform=ax.transAxes, ha="center", va="center"
            )

    fig.suptitle("AquaRestore: Image Quality Comparison")
    fig.tight_layout()

    destination = output_path / "quality_comparison.png"
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return destination


def plot_runtime_comparison(
    rows: list[dict],
    output_dir: str | Path,
) -> Path:
    """Plot average enhancement runtime for fixed and adaptive modes."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    modes = ("fixed", "adaptive")
    labels = ("Fixed", "Adaptive")
    averages = []

    for mode in modes:
        values = [
            value
            for row in rows
            if row["mode"] == mode
            and (value := _finite_value(
                row.get("processing_time_seconds", "")
            )) is not None
            and value >= 0
        ]
        averages.append(float(np.mean(values)) if values else np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, averages)
    ax.set_title("AquaRestore: Average Enhancement Runtime")
    ax.set_ylabel("Seconds per image")
    ax.grid(axis="y", alpha=0.25)

    destination = output_path / "runtime_comparison.png"
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return destination


def plot_paired_differences(
    rows: list[dict],
    output_dir: str | Path,
) -> Path:
    """Plot adaptive-minus-fixed metric differences for matched images."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paired: dict[str, dict[str, dict]] = {}

    for row in rows:
        if row["mode"] not in {"fixed", "adaptive"}:
            continue
        paired.setdefault(row["filename"], {})[row["mode"]] = row

    differences: dict[str, list[float]] = {
        metric: [] for metric in QUALITY_METRICS
    }

    for modes in paired.values():
        if "fixed" not in modes or "adaptive" not in modes:
            continue

        for metric in QUALITY_METRICS:
            fixed = _finite_value(modes["fixed"].get(metric, ""))
            adaptive = _finite_value(modes["adaptive"].get(metric, ""))

            if fixed is not None and adaptive is not None:
                differences[metric].append(adaptive - fixed)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    averages = []

    for metric in QUALITY_METRICS:
        values = differences[metric]
        if values:
            labels.append(metric.replace("_", " ").title())
            averages.append(float(np.mean(values)))

    if averages:
        ax.bar(labels, averages)
        ax.axhline(0, linewidth=1)
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylabel("Adaptive minus fixed")
        ax.set_title("AquaRestore: Paired Metric Differences")
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(
            0.5, 0.5, "No matched metric pairs available",
            transform=ax.transAxes, ha="center", va="center"
        )
        ax.set_axis_off()

    destination = output_path / "paired_differences.png"
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return destination


def generate_all_plots(
    csv_path: str | Path = "outputs/evaluation/results.csv",
    output_dir: str | Path = "outputs/evaluation/plots",
) -> list[Path]:
    """Generate all evaluation plots and return their paths."""
    rows = load_results(csv_path)

    paths = [
        plot_quality_comparison(rows, output_dir),
        plot_runtime_comparison(rows, output_dir),
        plot_paired_differences(rows, output_dir),
    ]

    for path in paths:
        print(f"Saved plot: {path.resolve()}")

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot AquaRestore experiment results."
    )
    parser.add_argument(
        "--csv",
        default="outputs/evaluation/results.csv",
        help="Path to the experiment results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation/plots",
        help="Directory in which to save plots.",
    )
    args = parser.parse_args()

    generate_all_plots(args.csv, args.output_dir)


if __name__ == "__main__":
    main()

