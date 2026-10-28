
"""Analyze paired fixed-versus-adaptive enhancement results."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_RESULTS_PATH = Path("outputs/evaluation/results.csv")

METRICS = (
    "mean_luminance",
    "luminance_contrast",
    "mean_saturation",
    "edge_strength",
    "processing_time_seconds",
)


def load_paired_results(
    results_path: str | Path,
) -> dict[str, dict[str, dict[str, str]]]:
    """Load successful fixed/adaptive rows and pair them by filename."""
    path = Path(results_path)

    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path.resolve()}")

    paired: dict[str, dict[str, dict[str, str]]] = {}

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        required = {"filename", "mode", "status", *METRICS}
        available = set(reader.fieldnames or [])
        missing = required - available

        if missing:
            raise ValueError(
                "Results CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            if row.get("status") != "ok":
                continue

            filename = row.get("filename", "")
            mode = row.get("mode", "")

            if not filename or mode not in {"fixed", "adaptive"}:
                continue

            # Duplicate rows for the same image and mode are ambiguous.
            # Keep the first successful row rather than silently replacing it.
            modes = paired.setdefault(filename, {})
            if mode not in modes:
                modes[mode] = row

    return {
        filename: modes
        for filename, modes in paired.items()
        if "fixed" in modes and "adaptive" in modes
    }


def _to_finite_float(value: Any) -> float | None:
    """Convert a CSV value to a finite float, or return None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not __import__("math").isfinite(number):
        return None

    return number


def summarize_metric(
    paired: dict[str, dict[str, dict[str, str]]],
    metric: str,
) -> dict[str, Any]:
    """Summarize paired fixed/adaptive values for one metric."""
    fixed_values: list[float] = []
    adaptive_values: list[float] = []
    deltas: list[float] = []
    excluded = 0

    for modes in paired.values():
        fixed = _to_finite_float(modes["fixed"].get(metric))
        adaptive = _to_finite_float(modes["adaptive"].get(metric))

        if fixed is None or adaptive is None:
            excluded += 1
            continue

        fixed_values.append(fixed)
        adaptive_values.append(adaptive)
        deltas.append(adaptive - fixed)

    if not deltas:
        return {
            "metric": metric,
            "valid_pairs": 0,
            "excluded_pairs": excluded,
            "fixed_mean": None,
            "adaptive_mean": None,
            "mean_paired_delta": None,
            "median_paired_delta": None,
            "adaptive_higher": 0,
            "adaptive_lower": 0,
            "equal": 0,
        }

    higher = sum(delta > 0 for delta in deltas)
    lower = sum(delta < 0 for delta in deltas)
    equal = sum(delta == 0 for delta in deltas)

    return {
        "metric": metric,
        "valid_pairs": len(deltas),
        "excluded_pairs": excluded,
        "fixed_mean": mean(fixed_values),
        "adaptive_mean": mean(adaptive_values),
        "mean_paired_delta": mean(deltas),
        "median_paired_delta": median(deltas),
        "adaptive_higher": higher,
        "adaptive_lower": lower,
        "equal": equal,
    }


def analyze_results(
    results_path: str | Path = DEFAULT_RESULTS_PATH,
) -> dict[str, Any]:
    """Analyze paired experiment results and return a JSON-ready summary."""
    paired = load_paired_results(results_path)

    summary: dict[str, Any] = {
        "results_path": str(Path(results_path).resolve()),
        "images_with_both_successful_modes": len(paired),
        "metrics": {},
    }

    for metric in METRICS:
        summary["metrics"][metric] = summarize_metric(paired, metric)

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable summary."""
    print(
        "Images with both successful modes: "
        f"{summary['images_with_both_successful_modes']}"
    )

    if summary["images_with_both_successful_modes"] == 0:
        print("No complete image pairs available.")
        return

    for metric, result in summary["metrics"].items():
        print(f"\n{metric}")
        print(f"  Valid pairs:          {result['valid_pairs']}")
        print(f"  Excluded pairs:       {result['excluded_pairs']}")

        if result["valid_pairs"] == 0:
            print("  No valid numeric pairs available.")
            continue

        print(f"  Fixed mean:           {result['fixed_mean']:.6f}")
        print(f"  Adaptive mean:        {result['adaptive_mean']:.6f}")
        print(f"  Mean paired delta:    {result['mean_paired_delta']:+.6f}")
        print(f"  Median paired delta:  {result['median_paired_delta']:+.6f}")
        print(
            f"  Adaptive higher:      {result['adaptive_higher']}"
        )
        print(
            f"  Adaptive lower:       {result['adaptive_lower']}"
        )
        print(f"  Equal:                {result['equal']}")

    print(
        "\nInterpretation: positive changes in contrast or edge strength "
        "are not automatically improvements in perceptual quality."
    )
    print(
        "Processing time measures enhancement runtime only; it excludes "
        "image loading, metric calculation, and output saving."
    )
    print(
        "Use diverse real underwater images and repeated runs before "
        "drawing conclusions about image quality or speed."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze fixed-versus-adaptive experiment results."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to the experiment results CSV.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for saving the analysis summary as JSON.",
    )
    args = parser.parse_args()

    summary = analyze_results(args.results)
    print_summary(summary)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as json_file:
            json.dump(summary, json_file, indent=2, allow_nan=False)
        print(f"\nJSON summary saved to: {args.json_output.resolve()}")


if __name__ == "__main__":
    main()

