
import csv
from pathlib import Path
from statistics import mean, median

RESULTS_PATH = Path("outputs/evaluation/results.csv")

METRICS = [
    "luminance_contrast",
    "mean_saturation",
    "edge_strength",
    "processing_time_seconds",
]


def main():
    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(f"Results file not found: {RESULTS_PATH}")

    paired = {}

    with RESULTS_PATH.open(
        "r", newline="", encoding="utf-8"
    ) as csv_file:
        for row in csv.DictReader(csv_file):
            if row["status"] != "ok":
                continue

            filename = row["filename"]
            mode = row["mode"]

            paired.setdefault(filename, {})[mode] = row

    valid = {
        filename: modes
        for filename, modes in paired.items()
        if "fixed" in modes and "adaptive" in modes
    }

    print(f"Images with both successful modes: {len(valid)}")

    if not valid:
        print("No complete image pairs available.")
        return

    for metric in METRICS:
        fixed_values = []
        adaptive_values = []
        deltas = []

        for modes in valid.values():
            fixed_value = float(modes["fixed"][metric])
            adaptive_value = float(modes["adaptive"][metric])

            fixed_values.append(fixed_value)
            adaptive_values.append(adaptive_value)
            deltas.append(adaptive_value - fixed_value)

        higher_count = sum(delta > 0 for delta in deltas)
        lower_count = sum(delta < 0 for delta in deltas)
        equal_count = sum(delta == 0 for delta in deltas)

        print(f"\n{metric}")
        print(f"  Fixed mean:          {mean(fixed_values):.6f}")
        print(f"  Adaptive mean:       {mean(adaptive_values):.6f}")
        print(f"  Mean paired delta:   {mean(deltas):+.6f}")
        print(f"  Median paired delta: {median(deltas):+.6f}")
        print(
            f"  Adaptive higher:     {higher_count}/{len(deltas)} "
            f"({100 * higher_count / len(deltas):.1f}%)"
        )
        print(
            f"  Adaptive lower:      {lower_count}/{len(deltas)} "
            f"({100 * lower_count / len(deltas):.1f}%)"
        )
        print(f"  Equal:               {equal_count}/{len(deltas)}")

    print(
        "\nInterpretation: positive changes in contrast or edge strength "
        "are not automatically improvements in perceptual quality."
    )
    print(
        "Processing time is measured per image and excludes dataset "
        "loading, metric calculation, and saving."
    )


if __name__ == "__main__":
    main()
