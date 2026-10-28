import csv

import pytest

from aquarestore.visualization.plots import (
    QUALITY_METRICS,
    generate_all_plots,
    load_results,
)


def create_results_csv(path, rows, fields=None):
    """Create a temporary results CSV."""
    if fields is None:
        fields = [
            "filename",
            "mode",
            "status",
            *QUALITY_METRICS,
            "processing_time_seconds",
        ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_rows():
    """Return sample original, fixed, and adaptive results."""
    rows = []

    values = {
        "original": (0.3, 0.1, 0.2, 0.04, 0.0),
        "fixed": (0.4, 0.2, 0.3, 0.05, 0.5),
        "adaptive": (0.5, 0.25, 0.35, 0.06, 0.4),
    }

    for mode, metrics in values.items():
        row = {
            "filename": "sample.png",
            "mode": mode,
            "status": "ok",
            "processing_time_seconds": str(metrics[4]),
        }

        for metric, value in zip(QUALITY_METRICS, metrics[:4]):
            row[metric] = str(value)

        rows.append(row)

    return rows


def test_load_results_reads_successful_rows(tmp_path):
    csv_path = tmp_path / "results.csv"
    create_results_csv(csv_path, sample_rows())

    rows = load_results(csv_path)

    assert len(rows) == 3
    assert {row["mode"] for row in rows} == {
        "original",
        "fixed",
        "adaptive",
    }


def test_load_results_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Results CSV not found"):
        load_results(tmp_path / "missing.csv")


def test_load_results_rejects_missing_columns(tmp_path):
    csv_path = tmp_path / "results.csv"

    create_results_csv(
        csv_path,
        [],
        fields=["filename", "mode", "status"],
    )

    with pytest.raises(ValueError, match="missing columns"):
        load_results(csv_path)


def test_load_results_rejects_no_successful_rows(tmp_path):
    csv_path = tmp_path / "results.csv"

    rows = sample_rows()
    for row in rows:
        row["status"] = "error"

    create_results_csv(csv_path, rows)

    with pytest.raises(ValueError, match="no successful rows"):
        load_results(csv_path)


def test_generate_all_plots_creates_three_png_files(tmp_path):
    csv_path = tmp_path / "results.csv"
    output_dir = tmp_path / "plots"

    create_results_csv(csv_path, sample_rows())

    paths = generate_all_plots(csv_path, output_dir)

    assert len(paths) == 3

    assert {path.name for path in paths} == {
        "quality_comparison.png",
        "runtime_comparison.png",
        "paired_differences.png",
    }

    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0