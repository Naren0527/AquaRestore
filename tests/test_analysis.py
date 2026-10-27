import csv

import pytest

from aquarestore.evaluation.analyze_results import (
    METRICS,
    analyze_results,
    load_paired_results,
    print_summary,
    summarize_metric,
)


def create_results_csv(path, rows, fields=None):
    """Create a temporary experiment results CSV."""
    if fields is None:
        fields = [
            "filename",
            "mode",
            "status",
            *METRICS,
        ]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_row(filename, mode, values, status="ok"):
    """Create one CSV result row."""
    row = {
        "filename": filename,
        "mode": mode,
        "status": status,
    }
    row.update(values)
    return row


def test_load_paired_results_pairs_successful_modes(tmp_path):
    csv_path = tmp_path / "results.csv"

    rows = [
        make_row(
            "image1.png",
            "fixed",
            {"mean_luminance": "0.4"},
        ),
        make_row(
            "image1.png",
            "adaptive",
            {"mean_luminance": "0.6"},
        ),
        make_row(
            "image2.png",
            "fixed",
            {"mean_luminance": "0.3"},
        ),
        make_row(
            "image3.png",
            "adaptive",
            {"mean_luminance": "0.7"},
        ),
    ]

    create_results_csv(csv_path, rows)

    paired = load_paired_results(csv_path)

    assert list(paired) == ["image1.png"]
    assert set(paired["image1.png"]) == {"fixed", "adaptive"}


def test_load_paired_results_ignores_failed_rows(tmp_path):
    csv_path = tmp_path / "results.csv"

    rows = [
        make_row(
            "image1.png",
            "fixed",
            {"mean_luminance": "0.4"},
        ),
        make_row(
            "image1.png",
            "adaptive",
            {"mean_luminance": "0.6"},
            status="error",
        ),
    ]

    create_results_csv(csv_path, rows)

    assert load_paired_results(csv_path) == {}


def test_load_paired_results_rejects_missing_columns(tmp_path):
    csv_path = tmp_path / "results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["filename", "mode", "status"],
        )
        writer.writeheader()

    with pytest.raises(ValueError, match="missing required columns"):
        load_paired_results(csv_path)


def test_load_paired_results_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_paired_results(tmp_path / "missing.csv")


def test_summarize_metric_calculates_paired_differences():
    paired = {
        "image1.png": {
            "fixed": {"mean_luminance": "0.2"},
            "adaptive": {"mean_luminance": "0.6"},
        },
        "image2.png": {
            "fixed": {"mean_luminance": "0.5"},
            "adaptive": {"mean_luminance": "0.3"},
        },
        "image3.png": {
            "fixed": {"mean_luminance": "0.4"},
            "adaptive": {"mean_luminance": "0.4"},
        },
    }

    summary = summarize_metric(paired, "mean_luminance")

    assert summary["valid_pairs"] == 3
    assert summary["excluded_pairs"] == 0
    assert summary["fixed_mean"] == pytest.approx(0.3666667)
    assert summary["adaptive_mean"] == pytest.approx(0.4333333)
    assert summary["mean_paired_delta"] == pytest.approx(0.0666667)
    assert summary["median_paired_delta"] == pytest.approx(0.0)
    assert summary["adaptive_higher"] == 1
    assert summary["adaptive_lower"] == 1
    assert summary["equal"] == 1


def test_summarize_metric_excludes_invalid_numeric_values():
    paired = {
        "image1.png": {
            "fixed": {"mean_luminance": "0.2"},
            "adaptive": {"mean_luminance": "0.5"},
        },
        "image2.png": {
            "fixed": {"mean_luminance": "invalid"},
            "adaptive": {"mean_luminance": "0.7"},
        },
        "image3.png": {
            "fixed": {"mean_luminance": "nan"},
            "adaptive": {"mean_luminance": "0.4"},
        },
    }

    summary = summarize_metric(paired, "mean_luminance")

    assert summary["valid_pairs"] == 1
    assert summary["excluded_pairs"] == 2
    assert summary["mean_paired_delta"] == pytest.approx(0.3)


def test_summarize_metric_handles_no_valid_pairs():
    paired = {
        "image1.png": {
            "fixed": {"mean_luminance": ""},
            "adaptive": {"mean_luminance": ""},
        }
    }

    summary = summarize_metric(paired, "mean_luminance")

    assert summary["valid_pairs"] == 0
    assert summary["excluded_pairs"] == 1
    assert summary["fixed_mean"] is None
    assert summary["adaptive_mean"] is None
    assert summary["mean_paired_delta"] is None


def test_analyze_results_returns_all_metrics(tmp_path):
    csv_path = tmp_path / "results.csv"

    rows = [
        make_row(
            "image1.png",
            "fixed",
            {
                "mean_luminance": "0.3",
                "luminance_contrast": "0.1",
                "mean_saturation": "0.2",
                "edge_strength": "0.05",
                "processing_time_seconds": "0.5",
            },
        ),
        make_row(
            "image1.png",
            "adaptive",
            {
                "mean_luminance": "0.4",
                "luminance_contrast": "0.2",
                "mean_saturation": "0.3",
                "edge_strength": "0.06",
                "processing_time_seconds": "0.4",
            },
        ),
    ]

    create_results_csv(csv_path, rows)

    summary = analyze_results(csv_path)

    assert summary["images_with_both_successful_modes"] == 1
    assert set(summary["metrics"]) == set(METRICS)
    assert summary["metrics"]["mean_luminance"]["mean_paired_delta"] == (
        pytest.approx(0.1)
    )


def test_print_summary_handles_no_complete_pairs(capsys):
    summary = {
        "images_with_both_successful_modes": 0,
        "metrics": {},
    }

    print_summary(summary)

    output = capsys.readouterr().out

    assert "No complete image pairs available." in output