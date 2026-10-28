
import csv

import numpy as np
import pytest
from PIL import Image as PILImage

from aquarestore.evaluation.experiments import (
    CSV_FIELDS,
    calculate_basic_metrics,
    run_experiments,
)


def create_sample_image(path):
    """Create a small RGB image for testing."""
    image = np.zeros((32, 40, 3), dtype=np.uint8)
    image[:, :20] = (30, 100, 180)
    image[:, 20:] = (180, 200, 220)

    PILImage.fromarray(image).save(path)


def test_calculate_basic_metrics_returns_expected_fields():
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    metrics = calculate_basic_metrics(image)

    assert set(metrics) == {
        "mean_luminance",
        "luminance_contrast",
        "mean_saturation",
        "edge_strength",
        "uiqm",
        "uciqe",
    }

    assert metrics["mean_luminance"] == 0
    assert metrics["luminance_contrast"] == 0
    assert metrics["mean_saturation"] == 0
    assert metrics["edge_strength"] == 0

    # These metrics have not been implemented yet.
    assert metrics["uiqm"] is None
    assert metrics["uciqe"] is None


def test_run_experiments_creates_results_and_enhanced_images(tmp_path):
    input_dir = tmp_path / "samples"
    output_dir = tmp_path / "evaluation"
    input_dir.mkdir()

    create_sample_image(input_dir / "sample.png")

    results_path = run_experiments(input_dir, output_dir)

    assert results_path == output_dir / "results.csv"
    assert results_path.exists()
    assert (output_dir / "fixed" / "sample.png").exists()
    assert (output_dir / "adaptive" / "sample.png").exists()

    with results_path.open(
        "r", newline="", encoding="utf-8"
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    assert reader.fieldnames == CSV_FIELDS
    assert len(rows) == 3

    rows_by_mode = {row["mode"]: row for row in rows}

    assert set(rows_by_mode) == {"original", "fixed", "adaptive"}

    for row in rows:
        assert row["filename"] == "sample.png"
        assert row["status"] == "ok"
        assert row["error"] == ""

    assert float(
        rows_by_mode["original"]["processing_time_seconds"]
    ) == 0.0

    for mode in ("original", "fixed", "adaptive"):
        assert rows_by_mode[mode]["mean_luminance"] != ""


def test_run_experiments_rejects_missing_input_directory(tmp_path):
    missing_dir = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        run_experiments(missing_dir, tmp_path / "output")


def test_run_experiments_rejects_empty_input_directory(tmp_path):
    input_dir = tmp_path / "empty"
    input_dir.mkdir()

    with pytest.raises(ValueError, match="No supported images found"):
        run_experiments(input_dir, tmp_path / "output")


def test_run_experiments_records_invalid_image(tmp_path):
    input_dir = tmp_path / "samples"
    output_dir = tmp_path / "evaluation"
    input_dir.mkdir()

    (input_dir / "broken.png").write_bytes(b"not a real image")

    results_path = run_experiments(input_dir, output_dir)

    with results_path.open(
        "r", newline="", encoding="utf-8"
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 3

    for row in rows:
        assert row["filename"] == "broken.png"
        assert row["status"] == "error"
        assert row["error"] != ""

    assert not (output_dir / "fixed" / "broken.png").exists()
    assert not (output_dir / "adaptive" / "broken.png").exists()

