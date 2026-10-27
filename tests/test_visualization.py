from PIL import Image

import pytest

from aquarestore.evaluation.visualize_results import build_contact_sheet


def create_image(path, color, size=(40, 30)):
    """Create a small test image."""
    Image.new("RGB", size, color).save(path)


def setup_image_set(tmp_path, count=1):
    """Create original, fixed, and adaptive images."""
    input_dir = tmp_path / "samples"
    output_dir = tmp_path / "evaluation"

    input_dir.mkdir()
    (output_dir / "fixed").mkdir(parents=True)
    (output_dir / "adaptive").mkdir()

    for index in range(count):
        filename = f"image_{index}.png"

        create_image(input_dir / filename, (30, 100, 180))
        create_image(output_dir / "fixed" / filename, (100, 150, 200))
        create_image(output_dir / "adaptive" / filename, (180, 200, 220))

    return input_dir, output_dir


def test_build_contact_sheet_creates_image(tmp_path):
    input_dir, output_dir = setup_image_set(tmp_path)

    result = build_contact_sheet(input_dir, output_dir)

    assert result.exists()
    assert result.name == "contact_sheet.png"

    with Image.open(result) as image:
        assert image.mode == "RGB"
        assert image.width > 0
        assert image.height > 0


def test_contact_sheet_is_reproducible(tmp_path):
    input_dir, output_dir = setup_image_set(tmp_path, count=4)

    first_result = build_contact_sheet(
        input_dir, output_dir, sample_size=2, seed=42
    )
    first_bytes = first_result.read_bytes()

    second_result = build_contact_sheet(
        input_dir, output_dir, sample_size=2, seed=42
    )
    second_bytes = second_result.read_bytes()

    assert first_bytes == second_bytes


def test_build_contact_sheet_rejects_invalid_sample_size(tmp_path):
    input_dir, output_dir = setup_image_set(tmp_path)

    with pytest.raises(ValueError, match="sample_size"):
        build_contact_sheet(input_dir, output_dir, sample_size=0)


def test_build_contact_sheet_rejects_missing_input_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Input directory not found"):
        build_contact_sheet(tmp_path / "missing", tmp_path / "output")


def test_build_contact_sheet_requires_both_enhanced_images(tmp_path):
    input_dir, output_dir = setup_image_set(tmp_path)

    (output_dir / "adaptive" / "image_0.png").unlink()

    with pytest.raises(
        FileNotFoundError, match="No matching original images"
    ):
        build_contact_sheet(input_dir, output_dir)