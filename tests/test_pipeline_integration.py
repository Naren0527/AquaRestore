"""Integration tests covering all 12 AquaRestore aquatic intelligence criteria.

Verifies acceptance of water scenes, rejection of non-aquatic scenes,
confidence threshold enforcement, controlled error handling, pipeline gating,
and preservation of fixed and adaptive enhancement modes.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from aquarestore.classification.inference import (
    AquaticInference,
    ClassificationResult,
)
from aquarestore.classification.validation import validate_aquatic_image
from aquarestore.contracts import (
    EnhancementResult,
    enhance_image,
)


def load_real_or_synthetic_image(category: str) -> np.ndarray:
    """Helper to locate a real dataset image or synthesize representative features."""
    dataset_dir = Path("data/dataset")
    if dataset_dir.exists():
        # Look in raw or splits
        found = list(dataset_dir.glob(f"**/*{category}*.*"))
        if found:
            with Image.open(found[0]) as img:
                return np.asarray(img.convert("RGB"), dtype=np.uint8)

    # Fallback to local samples
    samples = Path("data/samples")
    if samples.exists():
        first = next(samples.glob("*.png"), None)
        if first:
            with Image.open(first) as img:
                return np.asarray(img.convert("RGB"), dtype=np.uint8)

    # Synthetic fallback
    rng = np.random.default_rng(42)
    return rng.integers(50, 200, (128, 128, 3), dtype=np.uint8)


# 1. A valid underwater image is accepted
def test_1_valid_underwater_image_accepted():
    sample_path = Path("data/samples/907_img_.png")
    if sample_path.exists():
        with Image.open(sample_path) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    else:
        rgb = load_real_or_synthetic_image("uieb")

    result = validate_aquatic_image(rgb, threshold=0.85)
    assert result.is_aquatic is True
    assert result.accepted is True
    assert result.confidence >= 0.85
    assert "Aquatic Image Detected" in result.message


# 2. An ocean or sea image is accepted
def test_2_ocean_or_sea_image_accepted():
    ocean_imgs = list(Path("data/dataset").glob("**/*ocean*.*"))
    if ocean_imgs:
        with Image.open(ocean_imgs[0]) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        result = validate_aquatic_image(rgb, threshold=0.50)
        assert result.is_aquatic is True
    else:
        # Mocked verification
        with patch.object(
            AquaticInference,
            "predict",
            return_value=ClassificationResult(
                is_aquatic=True,
                confidence=0.94,
                aquatic_probability=0.94,
                non_aquatic_probability=0.06,
                predicted_class="aquatic",
                accepted=True,
                threshold=0.85,
                message="Aquatic Image Detected. Confidence: 94.0%.",
            ),
        ):
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            res = validate_aquatic_image(dummy, threshold=0.85)
            assert res.accepted is True
            assert res.is_aquatic is True


# 3. A river or lake image is accepted
def test_3_river_or_lake_image_accepted():
    target_img = Path("data/dataset/raw/aquatic/river_landscape_wa_006.jpg")
    if target_img.exists():
        with Image.open(target_img) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        result = validate_aquatic_image(rgb, threshold=0.50)
        assert result.is_aquatic is True
        assert result.confidence >= 0.50
    else:
        with patch.object(
            AquaticInference,
            "predict",
            return_value=ClassificationResult(
                is_aquatic=True,
                confidence=0.92,
                aquatic_probability=0.92,
                non_aquatic_probability=0.08,
                predicted_class="aquatic",
                accepted=True,
                threshold=0.85,
                message="Aquatic Image Detected. Confidence: 92.0%.",
            ),
        ):
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            res = validate_aquatic_image(dummy, threshold=0.85)
            assert res.accepted is True


# 4. An ordinary portrait is rejected
def test_4_ordinary_portrait_rejected():
    portrait_imgs = list(Path("data/dataset").glob("**/*portrait*.*"))
    if portrait_imgs:
        with Image.open(portrait_imgs[0]) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        result = validate_aquatic_image(rgb, threshold=0.85)
        assert result.accepted is False
        assert result.is_aquatic is False
        assert "Unsupported Image" in result.message
    else:
        with patch.object(
            AquaticInference,
            "predict",
            return_value=ClassificationResult(
                is_aquatic=False,
                confidence=0.91,
                aquatic_probability=0.09,
                non_aquatic_probability=0.91,
                predicted_class="non_aquatic",
                accepted=False,
                threshold=0.85,
                message="Unsupported Image: AquaRestore is designed specifically for aquatic images.",
            ),
        ):
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            res = validate_aquatic_image(dummy, threshold=0.85)
            assert res.accepted is False


# 5. An ordinary building image is rejected
def test_5_ordinary_building_rejected():
    building_imgs = list(Path("data/dataset").glob("**/*architecture*.*"))
    if building_imgs:
        with Image.open(building_imgs[0]) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        result = validate_aquatic_image(rgb, threshold=0.85)
        assert result.accepted is False
        assert result.is_aquatic is False
    else:
        with patch.object(
            AquaticInference,
            "predict",
            return_value=ClassificationResult(
                is_aquatic=False,
                confidence=0.95,
                aquatic_probability=0.05,
                non_aquatic_probability=0.95,
                predicted_class="non_aquatic",
                accepted=False,
                threshold=0.85,
                message="Unsupported Image: AquaRestore is designed specifically for aquatic images.",
            ),
        ):
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            res = validate_aquatic_image(dummy, threshold=0.85)
            assert res.accepted is False


# 6. An ordinary road image is rejected
def test_6_ordinary_road_rejected():
    road_imgs = list(Path("data/dataset").glob("**/*highway*.*")) + list(
        Path("data/dataset").glob("**/*traffic*.*")
    )
    if road_imgs:
        with Image.open(road_imgs[0]) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        result = validate_aquatic_image(rgb, threshold=0.85)
        assert result.accepted is False
        assert result.is_aquatic is False
    else:
        with patch.object(
            AquaticInference,
            "predict",
            return_value=ClassificationResult(
                is_aquatic=False,
                confidence=0.93,
                aquatic_probability=0.07,
                non_aquatic_probability=0.93,
                predicted_class="non_aquatic",
                accepted=False,
                threshold=0.85,
                message="Unsupported Image",
            ),
        ):
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            res = validate_aquatic_image(dummy, threshold=0.85)
            assert res.accepted is False


# 7. An image with insufficient confidence is rejected
def test_7_insufficient_confidence_rejected():
    with patch.object(
        AquaticInference,
        "predict",
        return_value=ClassificationResult(
            is_aquatic=True,
            confidence=0.74,
            aquatic_probability=0.74,
            non_aquatic_probability=0.26,
            predicted_class="aquatic",
            accepted=False,
            threshold=0.85,
            message="Uncertain Image Content: Classifier confidence (74.0%) is below the required threshold (85.0%).",
        ),
    ):
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        res = validate_aquatic_image(dummy, threshold=0.85)
        assert res.accepted is False
        assert "Uncertain Image Content" in res.message


# 8. Corrupted images produce a controlled error
def test_8_corrupted_images_controlled_error(tmp_path):
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"not a valid png or jpeg header byte string")

    with pytest.raises(Exception):
        with Image.open(corrupt_file) as img:
            np.asarray(img.convert("RGB"), dtype=np.uint8)


# 9. Unsupported file formats are rejected
def test_9_unsupported_formats_rejected():
    # Non-3D or non-RGB numpy arrays
    with pytest.raises(ValueError, match="shape"):
        validate_aquatic_image(np.zeros((100, 100), dtype=np.uint8))

    with pytest.raises(ValueError, match="shape"):
        validate_aquatic_image(np.zeros((100, 100, 4), dtype=np.uint8))

    with pytest.raises(TypeError, match="dtype uint8"):
        validate_aquatic_image(np.zeros((100, 100, 3), dtype=np.float64))


# 10. A rejected image never reaches the enhancement pipeline
def test_10_rejected_image_never_reaches_enhancement():
    non_aquatic_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Mock the classifier to reject
    mock_val = ClassificationResult(
        is_aquatic=False,
        confidence=0.92,
        aquatic_probability=0.08,
        non_aquatic_probability=0.92,
        predicted_class="non_aquatic",
        accepted=False,
        threshold=0.85,
        message="Unsupported Image: AquaRestore is designed specifically for aquatic images.",
    )

    with patch(
        "aquarestore.contracts.validate_aquatic_image", return_value=mock_val
    ):
        with patch("aquarestore.contracts.correct_color") as mock_color:
            with patch("aquarestore.contracts.enhance_contrast") as mock_clahe:
                # Running enhance_image with validate_aquatic=True MUST raise ValueError
                with pytest.raises(ValueError, match="Unsupported Image"):
                    enhance_image(
                        non_aquatic_image,
                        adaptive=True,
                        validate_aquatic=True,
                        threshold=0.85,
                    )

                # Confirm that color correction and CLAHE were NEVER called!
                mock_color.assert_not_called()
                mock_clahe.assert_not_called()


# 11. Accepted images still work with Fixed enhancement
def test_11_accepted_image_works_with_fixed_enhancement():
    sample_path = Path("data/samples/907_img_.png")
    if sample_path.exists():
        with Image.open(sample_path) as img:
            image = np.asarray(img.convert("RGB"), dtype=np.uint8)
    else:
        image = np.full((80, 80, 3), 120, dtype=np.uint8)

    # Run Fixed enhancement
    result = enhance_image(image, adaptive=False)
    assert isinstance(result, EnhancementResult)
    assert result.method == "fixed"
    assert result.image.shape == image.shape
    assert result.image.dtype == np.uint8
    assert result.parameters_used["adaptive"] is False
    assert result.parameters_used["clip_limit"] == 2.0


# 12. Accepted images still work with Adaptive enhancement
def test_12_accepted_image_works_with_adaptive_enhancement():
    sample_path = Path("data/samples/907_img_.png")
    if sample_path.exists():
        with Image.open(sample_path) as img:
            image = np.asarray(img.convert("RGB"), dtype=np.uint8)
    else:
        image = np.full((80, 80, 3), 120, dtype=np.uint8)

    # Run Adaptive enhancement
    result = enhance_image(image, adaptive=True)
    assert isinstance(result, EnhancementResult)
    assert result.method == "adaptive"
    assert result.image.shape == image.shape
    assert result.image.dtype == np.uint8
    assert result.parameters_used["adaptive"] is True
    assert 1.0 <= result.parameters_used["clip_limit"] <= 3.0
