"""Quick CLI tool to test individual images against the trained AquaRestore classifier."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from aquarestore.classification.inference import AquaticInference


def main():
    parser = argparse.ArgumentParser(
        description="Test an image with AquaRestore Classifier"
    )
    parser.add_argument("image_path", type=str, help="Path to image file")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/aquatic_classifier.pth",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Confidence threshold (default: 0.85)",
    )
    args = parser.parse_args()

    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}")
        sys.exit(2)

    try:
        with Image.open(img_path) as img:
            rgb_arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception as e:
        print(f"Error reading image: {e}")
        sys.exit(2)

    engine = AquaticInference(
        model_path=args.model_path, threshold=args.threshold
    )
    result = engine.predict(rgb_arr)

    print("\n" + "=" * 60)
    print("AQUARESTORE AQUATIC INTELLIGENCE: IMAGE CLASSIFICATION")
    print("=" * 60)
    print(f"Image:                 {img_path.name}")
    print(f"Resolution:            {rgb_arr.shape[1]}x{rgb_arr.shape[0]}")
    print(f"Predicted Class:       {result.predicted_class.upper()}")
    print(f"Aquatic Probability:   {result.aquatic_probability:.2%}")
    print(f"Non-Aquatic Prob:      {result.non_aquatic_probability:.2%}")
    print(f"Confidence:            {result.confidence:.2%}")
    print(f"Threshold Applied:     {result.threshold:.2%}")
    print(f"Pipeline Status:       {'ACCEPTED' if result.accepted else 'REJECTED'}")
    print(f"Message:               {result.message}")
    print("=" * 60 + "\n")

    sys.exit(0 if result.accepted else 1)


if __name__ == "__main__":
    main()
