
"""Generate a reproducible contact sheet of enhancement results."""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def build_contact_sheet(
    input_dir: str | Path = "data/samples",
    output_dir: str | Path = "outputs/evaluation",
    sample_size: int = 12,
    seed: int = 42,
    thumbnail_size: tuple[int, int] = (320, 220),
) -> Path:
    """Create a contact sheet showing original, fixed, and adaptive images."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")

    if (
        len(thumbnail_size) != 2
        or any(not isinstance(value, int) or value <= 0 for value in thumbnail_size)
    ):
        raise ValueError("thumbnail_size must contain two positive integers")

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path.resolve()}")

    files = sorted(
        path for path in input_path.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and (output_path / "fixed" / f"{path.stem}.png").is_file()
        and (output_path / "adaptive" / f"{path.stem}.png").is_file()
    )

    if not files:
        raise FileNotFoundError(
            "No matching original images with both fixed and adaptive outputs "
            f"were found. Input: {input_path.resolve()}, "
            f"output: {output_path.resolve()}"
        )

    rng = random.Random(seed)
    selected = sorted(rng.sample(files, min(sample_size, len(files))))

    padding = 12
    label_height = 30
    cell_width, cell_height = thumbnail_size
    headers = ("Original", "Fixed enhancement", "Adaptive enhancement")

    sheet_width = padding + 3 * (cell_width + padding)
    row_height = cell_height + label_height + padding
    sheet_height = padding + label_height + len(selected) * row_height

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    for column, header in enumerate(headers):
        x = padding + column * (cell_width + padding)
        draw.text((x, padding), header, fill="black", font=font)

    for row, original_path in enumerate(selected):
        paths = (
            original_path,
            output_path / "fixed" / f"{original_path.stem}.png",
            output_path / "adaptive" / f"{original_path.stem}.png",
        )

        y = padding + label_height + row * row_height

        for column, image_path in enumerate(paths):
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                image.thumbnail(thumbnail_size)

                cell = Image.new("RGB", thumbnail_size, "white")
                offset = (
                    (cell_width - image.width) // 2,
                    (cell_height - image.height) // 2,
                )
                cell.paste(image, offset)

            x = padding + column * (cell_width + padding)
            sheet.paste(cell, (x, y))

            if column == 0:
                draw.text(
                    (x, y + cell_height + 3),
                    original_path.name,
                    fill="black",
                    font=font,
                )

    output_path.mkdir(parents=True, exist_ok=True)
    sheet_path = output_path / "contact_sheet.png"
    sheet.save(sheet_path)

    print(f"Images included: {len(selected)}")
    print(f"Contact sheet saved to: {sheet_path.resolve()}")

    return sheet_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare original, fixed, and adaptive enhancement images."
    )
    parser.add_argument("--input-dir", default="data/samples")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_contact_sheet(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

