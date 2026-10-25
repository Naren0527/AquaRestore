
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

INPUT_DIR = Path("data/samples")
OUTPUT_DIR = Path("outputs/evaluation")
SHEET_PATH = OUTPUT_DIR / "contact_sheet.png"

SAMPLE_SIZE = 12
THUMBNAIL_SIZE = (320, 220)
PADDING = 12
LABEL_HEIGHT = 28


def main():
    rng = random.Random(42)

    files = sorted(INPUT_DIR.glob("*.png"))
    files = [p for p in files if
             (OUTPUT_DIR / "fixed" / f"{p.stem}.png").is_file()
             and (OUTPUT_DIR / "adaptive" / f"{p.stem}.png").is_file()]

    if not files:
        raise FileNotFoundError("No matching original and enhanced images found.")

    selected = sorted(rng.sample(files, min(SAMPLE_SIZE, len(files))))

    cell_width = THUMBNAIL_SIZE[0]
    cell_height = THUMBNAIL_SIZE[1] + LABEL_HEIGHT
    sheet_width = PADDING * 4 + cell_width * 3
    sheet_height = PADDING + len(selected) * (cell_height + PADDING)

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    headers = ["Original", "Fixed", "Adaptive"]
    for col, header in enumerate(headers):
        x = PADDING + col * (cell_width + PADDING)
        draw.text((x, 4), header, fill="black", font=font)

    for row, original_path in enumerate(selected):
        images = [
            Image.open(original_path).convert("RGB"),
            Image.open(OUTPUT_DIR / "fixed" / f"{original_path.stem}.png").convert("RGB"),
            Image.open(OUTPUT_DIR / "adaptive" / f"{original_path.stem}.png").convert("RGB"),
        ]

        y = PADDING + LABEL_HEIGHT + row * (cell_height + PADDING)

        for col, image in enumerate(images):
            image.thumbnail(THUMBNAIL_SIZE)
            cell = Image.new("RGB", THUMBNAIL_SIZE, "white")
            offset = (
                (THUMBNAIL_SIZE[0] - image.width) // 2,
                (THUMBNAIL_SIZE[1] - image.height) // 2,
            )
            cell.paste(image, offset)

            x = PADDING + col * (cell_width + PADDING)
            sheet.paste(cell, (x, y))
            if col == 0:
                draw.text(
                    (x, y + THUMBNAIL_SIZE[1] + 2),
                    original_path.name,
                    fill="black",
                    font=font,
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.save(SHEET_PATH)
    print(f"Saved contact sheet: {SHEET_PATH.resolve()}")


if __name__ == "__main__":
    main()
