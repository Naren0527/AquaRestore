"""Dataset preparation and acquisition script for AquaRestore aquatic classifier.

Downloads open-licensed Creative Commons images from Wikimedia Commons for diverse
aquatic and non-aquatic categories, combines with local UIEB underwater samples,
validates integrity, removes duplicates, and splits into train/val/test sets.
Uses concurrent downloads for high throughput.
"""

import concurrent.futures
import hashlib
import json
import os
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path
from PIL import Image

RANDOM_SEED = 42

AQUATIC_SEARCH_QUERIES = {
    "waterfall": 50,
    "swimming pool": 50,
    "river landscape water": 50,
    "lake water natural": 50,
    "ocean waves sea": 50,
    "coral reef marine": 50,
}

NON_AQUATIC_SEARCH_QUERIES = {
    "portrait photography": 40,
    "modern architecture": 40,
    "highway road": 40,
    "desert landscape": 40,
    "living room interior": 40,
    "kitchen interior": 40,
    "office room desk": 40,
    "city street traffic": 40,
    "rocky mountain peak": 40,
    "forest trees": 40,
    "clothing fashion": 30,
    "blue car automobile": 40,  # Hard negative (blue color)
    "blue sky sunny": 40,       # Hard negative (blue color)
    "blue wall room": 30,       # Hard negative (blue color)
}


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch_wikimedia_images(
    query: str, max_count: int, output_dir: Path, license_log: list
) -> list[Path]:
    """Fetch open-licensed images from Wikimedia Commons concurrently."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    encoded_query = urllib.parse.quote(query)
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&format=json"
        f"&generator=search&gsrnamespace=6&gsrsearch={encoded_query}&gsrlimit={max_count * 2}"
        f"&prop=imageinfo&iiprop=url|size|mime|extmetadata&iiurlwidth=800"
    )
    headers = {
        "User-Agent": "AquaRestoreResearchBot/1.0 (academic research; contact@aquarestore.org)"
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Error fetching metadata for query '{query}': {e}", flush=True)
        return saved_files

    pages = data.get("query", {}).get("pages", {})
    candidates = []

    for page_id, info in pages.items():
        if len(candidates) >= max_count:
            break
        imageinfo = info.get("imageinfo", [{}])[0]
        mime = imageinfo.get("mime", "")
        if mime not in ("image/jpeg", "image/png"):
            continue

        file_url = imageinfo.get("thumburl") or imageinfo.get("url")
        if not file_url:
            continue

        ext_metadata = imageinfo.get("extmetadata", {})
        license_name = ext_metadata.get("LicenseShortName", {}).get(
            "value", "CC / Public Domain"
        )
        artist = ext_metadata.get("Artist", {}).get(
            "value", "Wikimedia Contributor"
        )

        candidates.append((file_url, license_name, artist))

    def download_candidate(idx: int, item: tuple) -> Path | None:
        file_url, license_name, artist = item
        filename = f"{query.replace(' ', '_')[:16]}_{idx:03d}.jpg"
        target_path = output_dir / filename

        if target_path.exists() and target_path.stat().st_size > 1024:
            return target_path

        try:
            dl_req = urllib.request.Request(file_url, headers=headers)
            with urllib.request.urlopen(dl_req, timeout=15) as dl_resp:
                img_data = dl_resp.read()

            with open(target_path, "wb") as f:
                f.write(img_data)

            with Image.open(target_path) as img:
                img_rgb = img.convert("RGB")
                if img_rgb.size[0] < 64 or img_rgb.size[1] < 64:
                    target_path.unlink()
                    return None
                if max(img_rgb.size) > 1024:
                    img_rgb.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                img_rgb.save(target_path, "JPEG", quality=90)

            license_log.append(
                {
                    "file": filename,
                    "query": query,
                    "url": file_url,
                    "license": license_name,
                    "artist": artist[:100] if artist else "",
                    "source": "Wikimedia Commons",
                }
            )
            return target_path
        except Exception:
            if target_path.exists():
                try:
                    target_path.unlink()
                except OSError:
                    pass
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(download_candidate, i, c)
            for i, c in enumerate(candidates)
        ]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res is not None:
                saved_files.append(res)

    print(
        f"Query '{query}': downloaded {len(saved_files)} images.", flush=True
    )
    return saved_files


def prepare_dataset(
    samples_dir: Path, output_base: Path, num_uieb_samples: int = 400
):
    print("=" * 60, flush=True)
    print("STARTING DATASET PREPARATION PIPELINE", flush=True)
    print("=" * 60, flush=True)
    random.seed(RANDOM_SEED)

    raw_dir = output_base / "raw"
    raw_aquatic = raw_dir / "aquatic"
    raw_non_aquatic = raw_dir / "non_aquatic"
    raw_aquatic.mkdir(parents=True, exist_ok=True)
    raw_non_aquatic.mkdir(parents=True, exist_ok=True)

    licenses = []

    # 1. Process local UIEB underwater samples
    print(
        f"\n1. Sampling {num_uieb_samples} images from UIEB underwater dataset ({samples_dir})...",
        flush=True,
    )
    uieb_files = [f for f in samples_dir.glob("*.png")] + [
        f for f in samples_dir.glob("*.jpg")
    ]
    random.shuffle(uieb_files)
    selected_uieb = uieb_files[:num_uieb_samples]

    uieb_count = 0
    for fpath in selected_uieb:
        try:
            dst = raw_aquatic / f"uieb_{fpath.stem}.jpg"
            if not dst.exists() or dst.stat().st_size == 0:
                with Image.open(fpath) as img:
                    img_rgb = img.convert("RGB")
                    if max(img_rgb.size) > 1024:
                        img_rgb.thumbnail(
                            (1024, 1024), Image.Resampling.LANCZOS
                        )
                    img_rgb.save(dst, "JPEG", quality=92)
            licenses.append(
                {
                    "file": dst.name,
                    "query": "UIEB underwater benchmark",
                    "url": "local/data/samples",
                    "license": "Academic Research Use (UIEB)",
                    "source": "UIEB Dataset",
                }
            )
            uieb_count += 1
        except Exception as e:
            print(f"Skipping corrupt sample {fpath}: {e}", flush=True)
    print(
        f"Imported/verified {uieb_count} valid UIEB underwater images.",
        flush=True,
    )

    # 2. Download additional aquatic images
    print(
        "\n2. Downloading diverse aquatic images (waterfalls, pools, rivers, lakes, waves)...",
        flush=True,
    )
    for query, count in AQUATIC_SEARCH_QUERIES.items():
        fetch_wikimedia_images(query, count, raw_aquatic, licenses)

    # 3. Download non-aquatic images
    print(
        "\n3. Downloading diverse non-aquatic images (portraits, buildings, roads, forests, blue objects)...",
        flush=True,
    )
    for query, count in NON_AQUATIC_SEARCH_QUERIES.items():
        fetch_wikimedia_images(query, count, raw_non_aquatic, licenses)

    # 4. Deduplicate and validate all raw images
    print("\n4. Validating and deduplicating images...", flush=True)

    def clean_category(category_dir: Path) -> list[Path]:
        seen_hashes = set()
        valid_files = []
        for p in category_dir.glob("*.*"):
            try:
                sha = compute_sha256(p)
                if sha in seen_hashes:
                    p.unlink()
                    continue
                seen_hashes.add(sha)

                with Image.open(p) as img:
                    img.convert("RGB")
                    if img.size[0] < 64 or img.size[1] < 64:
                        p.unlink()
                        continue
                valid_files.append(p)
            except Exception:
                if p.exists():
                    p.unlink()
        return valid_files

    valid_aquatic = clean_category(raw_aquatic)
    valid_non_aquatic = clean_category(raw_non_aquatic)

    print(f"Total valid aquatic images: {len(valid_aquatic)}", flush=True)
    print(
        f"Total valid non-aquatic images: {len(valid_non_aquatic)}", flush=True
    )

    min_class_count = min(len(valid_aquatic), len(valid_non_aquatic))
    print(
        f"Balancing classes to {min_class_count} images per class...",
        flush=True,
    )
    random.shuffle(valid_aquatic)
    random.shuffle(valid_non_aquatic)
    final_aquatic = sorted(
        valid_aquatic[:min_class_count], key=lambda p: p.name
    )
    final_non_aquatic = sorted(
        valid_non_aquatic[:min_class_count], key=lambda p: p.name
    )

    # 5. Split into train (70%), val (15%), test (15%)
    splits = {"train": (0.0, 0.70), "val": (0.70, 0.85), "test": (0.85, 1.0)}
    split_counts = {}

    for split_name, (start_frac, end_frac) in splits.items():
        for class_name, files in [
            ("aquatic", final_aquatic),
            ("non_aquatic", final_non_aquatic),
        ]:
            start_idx = int(len(files) * start_frac)
            end_idx = int(len(files) * end_frac)
            split_files = files[start_idx:end_idx]

            target_dir = output_base / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            # Clear old files if any
            for existing in target_dir.glob("*.*"):
                try:
                    existing.unlink()
                except OSError:
                    pass

            for f in split_files:
                dest = target_dir / f.name
                if not dest.exists():
                    with open(f, "rb") as src_f, open(dest, "wb") as dst_f:
                        dst_f.write(src_f.read())

            split_counts[f"{split_name}_{class_name}"] = len(split_files)

    # Save license and metadata
    metadata = {
        "random_seed": RANDOM_SEED,
        "classes": ["non_aquatic", "aquatic"],
        "class_mapping": {"0": "non_aquatic", "1": "aquatic"},
        "counts": split_counts,
        "total_images": sum(split_counts.values()),
        "aquatic_total": len(final_aquatic),
        "non_aquatic_total": len(final_non_aquatic),
    }

    with open(output_base / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(output_base / "licenses.json", "w") as f:
        json.dump(licenses, f, indent=2)

    print("\nDataset preparation completed successfully!", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    samples = Path("data/samples")
    out_dir = Path("data/dataset")
    prepare_dataset(samples, out_dir, num_uieb_samples=400)
