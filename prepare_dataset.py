"""Create a reproducible, balanced manifest from label/image folders. No API calls."""

import argparse
import csv
import json
import os
from pathlib import Path
import random

from PIL import Image

LABELS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def prepare(image_root, output, per_class=10, seed=42, split="test"):
    image_root, output = Path(image_root).resolve(), Path(output).resolve()
    if per_class < 0:
        raise ValueError("per-class must be >= 0 (0 means all images)")
    if output.exists() or output.with_suffix(".meta.json").exists():
        raise ValueError("Output exists; choose another --output to preserve the sample.")
    rng = random.Random(seed)
    rows, counts = [], {}
    for label in LABELS:
        folder = image_root / label
        candidates = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS and p.is_file())
        if not candidates:
            raise ValueError(f"No images found: {folder}")
        if per_class and len(candidates) < per_class:
            raise ValueError(f"{label}: requested {per_class}, available {len(candidates)}")
        selected = rng.sample(candidates, per_class) if per_class else candidates
        for path in selected:
            with Image.open(path) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError(f"Animated image is unsupported: {path}")
                image.verify()
            rows.append({"path": os.path.relpath(path, output.parent), "label": label, "split": split})
        counts[label] = len(selected)
    rng.shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".meta.json").write_text(json.dumps({
        "image_root": str(image_root), "seed": seed, "per_class": per_class,
        "split": split, "counts": counts, "total": len(rows),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True, help="Folder containing the seven label folders")
    parser.add_argument("--output", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--per-class", type=int, default=10, help="0 = all; default: 10 per label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    try:
        counts = prepare(args.image_root, args.output, args.per_class, args.seed, args.split)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Dataset error: {exc}\n")
    print(json.dumps({"manifest": str(args.output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
