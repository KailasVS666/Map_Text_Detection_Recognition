"""
Prepare Detection Training Data
================================
Converts ICDAR 2024 map annotations (JSON) to PaddleOCR detection label format.

Output format (one line per image):
    relative/path/to/image.png\t[{"transcription": "word", "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]},...]\n

Run from the root Rumsey_Map_OCR/ directory:
    python training_setup/prepare_det_data.py
"""

import os
import json
import math
import random
import shutil

# ── CONFIG ────────────────────────────────────────────────────────────────────
ICDAR_DATA_ROOT = "Rumsey_Map_OCR_Data/rumsey"
TRAIN_ANNOT     = f"{ICDAR_DATA_ROOT}/icdar24-train-png/annotations.json"
TRAIN_IMGS_DIR  = f"{ICDAR_DATA_ROOT}/icdar24-train-png/train_images"
VAL_ANNOT       = f"{ICDAR_DATA_ROOT}/icdar24-val-png/annotations.json"
VAL_IMGS_DIR    = f"{ICDAR_DATA_ROOT}/icdar24-val-png/val_images"

OUTPUT_DIR      = "train_data/det"
VAL_SPLIT       = 0.1      # if no val set, use 10% of train
MIN_TEXT_SIZE   = 8        # skip tiny boxes (pixels)
# ──────────────────────────────────────────────────────────────────────────────


def polygon_to_quad(points):
    """Convert a polygon with N points to a 4-point bounding quad."""
    if len(points) == 4:
        return points
    # Use min bounding rectangle approach for >4 points
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ]


def box_size(quad):
    """Return approximate size of a quad (diagonal length)."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return math.sqrt(w * w + h * h)


def convert_annotations(annot_path, imgs_dir, output_label_file, imgs_output_dir):
    """Convert a single annotation file and copy images."""
    os.makedirs(imgs_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_label_file), exist_ok=True)

    with open(annot_path, encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    skipped_imgs   = 0
    total_boxes    = 0
    skipped_boxes  = 0

    for entry in data:
        img_name = entry["image"]
        src_img  = os.path.join(imgs_dir, img_name)

        if not os.path.exists(src_img):
            skipped_imgs += 1
            continue

        # Copy image to output dir
        dst_img = os.path.join(imgs_output_dir, img_name)
        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)

        annotations = []
        for group in entry.get("groups", []):
            for word in group:
                total_boxes += 1
                illegible = word.get("illegible", False)
                text      = word.get("text", "").strip()

                # Get polygon from 'polygon' field or fall back to bbox
                if "polygon" in word and word["polygon"]:
                    raw_pts = word["polygon"]
                    pts = [[int(p[0]), int(p[1])] for p in raw_pts]
                elif "bbox" in word:
                    x, y, w, h = word["bbox"]
                    pts = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                else:
                    skipped_boxes += 1
                    continue

                quad = polygon_to_quad(pts)

                # Skip tiny boxes
                if box_size(quad) < MIN_TEXT_SIZE:
                    skipped_boxes += 1
                    continue

                annotations.append({
                    "transcription": "###" if illegible else (text if text else "###"),
                    "points": quad,
                })

        if annotations:
            # Path relative to OUTPUT_DIR (what PaddleOCR expects)
            rel_img = os.path.relpath(dst_img, OUTPUT_DIR).replace("\\", "/")
            label   = json.dumps(annotations, ensure_ascii=False)
            lines.append(f"{rel_img}\t{label}\n")

    with open(output_label_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"  Wrote {len(lines):,} image entries "
          f"({total_boxes - skipped_boxes:,} boxes kept, "
          f"{skipped_boxes} skipped) → {output_label_file}")
    if skipped_imgs:
        print(f"  ⚠️  {skipped_imgs} images not found and skipped.")

    return lines


def main():
    print("🔷 Preparing Detection Training Data")
    print("=" * 50)

    train_imgs_out = os.path.join(OUTPUT_DIR, "train_images")
    val_imgs_out   = os.path.join(OUTPUT_DIR, "val_images")

    # ── Training data ──────────────────────────────────────────────────────
    print("\n📂 Processing TRAINING annotations...")
    train_lines = convert_annotations(
        annot_path        = TRAIN_ANNOT,
        imgs_dir          = TRAIN_IMGS_DIR,
        output_label_file = os.path.join(OUTPUT_DIR, "train_list.txt"),
        imgs_output_dir   = train_imgs_out,
    )

    # ── Validation data ────────────────────────────────────────────────────
    if os.path.exists(VAL_ANNOT):
        print("\n📂 Processing VALIDATION annotations...")
        convert_annotations(
            annot_path        = VAL_ANNOT,
            imgs_dir          = VAL_IMGS_DIR,
            output_label_file = os.path.join(OUTPUT_DIR, "val_list.txt"),
            imgs_output_dir   = val_imgs_out,
        )
    else:
        print(f"\n⚠️  No validation annotation found at {VAL_ANNOT}")
        print(f"   Splitting {VAL_SPLIT:.0%} of training data for validation...")
        random.seed(42)
        random.shuffle(train_lines)
        split    = max(1, int(len(train_lines) * VAL_SPLIT))
        val_lines   = train_lines[:split]
        train_lines = train_lines[split:]

        # Re-write train list (without val samples)
        with open(os.path.join(OUTPUT_DIR, "train_list.txt"), "w", encoding="utf-8") as f:
            f.writelines(train_lines)
        with open(os.path.join(OUTPUT_DIR, "val_list.txt"), "w", encoding="utf-8") as f:
            f.writelines(val_lines)
        print(f"   Train: {len(train_lines)} | Val: {len(val_lines)}")

    print("\n✅ Detection training data ready!")
    print(f"   Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"   train_list.txt + val_list.txt")


if __name__ == "__main__":
    main()
