"""
STEP 1: Extract Word Crops from ICDAR 2024 Ground Truth
========================================================
This script uses the OFFICIAL human-labeled annotations from the ICDAR 2024
dataset to extract word-level image crops. These crops are the gold standard
training data for fine-tuning the recognition model.

Dataset Stats:
  - 200 training maps → ~34,521 labeled word crops
  - 40 validation maps → ~5,543 labeled word crops

Output Format (PaddleOCR compatible):
  train_data/rec/train/  ← word crop images
  train_data/rec/val/    ← validation crop images
  train_data/rec/train_list.txt  ← "path/to/crop.jpg\tword_text"
  train_data/rec/val_list.txt

Usage:
  python step1_extract_icdar_crops.py
"""

import json
import os
import cv2
import numpy as np
from tqdm import tqdm

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
TRAIN_ANNOTATIONS = "Rumsey_Map_OCR_Data/rumsey/icdar24-train-png/annotations.json"
TRAIN_IMAGES_DIR  = "Rumsey_Map_OCR_Data/rumsey/icdar24-train-png/train_images"

VAL_ANNOTATIONS   = "Rumsey_Map_OCR_Data/rumsey/icdar24-val-png/annotations.json"
VAL_IMAGES_DIR    = "Rumsey_Map_OCR_Data/rumsey/icdar24-val-png/val_images"

OUTPUT_DIR        = "train_data/rec"
MIN_CROP_SIZE     = 8   # Minimum width/height in pixels to keep a crop
# ──────────────────────────────────────────────────────────────────────────────


def polygon_to_crop(img, vertices):
    """Crop a word region using its polygon vertices (handles rotated text)."""
    pts = np.array(vertices, dtype=np.float32)

    # Get bounding box of the polygon
    x_min = max(0, int(np.min(pts[:, 0])))
    x_max = min(img.shape[1], int(np.max(pts[:, 0])) + 1)
    y_min = max(0, int(np.min(pts[:, 1])))
    y_max = min(img.shape[0], int(np.max(pts[:, 1])) + 1)

    if x_max - x_min < MIN_CROP_SIZE or y_max - y_min < MIN_CROP_SIZE:
        return None

    crop = img[y_min:y_max, x_min:x_max]
    return crop


def process_split(annotations_path, images_dir, output_images_dir, label_file_path, split_name):
    """Process one split (train or val) of the ICDAR dataset."""
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split...")
    print(f"{'='*60}")

    os.makedirs(output_images_dir, exist_ok=True)

    with open(annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    labels = []
    skipped_illegible = 0
    skipped_small = 0
    skipped_missing = 0
    saved = 0

    for entry in tqdm(data, desc=f"  Extracting {split_name} crops"):
        image_name = entry['image']
        img_path = os.path.join(images_dir, image_name)

        if not os.path.exists(img_path):
            skipped_missing += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skipped_missing += 1
            continue

        # Each entry has 'groups' — a list of text lines
        # Each group is a list of word dicts with 'vertices', 'text', 'illegible'
        for group_idx, group in enumerate(entry['groups']):
            for word_idx, word in enumerate(group):

                # Skip illegible words (no ground truth text)
                if word.get('illegible', False):
                    skipped_illegible += 1
                    continue

                text = word.get('text', '').strip()
                if not text:
                    skipped_illegible += 1
                    continue

                vertices = word['vertices']

                # Extract the crop
                crop = polygon_to_crop(img, vertices)
                if crop is None:
                    skipped_small += 1
                    continue

                # Save crop
                base_name = os.path.splitext(image_name)[0]
                crop_filename = f"{base_name}_g{group_idx}_w{word_idx}.jpg"
                crop_save_path = os.path.join(output_images_dir, crop_filename)
                cv2.imwrite(crop_save_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Record label (relative path from OUTPUT_DIR)
                rel_path = os.path.join(split_name, crop_filename).replace("\\", "/")
                labels.append(f"{rel_path}\t{text}")
                saved += 1

    # Write label file
    with open(label_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels))

    print(f"\n  ✅ {split_name} Results:")
    print(f"     Saved crops:        {saved:,}")
    print(f"     Skipped illegible:  {skipped_illegible:,}")
    print(f"     Skipped too small:  {skipped_small:,}")
    print(f"     Skipped missing:    {skipped_missing:,}")
    print(f"     Label file:         {label_file_path}")
    return saved


def main():
    print("🗺️  ICDAR 2024 Word Crop Extractor")
    print("=" * 60)
    print("This uses OFFICIAL human-labeled ground truth annotations.")
    print("These crops are the gold standard for fine-tuning.\n")

    # Create output directories
    train_out = os.path.join(OUTPUT_DIR, "train")
    val_out   = os.path.join(OUTPUT_DIR, "val")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    # Process training split
    train_saved = process_split(
        annotations_path   = TRAIN_ANNOTATIONS,
        images_dir         = TRAIN_IMAGES_DIR,
        output_images_dir  = train_out,
        label_file_path    = os.path.join(OUTPUT_DIR, "train_list.txt"),
        split_name         = "train"
    )

    # Process validation split
    val_saved = process_split(
        annotations_path   = VAL_ANNOTATIONS,
        images_dir         = VAL_IMAGES_DIR,
        output_images_dir  = val_out,
        label_file_path    = os.path.join(OUTPUT_DIR, "val_list.txt"),
        split_name         = "val"
    )

    print(f"\n{'='*60}")
    print(f"🎉 DONE! Total crops extracted:")
    print(f"   Training:   {train_saved:,} word crops")
    print(f"   Validation: {val_saved:,} word crops")
    print(f"\n📁 Output saved to: {OUTPUT_DIR}/")
    print(f"\n▶️  Next step: Run step2_finetune_recognition.py")


if __name__ == "__main__":
    main()
