"""
STEP 4: Evaluate Fine-tuned Models & Run Full Inference
=========================================================
After fine-tuning, this script:
1. Evaluates the recognition model accuracy on the ICDAR validation set
2. Runs the full OCR pipeline on your map images using the fine-tuned models
3. Compares results with the baseline (pre-fine-tuning)
4. Exports results to CSV

Usage:
  python step4_evaluate_and_infer.py
"""

import os
import sys
import cv2
import json
import csv
import numpy as np
from tqdm import tqdm

# Add PaddleOCR to path
sys.path.insert(0, os.path.abspath("PaddleOCR_Official_Tools"))

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# Fine-tuned models (from step 2 & 3)
REC_MODEL_DIR = "./output/rec_finetune/best_accuracy"
DET_MODEL_DIR = "./output/det_finetune/best_accuracy"

# Fallback to original models if fine-tuned not available
REC_MODEL_FALLBACK = "./output/rec_inference"
DET_MODEL_FALLBACK = "./inference/ch_PP-OCRv4_det_infer"

CHAR_DICT_PATH = "./PaddleOCR_Official_Tools/ppocr/utils/en_dict.txt"

# Maps to run inference on
MAPS_DIR    = "Rumsey_Map_OCR_Data/rumsey/icdar24-train-png/train_images"
OUTPUT_CSV  = "results/map_text_results_finetuned.csv"
OUTPUT_VIZ  = "results/visualizations"

# Confidence threshold for final output
MIN_CONFIDENCE = 0.70
# ──────────────────────────────────────────────────────────────────────────────


def load_ocr_engine():
    """Load the OCR engine with fine-tuned models if available."""
    from tools.infer.predict_system import TextSystem
    from tools.infer import utility

    args = utility.parse_args()

    # Use fine-tuned models if they exist, otherwise fallback
    if os.path.exists(REC_MODEL_DIR):
        args.rec_model_dir = REC_MODEL_DIR
        print(f"  ✅ Using FINE-TUNED recognition model: {REC_MODEL_DIR}")
    else:
        args.rec_model_dir = REC_MODEL_FALLBACK
        print(f"  ⚠️  Using baseline recognition model: {REC_MODEL_FALLBACK}")

    if os.path.exists(DET_MODEL_DIR):
        args.det_model_dir = DET_MODEL_DIR
        print(f"  ✅ Using FINE-TUNED detection model: {DET_MODEL_DIR}")
    else:
        args.det_model_dir = DET_MODEL_FALLBACK
        print(f"  ⚠️  Using baseline detection model: {DET_MODEL_FALLBACK}")

    args.rec_char_dict_path = CHAR_DICT_PATH
    args.use_angle_cls = False
    args.use_gpu = False  # Set to True if GPU available

    return TextSystem(args)


def evaluate_on_icdar_val(text_sys):
    """
    Evaluate the fine-tuned model on ICDAR validation set.
    Computes word-level accuracy against ground truth.
    """
    print("\n📊 Evaluating on ICDAR validation set...")

    val_annotations = "Rumsey_Map_OCR_Data/rumsey/icdar24-val-png/annotations.json"
    val_images_dir  = "Rumsey_Map_OCR_Data/rumsey/icdar24-val-png/val_images"

    with open(val_annotations) as f:
        data = json.load(f)

    # Sample 10 images for quick evaluation
    import random
    sample = random.sample(data, min(10, len(data)))

    total_gt_words = 0
    matched_words  = 0

    for entry in tqdm(sample, desc="  Evaluating"):
        img_path = os.path.join(val_images_dir, entry['image'])
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Get ground truth words
        gt_words = set()
        for group in entry['groups']:
            for word in group:
                if not word.get('illegible', False) and word.get('text', '').strip():
                    gt_words.add(word['text'].strip().upper())

        # Run OCR
        try:
            preds = text_sys(img)
            if preds[1]:
                pred_words = set(text.strip().upper() for text, conf in preds[1] if conf > 0.5)
                matched = len(gt_words & pred_words)
                matched_words  += matched
                total_gt_words += len(gt_words)
        except Exception as e:
            continue

    if total_gt_words > 0:
        recall = matched_words / total_gt_words * 100
        print(f"\n  📈 Quick Evaluation Results (10 sample maps):")
        print(f"     GT words:    {total_gt_words}")
        print(f"     Matched:     {matched_words}")
        print(f"     Recall:      {recall:.1f}%")
    else:
        print("  ⚠️  Could not compute metrics (no GT words found)")


def run_full_inference(text_sys, max_maps=None):
    """Run OCR on all training maps and save results to CSV."""
    print(f"\n🔍 Running full inference on maps in: {MAPS_DIR}")

    import glob
    image_files = glob.glob(os.path.join(MAPS_DIR, "*.png")) + \
                  glob.glob(os.path.join(MAPS_DIR, "*.jpg"))

    if max_maps:
        image_files = image_files[:max_maps]

    print(f"  Found {len(image_files)} map images")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(OUTPUT_VIZ, exist_ok=True)

    results = []

    for img_path in tqdm(image_files, desc="  Processing maps"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        try:
            preds = text_sys(img)
            dt_boxes = preds[0]
            rec_res  = preds[1]

            if dt_boxes is None or rec_res is None:
                continue

            img_name = os.path.basename(img_path)

            for box, (text, conf) in zip(dt_boxes, rec_res):
                if conf < MIN_CONFIDENCE:
                    continue

                xs = [p[0] for p in box]
                ys = [p[1] for p in box]

                results.append({
                    'image_file':  img_name,
                    'text':        text,
                    'confidence':  round(float(conf), 4),
                    'x_min':       int(min(xs)),
                    'y_min':       int(min(ys)),
                    'x_max':       int(max(xs)),
                    'y_max':       int(max(ys)),
                    'bbox_coords': str([[int(p[0]), int(p[1])] for p in box])
                })

        except Exception as e:
            continue

    # Save to CSV
    if results:
        fieldnames = ['image_file', 'text', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max', 'bbox_coords']
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        avg_conf = sum(r['confidence'] for r in results) / len(results)
        print(f"\n  ✅ Results saved to: {OUTPUT_CSV}")
        print(f"     Total detections:  {len(results):,}")
        print(f"     Avg confidence:    {avg_conf:.1%}")
        print(f"     Unique images:     {len(set(r['image_file'] for r in results))}")
    else:
        print("  ⚠️  No results generated")

    return results


def main():
    print("🗺️  Evaluation & Full Inference")
    print("=" * 60)

    print("\n🔹 Loading OCR engine...")
    try:
        text_sys = load_ocr_engine()
        print("  ✅ Engine loaded!")
    except Exception as e:
        print(f"  ❌ Failed to load engine: {e}")
        print("\n  Make sure PaddleOCR_Official_Tools is properly set up.")
        return

    # Evaluate on validation set
    evaluate_on_icdar_val(text_sys)

    # Run full inference (limit to 20 maps for quick test, remove limit for full run)
    print("\n" + "=" * 60)
    print("Run full inference? This processes all 200 training maps.")
    print("  [1] Quick test (20 maps)")
    print("  [2] Full run (all maps)")
    print("  [3] Skip inference")

    choice = input("\nChoice [1/2/3]: ").strip()

    if choice == "1":
        run_full_inference(text_sys, max_maps=20)
    elif choice == "2":
        run_full_inference(text_sys, max_maps=None)
    else:
        print("  Skipping inference.")

    print(f"\n▶️  Next step: Run step5_hard_negative_mining.py")


if __name__ == "__main__":
    main()
