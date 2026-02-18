"""
STEP 3: Fine-tune PaddleOCR Detection Model on ICDAR Data
===========================================================
This script prepares the detection training data and launches fine-tuning
of the DBNet text detection model using the ICDAR polygon annotations.

This is what kills the false positive problem — the model learns that
mountain hachures, river lines, and map symbols are NOT text.

Prerequisites:
  - Run step1_extract_icdar_crops.py first (to verify data is accessible)
  - PaddlePaddle installed (GPU strongly recommended for detection)

Usage:
  python step3_finetune_detection.py
"""

import json
import os
import subprocess
import sys
from tqdm import tqdm

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
TRAIN_ANNOTATIONS = "Rumsey_Map_OCR_Data/rumsey/icdar24-train-png/annotations.json"
TRAIN_IMAGES_DIR  = "Rumsey_Map_OCR_Data/rumsey/icdar24-train-png/train_images"

VAL_ANNOTATIONS   = "Rumsey_Map_OCR_Data/rumsey/icdar24-val-png/annotations.json"
VAL_IMAGES_DIR    = "Rumsey_Map_OCR_Data/rumsey/icdar24-val-png/val_images"

OUTPUT_LABELS_DIR = "train_data/det"
OUTPUT_MODEL_DIR  = "./output/det_finetune"

# Pretrained DBNet detection model
# Download: https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_det_train.tar
PRETRAINED_MODEL  = "./pretrained/en_PP-OCRv4_det_train/best_accuracy"

EPOCHS     = 100
BATCH_SIZE = 8   # Detection needs more memory — reduce if OOM
LR         = 0.00005
# ──────────────────────────────────────────────────────────────────────────────


def convert_icdar_to_paddle_det(annotations_path, images_dir, output_label_path, split_name):
    """
    Convert ICDAR polygon annotations to PaddleOCR detection format.
    
    PaddleOCR detection format (one line per image):
    image_path\t[{"transcription": "text", "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}]
    """
    print(f"\n  Converting {split_name} annotations...")

    with open(annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lines = []
    skipped = 0

    for entry in tqdm(data, desc=f"    {split_name}"):
        image_name = entry['image']
        img_path = os.path.join(images_dir, image_name)

        if not os.path.exists(img_path):
            skipped += 1
            continue

        annotations_for_image = []

        for group in entry['groups']:
            for word in group:
                text = word.get('text', '').strip()
                illegible = word.get('illegible', False)
                vertices = word['vertices']

                # For detection, we include illegible regions too
                # (we want the model to detect ALL text, even if unreadable)
                transcription = "###" if illegible or not text else text

                # Convert polygon to list of [x, y] points
                # PaddleOCR expects a 4-point quadrilateral
                pts = [[float(v[0]), float(v[1])] for v in vertices]

                # If more than 4 points, take the bounding box corners
                if len(pts) != 4:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    pts = [
                        [min(xs), min(ys)],
                        [max(xs), min(ys)],
                        [max(xs), max(ys)],
                        [min(xs), max(ys)],
                    ]

                annotations_for_image.append({
                    "transcription": transcription,
                    "points": pts
                })

        if annotations_for_image:
            import json as _json
            ann_str = _json.dumps(annotations_for_image, ensure_ascii=False)
            lines.append(f"{img_path}\t{ann_str}")

    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
    with open(output_label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"  ✅ {split_name}: {len(lines)} images, {skipped} skipped")
    print(f"     Saved to: {output_label_path}")
    return len(lines)


DET_CONFIG_TEMPLATE = """
Global:
  use_gpu: true
  epoch_num: {epochs}
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: {output_dir}
  save_epoch_step: 10
  eval_batch_step: [0, 200]
  cal_metric_during_train: false
  pretrained_model: {pretrained_model}
  checkpoints:
  use_visualdl: false
  infer_img:
  save_inference_dir:

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: {lr}
    warmup_epoch: 2
  regularizer:
    name: 'L2'
    factor: 0

Architecture:
  model_type: det
  algorithm: DB
  Transform:
  Backbone:
    name: ResNet
    layers: 50
    dcn_stage: [False, True, True, True]
    out_indices: [0, 1, 2, 3]
  Neck:
    name: LKPAN
    out_channels: 256
  Head:
    name: DBHead
    k: 50

Loss:
  name: DBLoss
  balance_loss: true
  main_loss_type: DiceLoss
  alpha: 5
  beta: 10
  ohem_ratio: 3

PostProcess:
  name: DBPostProcess
  thresh: 0.3
  box_thresh: 0.6
  max_candidates: 1000
  unclip_ratio: 1.5

Metric:
  name: DetMetric
  main_indicator: hmean

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./
    label_file_list:
      - {train_label}
    ratio_list: [1.0]
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - DetLabelEncode:
      - IaaAugment:
          augmenter_args:
            - type: Fliplr
              args:
                p: 0.5
            - type: Affine
              args:
                rotate:
                  - -10
                  - 10
            - type: Resize
              args:
                size:
                  - 0.5
                  - 3
      - EastRandomCropData:
          size: [960, 960]
          max_tries: 50
          keep_ratio: true
      - MakeBorderMap:
          shrink_ratio: 0.4
          thresh_min: 0.3
          thresh_max: 0.7
      - MakeShrinkMap:
          shrink_ratio: 0.4
          min_text_size: 8
      - NormalizeImage:
          scale: 1./255.
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
          order: hwc
      - ToCHWImage:
      - KeepKeys:
          keep_keys: ['image', 'threshold_map', 'threshold_mask', 'shrink_map', 'shrink_mask']
  loader:
    shuffle: true
    drop_last: false
    batch_size_per_card: {batch_size}
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./
    label_file_list:
      - {val_label}
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - DetLabelEncode:
      - DetResizeForTest:
      - NormalizeImage:
          scale: 1./255.
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
          order: hwc
      - ToCHWImage:
      - KeepKeys:
          keep_keys: ['image', 'shape', 'polys', 'ignore_tags']
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 1
    num_workers: 2
"""


def main():
    print("🗺️  PaddleOCR Detection Fine-tuning")
    print("=" * 60)
    print("This teaches DBNet to detect text specifically in historical maps.")
    print("Reduces false positives on hachures, rivers, and map symbols.\n")

    # Step 1: Convert annotations to PaddleOCR detection format
    print("📋 Converting ICDAR annotations to PaddleOCR detection format...")

    train_label = os.path.join(OUTPUT_LABELS_DIR, "train_label.txt")
    val_label   = os.path.join(OUTPUT_LABELS_DIR, "val_label.txt")

    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

    train_count = convert_icdar_to_paddle_det(
        TRAIN_ANNOTATIONS, TRAIN_IMAGES_DIR, train_label, "train"
    )
    val_count = convert_icdar_to_paddle_det(
        VAL_ANNOTATIONS, VAL_IMAGES_DIR, val_label, "val"
    )

    # Step 2: Generate config
    print("\n📝 Generating detection training config...")
    config_path = "training_setup/configs/det_finetune.yml"

    config = DET_CONFIG_TEMPLATE.format(
        epochs          = EPOCHS,
        output_dir      = OUTPUT_MODEL_DIR,
        pretrained_model = PRETRAINED_MODEL,
        lr              = LR,
        batch_size      = BATCH_SIZE,
        train_label     = train_label,
        val_label       = val_label,
    )

    with open(config_path, 'w') as f:
        f.write(config.strip())
    print(f"  ✅ Config written to: {config_path}")

    # Step 3: Check pretrained model
    if not os.path.exists(PRETRAINED_MODEL + ".pdparams"):
        print(f"\n  ⚠️  Pretrained detection model not found.")
        print(f"  Download it:")
        print(f"    mkdir pretrained")
        print(f"    cd pretrained")
        print(f"    curl -O https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_det_train.tar")
        print(f"    tar -xf en_PP-OCRv4_det_train.tar")

    # Step 4: Launch training
    train_script = "PaddleOCR_Official_Tools/tools/train.py"
    cmd = [sys.executable, train_script, "-c", config_path]

    print(f"\n🚀 To launch detection fine-tuning, run:")
    print(f"   {' '.join(cmd)}")
    print(f"\n   Or run this script again after downloading the pretrained model.")
    print(f"\n   Output: {OUTPUT_MODEL_DIR}/best_accuracy")
    print(f"\n▶️  Next step: Run step4_evaluate_and_infer.py")

    # Auto-launch if pretrained model exists
    if os.path.exists(PRETRAINED_MODEL + ".pdparams"):
        print(f"\n🚀 Launching detection fine-tuning...")
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
