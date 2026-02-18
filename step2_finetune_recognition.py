"""
STEP 2: Fine-tune PaddleOCR Recognition Model on ICDAR Data
=============================================================
This script sets up and launches fine-tuning of the PaddleOCR recognition
model (PP-OCRv4) using the word crops extracted in Step 1.

The recognition model learns the specific fonts, styles, and character
patterns found in historical maps (cursive, serif, small caps, etc.)

Prerequisites:
  - Run step1_extract_icdar_crops.py first
  - PaddlePaddle installed (GPU recommended)
  - ~34,000 training crops in train_data/rec/train/

Usage:
  python step2_finetune_recognition.py
  
  This will generate the config and launch training via:
  python PaddleOCR_Official_Tools/tools/train.py -c training_setup/configs/rec_finetune.yml
"""

import os
import subprocess
import sys

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# Pretrained model to fine-tune from (PP-OCRv4 English recognition)
# Download from: https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar
PRETRAINED_MODEL = "./pretrained/en_PP-OCRv4_rec_train/best_accuracy"

TRAIN_DATA_DIR   = "./train_data/rec"
OUTPUT_MODEL_DIR = "./output/rec_finetune"
CHAR_DICT_PATH   = "./PaddleOCR_Official_Tools/ppocr/utils/en_dict.txt"

# Training hyperparameters
EPOCHS           = 100
BATCH_SIZE       = 128   # Reduce to 64 if GPU OOM
LEARNING_RATE    = 0.0001  # Low LR for fine-tuning (don't destroy pretrained weights)
IMAGE_SHAPE      = [3, 48, 320]  # [C, H, W]
# ──────────────────────────────────────────────────────────────────────────────


CONFIG_TEMPLATE = """
Global:
  use_gpu: true
  epoch_num: {epochs}
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: {output_dir}
  save_epoch_step: 5
  eval_batch_step: [0, 500]
  cal_metric_during_train: true
  pretrained_model: {pretrained_model}
  checkpoints:
  use_visualdl: false
  infer_img:
  character_dict_path: {char_dict}
  max_text_length: 40
  infer_mode: false
  use_space_char: true
  distributed: false
  save_res_path: {output_dir}/predicts.txt

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: {lr}
    warmup_epoch: 5
  regularizer:
    name: 'L2'
    factor: 3.0e-05

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
    model_name: large
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr
            dims: 64
            depth: 2
            hidden_dims: 120
            use_guide: True
          Head:
            fc_decay: 0.00001
      - SARHead:
          enc_dim: 512
          max_text_length: 40

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - SARLoss:

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: {train_data_dir}
    label_file_list:
      - {train_data_dir}/train_list.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - RecAug:
      - MultiLabelEncode:
      - RecResizeImg:
          image_shape: {image_shape}
      - KeepKeys:
          keep_keys: ['image', 'label_ctc', 'label_sar', 'length', 'valid_ratio']
  loader:
    shuffle: true
    batch_size_per_card: {batch_size}
    drop_last: true
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {train_data_dir}
    label_file_list:
      - {train_data_dir}/val_list.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - MultiLabelEncode:
      - RecResizeImg:
          image_shape: {image_shape}
      - KeepKeys:
          keep_keys: ['image', 'label_ctc', 'label_sar', 'length', 'valid_ratio']
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: {batch_size}
    num_workers: 4
"""


def check_prerequisites():
    """Check that all required files exist before starting."""
    print("🔍 Checking prerequisites...")
    ok = True

    train_list = os.path.join(TRAIN_DATA_DIR, "train_list.txt")
    val_list   = os.path.join(TRAIN_DATA_DIR, "val_list.txt")

    if not os.path.exists(train_list):
        print(f"  ❌ Missing: {train_list}")
        print(f"     → Run step1_extract_icdar_crops.py first!")
        ok = False
    else:
        with open(train_list) as f:
            count = sum(1 for _ in f)
        print(f"  ✅ Train list: {count:,} samples")

    if not os.path.exists(val_list):
        print(f"  ❌ Missing: {val_list}")
        ok = False
    else:
        with open(val_list) as f:
            count = sum(1 for _ in f)
        print(f"  ✅ Val list:   {count:,} samples")

    if not os.path.exists(CHAR_DICT_PATH):
        print(f"  ❌ Missing char dict: {CHAR_DICT_PATH}")
        ok = False
    else:
        print(f"  ✅ Char dict found")

    if not os.path.exists(PRETRAINED_MODEL + ".pdparams"):
        print(f"\n  ⚠️  Pretrained model not found at: {PRETRAINED_MODEL}")
        print(f"  Download it with:")
        print(f"    mkdir pretrained")
        print(f"    cd pretrained")
        print(f"    curl -O https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar")
        print(f"    tar -xf en_PP-OCRv4_rec_train.tar")
        print(f"\n  Training will start from scratch without pretrained weights (slower).")
        ok = False

    return ok


def generate_config():
    """Generate the training config YAML file."""
    config_path = "training_setup/configs/rec_finetune.yml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    config = CONFIG_TEMPLATE.format(
        epochs         = EPOCHS,
        output_dir     = OUTPUT_MODEL_DIR,
        pretrained_model = PRETRAINED_MODEL,
        char_dict      = CHAR_DICT_PATH,
        lr             = LEARNING_RATE,
        train_data_dir = TRAIN_DATA_DIR,
        image_shape    = str(IMAGE_SHAPE),
        batch_size     = BATCH_SIZE,
    )

    with open(config_path, 'w') as f:
        f.write(config.strip())

    print(f"  ✅ Config written to: {config_path}")
    return config_path


def main():
    print("🗺️  PaddleOCR Recognition Fine-tuning")
    print("=" * 60)

    prereqs_ok = check_prerequisites()

    print("\n📝 Generating training config...")
    config_path = generate_config()

    train_script = "PaddleOCR_Official_Tools/tools/train.py"
    if not os.path.exists(train_script):
        print(f"\n❌ Training script not found: {train_script}")
        return

    cmd = [sys.executable, train_script, "-c", config_path]

    print(f"\n🚀 Launching fine-tuning...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Epochs:  {EPOCHS}")
    print(f"   Batch:   {BATCH_SIZE}")
    print(f"   LR:      {LEARNING_RATE}")
    print(f"\n   Output will be saved to: {OUTPUT_MODEL_DIR}/")
    print(f"   Best model: {OUTPUT_MODEL_DIR}/best_accuracy\n")
    print("=" * 60)

    if not prereqs_ok:
        print("\n⚠️  Some prerequisites are missing (see above).")
        print("   Fix them, then re-run this script.")
        print("\n   To run training manually:")
        print(f"   python {train_script} -c {config_path}")
        return

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
