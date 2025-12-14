import os
import random
import yaml

# --- CONFIGURATION ---
BASE_DIR = os.getcwd().replace('\\', '/') # Ensure forward slashes for Paddle
DATA_DIR = f"{BASE_DIR}/PaddleOCR/train_data"
# Input Label File (from augmentation step)
INPUT_LABEL_FILE = f"{DATA_DIR}/rec_gt_augmented.txt"

# Output Paths
TRAIN_LABEL_FILE = f"{DATA_DIR}/rec_gt_train.txt"
VAL_LABEL_FILE = f"{DATA_DIR}/rec_gt_val.txt"
CONFIG_FILE = "my_config.yml"

# Pretrained Model Path
PRETRAINED_MODEL = f"{BASE_DIR}/PaddleOCR/pretrain_models/en_PP-OCRv4_rec_train/student"

def prepare_training():
    # --- 1. SPLIT DATA (90% Train, 10% Val) ---
    if not os.path.exists(INPUT_LABEL_FILE):
        print(f"❌ Error: Could not find {INPUT_LABEL_FILE}")
        return

    print(f"📖 Reading {INPUT_LABEL_FILE}...")
    with open(INPUT_LABEL_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Shuffle to ensure randomness
    random.shuffle(lines)
    
    split_idx = int(len(lines) * 0.9)
    train_data = lines[:split_idx]
    val_data = lines[split_idx:]
    
    with open(TRAIN_LABEL_FILE, 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    with open(VAL_LABEL_FILE, 'w', encoding='utf-8') as f:
        f.writelines(val_data)
        
    print(f"✅ Data Split Complete:")
    print(f"   - Train: {len(train_data)} samples")
    print(f"   - Val:   {len(val_data)} samples")

    # --- 2. GENERATE YAML CONFIG ---
    # This is the exact configuration for Fine-Tuning PP-OCRv4
    config = f"""
Global:
  use_gpu: true
  epoch_num: 50
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: {BASE_DIR}/PaddleOCR/output/rec_finetune
  save_epoch_step: 10
  eval_batch_step: [0, 500]
  cal_metric_during_train: true
  pretrained_model: {PRETRAINED_MODEL}
  checkpoints: null
  save_inference_dir: null
  use_visualdl: false
  infer_img: null
  character_dict_path: {BASE_DIR}/PaddleOCR_Engine/ppocr/utils/en_dict.txt
  max_text_length: 25
  infer_mode: false
  use_space_char: true
  distributed: false 
  save_res_path: {BASE_DIR}/PaddleOCR/output/rec/predicts_ppocrv4.txt

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.0001
    warmup_epoch: 5
  regularizer:
    name: 'L2'
    factor: 0.00001

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: PPLCNetV3
    scale: 0.95
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr
            dims: 120
            depth: 2
            hidden_dims: 120
            use_guide: True
          Head:
            fc_decay: 0.00001
      - NRTRHead:
          nrtr_dim: 384
          max_text_length: 25

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - NRTRLoss:

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: {BASE_DIR}/PaddleOCR/train_data
    ext_op_transform_idx: 1
    label_file_list:
    - {TRAIN_LABEL_FILE}
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - RecAug:
    - MultiLabelEncode:
    - RecResizeImg:
        image_shape: [3, 48, 320]
    - KeepKeys:
        keep_keys:
        - image
        - label_ctc
        - label_nrtr
        - length
        - valid_ratio
  loader:
    shuffle: true
    batch_size_per_card: 64
    drop_last: true
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {BASE_DIR}/PaddleOCR/train_data
    label_file_list:
    - {VAL_LABEL_FILE}
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - MultiLabelEncode:
    - RecResizeImg:
        image_shape: [3, 48, 320]
    - KeepKeys:
        keep_keys:
        - image
        - label_ctc
        - label_nrtr
        - length
        - valid_ratio
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 64
    num_workers: 4
"""
    
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        f.write(config)
        
    print(f"✅ Config File Generated: {CONFIG_FILE}")
    print("-" * 30)
    print("🚀 READY TO TRAIN!")

if __name__ == "__main__":
    prepare_training()