import os
import subprocess
import sys

# --- CONFIGURATION ---
TRAINED_MODEL_PATH = os.path.join("PaddleOCR", "output", "rec_finetune", "best_accuracy")
OUTPUT_DIR = os.path.join("output", "rec_inference")
TOOLS_DIR = "PaddleOCR_Official_Tools"

# --- FIX: Calculate Absolute Path to Dictionary ---
# We find the full path to the dictionary file inside the downloaded tools
abs_tools_dir = os.path.abspath(TOOLS_DIR)
dictionary_path = os.path.join(abs_tools_dir, "ppocr", "utils", "en_dict.txt")

# Convert Windows backslashes to forward slashes for YAML compatibility
dictionary_path = dictionary_path.replace("\\", "/")

if not os.path.exists(dictionary_path):
    print(f"❌ Error: Dictionary file not found at: {dictionary_path}")
    print("   Please ensure the 'PaddleOCR_Official_Tools' folder is not empty.")
    sys.exit(1)

print(f"✅ Found dictionary at: {dictionary_path}")

# --- STEP 1: CREATE THE CONFIG FILE MANUALLY ---
# We inject the correct dictionary_path into the config
config_content = f"""
Global:
  use_gpu: false
  epoch_num: 500
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/rec_ppocr_v3_distillation
  save_epoch_step: 3
  eval_batch_step: [0, 2000]
  cal_metric_during_train: true
  pretrained_model:
  checkpoints:
  save_inference_dir:
  use_visualdl: false
  infer_img: doc/imgs_words/ch/word_1.jpg
  character_dict_path: {dictionary_path}
  max_text_length: 25
  infer_mode: false
  use_space_char: true
  distributed: true
  save_res_path: ./output/rec/predicts_ppocrv3_distillation.txt

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.0005
    warmup_epoch: 5
  regularizer:
    name: 'L2'
    factor: 3.0e-05

Architecture:
  model_type: rec
  algorithm: SVTR
  Transform:
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
    model_name: large
    visible_devices: 0,1,2,3
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
          max_text_length: 25

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
    data_dir: ./train_data/
    label_file_list:
      - ./train_data/train_list.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - RecAug:
      - MultiLabelEncode:
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label_ctc', 'label_sar', 'length', 'valid_ratio']
  loader:
    shuffle: true
    batch_size_per_card: 128
    drop_last: true
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/
    label_file_list:
      - ./train_data/val_list.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - MultiLabelEncode:
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label_ctc', 'label_sar', 'length', 'valid_ratio']
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 128
    num_workers: 4
"""

# CORRECT: Point to your actual training config
config_path = "my_config.yml"
with open(config_path, "w") as f:
    f.write(config_content)
print(f"✅ Created temporary config file: {config_path}")

# --- STEP 2: LOCATE THE CORRECT EXPORT SCRIPT ---
export_script = os.path.join(TOOLS_DIR, "tools", "export_model.py")

if not os.path.exists(export_script):
    print(f"❌ Error: Expected to find export script at: {export_script}")
    sys.exit(1)

# --- STEP 3: RUN EXPORT ---
env = os.environ.copy()
env["PYTHONPATH"] = TOOLS_DIR + os.pathsep + env.get("PYTHONPATH", "")

cmd = [
    sys.executable, export_script,
    "-c", config_path,
    "-o", f"Global.pretrained_model={TRAINED_MODEL_PATH}",
    f"Global.save_inference_dir={OUTPUT_DIR}"
]

print("-" * 50)
print("📦 Exporting Model...")
print("-" * 50)

try:
    subprocess.check_call(cmd, env=env)
    print("\n✅ SUCCESS! Model exported.")
    print(f"📂 New inference model is at: {os.path.abspath(OUTPUT_DIR)}")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Export failed with code {e.returncode}")