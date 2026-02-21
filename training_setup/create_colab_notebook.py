"""
Run this script ONCE on your laptop to generate the Colab notebook:
    python create_colab_notebook.py

Then upload the generated  colab_detection_finetune.ipynb  to Google Colab.
"""

import json, os

REPO_URL = "https://github.com/KailasVS666/Map_Text_Detection_Recognition.git"

# ── Notebook cells ─────────────────────────────────────────────────────────

def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": [text.strip() + "\n"]}

def code(lines):
    src = [l + "\n" for l in lines]
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

cells = [

md("# 🗺️ Rumsey Map OCR — Detection Model Fine-tuning\n"
   "Run each cell in order. Training takes **4-6 hours** on a T4 GPU.\n\n"
   "> **Before running:** Go to `Runtime → Change runtime type → T4 GPU`"),

md("## Step 1 — Check GPU"),
code([
    "import subprocess",
    "result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)",
    "print(result.stdout or 'No GPU found — make sure you selected T4 GPU!')",
]),

md("## Step 2 — Install PaddlePaddle GPU"),
code([
    "# Install PaddlePaddle with CUDA 11.8 (matches Colab's default CUDA)",
    "!pip install paddlepaddle-gpu==2.6.1.post120 \\",
    "    -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html -q",
    "import paddle",
    "print(f'PaddlePaddle version: {paddle.__version__}')",
    "print(f'GPU available: {paddle.is_compiled_with_cuda()}')",
]),

md("## Step 3 — Mount Google Drive\n"
   "Your training data must be uploaded to Drive before this step.\n\n"
   "**Required folder structure on Drive:**\n"
   "```\n"
   "My Drive/\n"
   "└── Rumsey_OCR/\n"
   "    ├── train_data/          ← upload this from your laptop\n"
   "    │   ├── rec/             ← (already done, from recognition training)\n"
   "    │   └── det/             ← will be created by prepare_det_data.py\n"
   "    └── Rumsey_Map_OCR_Data/ ← upload this from your laptop\n"
   "```"),
code([
    "from google.colab import drive",
    "drive.mount('/content/drive')",
    "",
    "import os",
    "DRIVE_ROOT = '/content/drive/MyDrive/Rumsey_OCR'",
    "print('Drive contents:', os.listdir(DRIVE_ROOT) if os.path.exists(DRIVE_ROOT) else 'FOLDER NOT FOUND!')",
]),

md("## Step 4 — Clone Repository & Setup Paths"),
code([
    f"!git clone {REPO_URL} /content/rumsey_ocr 2>&1 | tail -5",
    "%cd /content/rumsey_ocr",
    "",
    "import os, sys",
    "DRIVE_ROOT = '/content/drive/MyDrive/Rumsey_OCR'",
    "",
    "# Link Drive folders into the repo directory",
    "for folder in ['train_data', 'Rumsey_Map_OCR_Data']:",
    "    src = os.path.join(DRIVE_ROOT, folder)",
    "    dst = os.path.join('/content/rumsey_ocr', folder)",
    "    if os.path.exists(src) and not os.path.exists(dst):",
    "        os.symlink(src, dst)",
    "        print(f'Linked {folder}')",
    "    elif not os.path.exists(src):",
    "        print(f'WARNING: {src} not found on Drive!')",
    "    else:",
    "        print(f'{folder} already linked')",
]),

md("## Step 5 — Install Python Requirements"),
code([
    "%cd /content/rumsey_ocr",
    "!pip install -r PaddleOCR_Official_Tools/requirements.txt -q",
    "!pip install shapely scikit-image lmdb imgaug -q",
    "print('Requirements installed!')",
]),

md("## Step 6 — Download Pretrained Detection Model"),
code([
    "%cd /content/rumsey_ocr",
    "!mkdir -p pretrained",
    "",
    "# Download PP-OCRv4 English detection pretrained weights",
    "DET_URL = 'https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_det_train.tar'",
    "!wget -q --show-progress -P pretrained/ {DET_URL}",
    "!cd pretrained && tar -xf en_PP-OCRv4_det_train.tar",
    "!ls pretrained/en_PP-OCRv4_det_train/",
]),

md("## Step 7 — Prepare Detection Training Data\n"
   "Converts ICDAR 2024 JSON annotations → PaddleOCR detection format.\n"
   "Skip this step if `train_data/det/` already exists on Drive."),
code([
    "%cd /content/rumsey_ocr",
    "",
    "DET_DATA = 'train_data/det'",
    "if os.path.exists(os.path.join(DET_DATA, 'train_list.txt')):",
    "    print('Detection data already prepared, skipping.')",
    "else:",
    "    print('Preparing detection data...')",
    "    !python training_setup/prepare_det_data.py",
    "    # Copy prepared data to Drive for future use",
    "    !cp -r train_data/det {DRIVE_ROOT}/train_data/det",
    "    print('Copied to Drive!')",
]),

md("## Step 8 — Verify Data"),
code([
    "# Check that label files exist and are populated",
    "for fname in ['train_data/det/train_list.txt', 'train_data/det/val_list.txt']:",
    "    if os.path.exists(fname):",
    "        with open(fname) as f:",
    "            lines = f.readlines()",
    "        print(f'{fname}: {len(lines):,} images')",
    "    else:",
    "        print(f'MISSING: {fname}')",
]),

md("## Step 9 — Train Detection Model 🔥\n"
   "This is the main training cell. It takes **4-6 hours**.\n"
   "> Colab will stay alive as long as this cell is running."),
code([
    "%cd /content/rumsey_ocr/PaddleOCR_Official_Tools",
    "",
    "!python tools/train.py \\",
    "    -c ../training_setup/configs/det_icdar_finetune.yml",
]),

md("## Step 10 — Export Trained Model"),
code([
    "%cd /content/rumsey_ocr/PaddleOCR_Official_Tools",
    "",
    "!python tools/export_model.py \\",
    "    -c ../training_setup/configs/det_icdar_finetune.yml \\",
    "    -o Global.pretrained_model=../output/det_finetune/best_accuracy \\",
    "       Global.save_inference_dir=../output/det_inference_finetuned",
    "",
    "print('Export complete!')",
]),

md("## Step 11 — Save Results to Google Drive ✅"),
code([
    "import shutil",
    "DRIVE_OUTPUT = os.path.join(DRIVE_ROOT, 'output')",
    "os.makedirs(DRIVE_OUTPUT, exist_ok=True)",
    "",
    "# Save training checkpoints",
    "if os.path.exists('../output/det_finetune'):",
    "    shutil.copytree('../output/det_finetune',",
    "                    os.path.join(DRIVE_OUTPUT, 'det_finetune'),",
    "                    dirs_exist_ok=True)",
    "    print('Checkpoints saved to Drive')",
    "",
    "# Save inference model",
    "if os.path.exists('../output/det_inference_finetuned'):",
    "    shutil.copytree('../output/det_inference_finetuned',",
    "                    os.path.join(DRIVE_OUTPUT, 'det_inference_finetuned'),",
    "                    dirs_exist_ok=True)",
    "    print('Inference model saved to Drive')",
    "",
    "print(f'All outputs saved to: {DRIVE_OUTPUT}')",
]),

md("## ✅ Done!\n"
   "After this notebook finishes:\n"
   "1. Download `output/det_inference_finetuned/` from your Drive\n"
   "2. Place it in `Rumsey_Map_OCR/output/det_inference_finetuned/` on your laptop\n"
   "3. Run `step4_evaluate_and_infer.py` again to see the improved results\n\n"
   "Expected improvement: **27% → 50-60%+ recall** 🎯"),

]

# ── Write notebook ──────────────────────────────────────────────────────────

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(__file__), "..", "colab_detection_finetune.ipynb")
out_path = os.path.normpath(out_path)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook created: {out_path}")
print("   Next: Upload this file to Google Colab (File → Upload notebook)")
