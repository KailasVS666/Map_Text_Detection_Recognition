"""
Run this script ONCE on your laptop to generate the Colab notebook:
    python training_setup/create_colab_notebook.py

Then upload:
  1. colab_detection_finetune.ipynb  → open directly in Colab
  2. train_data/det/                 → My Drive/Rumsey_OCR/train_data/det/
  3. training_setup/configs/det_icdar_finetune.yml → My Drive/Rumsey_OCR/det_icdar_finetune.yml
"""

import json, os

DRIVE_ROOT = "/content/drive/MyDrive/Rumsey_OCR"

# ── Helpers ────────────────────────────────────────────────────────────────

def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": [text.strip() + "\n"]}

def code(lines):
    src = [l + "\n" for l in lines]
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

# ── Cells ──────────────────────────────────────────────────────────────────

cells = [

md("""# 🗺️ Rumsey Map OCR — Detection Model Fine-tuning
Run each cell **top to bottom**. Training takes **4-6 hours** on a T4 GPU.

> **Before running:** `Runtime → Change runtime type → T4 GPU`

### What you need on Google Drive first:
```
My Drive/
└── Rumsey_OCR/
    ├── train_data/
    │   └── det/            ← upload this from your laptop
    │       ├── train_list.txt
    │       ├── val_list.txt
    │       └── train_images/
    └── det_icdar_finetune.yml   ← upload this config file
```
"""),

# ── Step 1: GPU check ──────────────────────────────────────────────────────
md("## Step 1 — Verify GPU"),
code([
    "import subprocess",
    "r = subprocess.run(['nvidia-smi'], capture_output=True, text=True)",
    "print(r.stdout or '❌ No GPU — go to Runtime → Change runtime type → T4 GPU')",
]),

# ── Step 2: Mount Drive ────────────────────────────────────────────────────
md("## Step 2 — Mount Google Drive"),
code([
    "from google.colab import drive",
    "drive.mount('/content/drive')",
    "",
    "import os",
    f"DRIVE_ROOT = '{DRIVE_ROOT}'",
    "",
    "# Verify required files are on Drive",
    "checks = {",
    f"    'Training data':  os.path.join(DRIVE_ROOT, 'train_data/det/train_list.txt'),",
    f"    'Val data':       os.path.join(DRIVE_ROOT, 'train_data/det/val_list.txt'),",
    f"    'Config file':    os.path.join(DRIVE_ROOT, 'det_icdar_finetune.yml'),",
    "}",
    "all_ok = True",
    "for name, path in checks.items():",
    "    exists = os.path.exists(path)",
    "    status = '✅' if exists else '❌  MISSING'",
    "    print(f'  {status}  {name}: {path}')",
    "    if not exists: all_ok = False",
    "if not all_ok:",
    "    print('\\n⚠️  Upload missing files to Drive before continuing!')",
    "else:",
    "    print('\\n✅ All required files found!')",
]),

# ── Step 3: Install PaddlePaddle ───────────────────────────────────────────
md("## Step 3 — Install PaddlePaddle GPU\n*(Takes ~2 min — run once per session)*"),
code([
    "!pip install paddlepaddle-gpu==2.6.1.post120 \\",
    "    -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html -q",
    "",
    "import paddle",
    "print(f'PaddlePaddle: {paddle.__version__}')",
    "print(f'GPU available: {paddle.is_compiled_with_cuda()}')",
]),

# ── Step 4: Clone official PaddleOCR toolkit ───────────────────────────────
md("## Step 4 — Get PaddleOCR Toolkit\n"
   "Clones the official PaddleOCR training tools. "
   "*(No need to upload — pulled from PaddlePaddle directly)*"),
code([
    "import os",
    "",
    "PADDLEOCR_DIR = '/content/PaddleOCR'",
    "",
    "if not os.path.exists(PADDLEOCR_DIR):",
    "    !git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git {PADDLEOCR_DIR} -q",
    "    print('Cloned PaddleOCR')",
    "else:",
    "    print('PaddleOCR already present')",
    "",
    "# Install requirements",
    "!pip install -r {PADDLEOCR_DIR}/requirements.txt -q",
    "print('Requirements installed!')",
]),

# ── Step 5: Setup workspace ────────────────────────────────────────────────
md("## Step 5 — Setup Workspace"),
code([
    "import os, shutil",
    "",
    f"DRIVE_ROOT   = '{DRIVE_ROOT}'",
    "PADDLEOCR_DIR = '/content/PaddleOCR'",
    "WORK_DIR      = '/content/workspace'",
    "os.makedirs(WORK_DIR, exist_ok=True)",
    "",
    "# Link training data from Drive into workspace",
    "DET_DATA_SRC = os.path.join(DRIVE_ROOT, 'train_data/det')",
    "DET_DATA_DST = os.path.join(WORK_DIR, 'train_data/det')",
    "os.makedirs(os.path.dirname(DET_DATA_DST), exist_ok=True)",
    "if not os.path.exists(DET_DATA_DST):",
    "    os.symlink(DET_DATA_SRC, DET_DATA_DST)",
    "    print(f'Linked train_data/det from Drive')",
    "",
    "# Copy config into workspace (and fix paths to be absolute)",
    "CONFIG_SRC = os.path.join(DRIVE_ROOT, 'det_icdar_finetune.yml')",
    "CONFIG_DST = os.path.join(WORK_DIR, 'det_icdar_finetune.yml')",
    "shutil.copy2(CONFIG_SRC, CONFIG_DST)",
    "",
    "# Link output directory to Drive for REAL-TIME checkpoint saving",
    "# This ensures that if Colab disconnects, your progress is saved on Drive!",
    f"DRIVE_OUT_DIR = os.path.join(DRIVE_ROOT, 'output/det_finetune')",
    f"LOCAL_OUT_DIR = os.path.join(WORK_DIR, 'output/det_finetune')",
    "os.makedirs(DRIVE_OUT_DIR, exist_ok=True)",
    "os.makedirs(os.path.dirname(LOCAL_OUT_DIR), exist_ok=True)",
    "",
    "if not os.path.exists(LOCAL_OUT_DIR):",
    "    os.symlink(DRIVE_OUT_DIR, LOCAL_OUT_DIR)",
    "    print(f'✅ Output directory symlinked to Drive: {DRIVE_OUT_DIR}')",
    "",
    "# Patch paths in config to be absolute",
    "with open(CONFIG_DST) as f:",
    "    cfg = f.read()",
    "cfg = cfg.replace('../train_data/det', DET_DATA_DST)",
    "cfg = cfg.replace('../output/det_finetune', LOCAL_OUT_DIR)",
    "with open(CONFIG_DST, 'w') as f:",
    "    f.write(cfg)",
    "print('Config ready at:', CONFIG_DST)",
    "print('Workspace ready!')",
]),

# ── Step 6: Download pretrained weights ────────────────────────────────────
# Step 6 — Download Pretrained Detection Model
md("## Step 6 — Download Pretrained Detection Model"),
code([
    "import os",
    "WORK_DIR    = '/content/workspace'",
    "PRETRAINED  = os.path.join(WORK_DIR, 'pretrained/ch_PP-OCRv4_det_train')",
    "TAR_FILE    = os.path.join(WORK_DIR, 'pretrained/ch_PP-OCRv4_det_train.tar')",
    "",
    "if not os.path.exists(PRETRAINED):",
    "    os.makedirs(os.path.join(WORK_DIR, 'pretrained'), exist_ok=True)",
    "    print('Downloading pretrained model...')",
    "    URL = 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar'",
    "    # Remove existing partial downloads",
    "    if os.path.exists(TAR_FILE): os.remove(TAR_FILE)",
    "    ",
    "    # Use curl (more reliable than wget in Colab)",
    "    import subprocess",
    "    subprocess.run(['curl', '-L', '--progress-bar', '-o', TAR_FILE, URL], check=False)",
    "    ",
    "    size = os.path.getsize(TAR_FILE) if os.path.exists(TAR_FILE) else 0",
    "    print(f'Downloaded: {size/1024/1024:.1f} MB')",
    "    if size > 10_000_000:",
    "        print('Verifying archive...')",
    "        try:",
    "            # Check if it's a valid tar file",
    "            is_valid = subprocess.run(['tar', '-tf', TAR_FILE], capture_output=True).returncode == 0",
    "            if is_valid:",
    "                print('Extracting...')",
    "                !tar -xf {TAR_FILE} -C {os.path.join(WORK_DIR, 'pretrained')}",
    "                print('✅ Extracted successfully!')",
    "            else:",
    "                print('❌ Downloaded file is not a valid archive. Retrying download might be needed.')",
    "        except:",
    "            !tar -xf {TAR_FILE} -C {os.path.join(WORK_DIR, 'pretrained')}",
    "    else:",
    "        print('❌ Download failed or file too small.')",
    "else:",
    "    print('✅ Pretrained model already present')",
    "",
    "# Patch config with pretrained model path",
    "CONFIG_DST = os.path.join(WORK_DIR, 'det_icdar_finetune.yml')",
    "if os.path.exists(CONFIG_DST):",
    "    with open(CONFIG_DST) as f:",
    "        cfg = f.read()",
    "    cfg = cfg.replace('../pretrained/ch_PP-OCRv4_det_train/best_accuracy',",
    "                      os.path.join(PRETRAINED, 'best_accuracy'))",
    "    with open(CONFIG_DST, 'w') as f:",
    "        f.write(cfg)",
    "    print('Config updated with pretrained model path')",
    "else:",
    "    print('⚠️  Config file not found in workspace yet!')",
]),

# ── Step 7: Quick data check ───────────────────────────────────────────────
md("## Step 7 — Verify Training Data"),
code([
    "WORK_DIR = '/content/workspace'",
    "import os",
    "",
    "for fname in ['train_list.txt', 'val_list.txt']:",
    "    path = os.path.join(WORK_DIR, 'train_data/det', fname)",
    "    if os.path.exists(path):",
    "        with open(path) as f: n = sum(1 for _ in f)",
    "        print(f'✅ {fname}: {n:,} images')",
    "    else:",
    "        print(f'❌ MISSING: {path}')",
]),

# ── Step 8: TRAIN ──────────────────────────────────────────────────────────
md("## Step 8 — Train Detection Model 🔥\n"
   "This is the main cell. Takes **4-6 hours**.\n"
   "> The session stays alive as long as this cell is running.\n\n"
   "> **Tip:** Keep your browser tab open (or use Colab Pro for longer sessions)."),
code([
    "WORK_DIR      = '/content/workspace'",
    "PADDLEOCR_DIR = '/content/PaddleOCR'",
    "CONFIG        = os.path.join(WORK_DIR, 'det_icdar_finetune.yml')",
    "CKPT_DIR      = os.path.join(WORK_DIR, 'output/det_finetune')",
    "",
    "import os, yaml",
    "os.chdir(PADDLEOCR_DIR)",
    "",
    "# ── 1. Patch db_postprocess.py to accept 4 or 6 element shapes ──",
    "db_path = os.path.join(PADDLEOCR_DIR, 'ppocr/postprocess/db_postprocess.py')",
    "with open(db_path, 'r') as f: db_code = f.read()",
    "_old = 'src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]'",
    "_new = 'src_h, src_w, ratio_h, ratio_w = shape_list[batch_index][:4]'",
    "if _old in db_code:",
    "    db_code = db_code.replace(_old, _new)",
    "    with open(db_path, 'w') as f: f.write(db_code)",
    "    print('✓ Patched db_postprocess.py (shape[:4])')",
    "else:",
    "    print('✓ db_postprocess.py already patched or not needed')",
    "",
    "# ── 2. Patch eval_det_iou.py for Shapely 2.x (Python 3.12) ──",
    "# Always restore from git first so patch runs on a clean file every time",
    "import subprocess as _sp",
    "_sp.run(['git', 'checkout', 'HEAD', '--', 'ppocr/metrics/eval_det_iou.py'],",
    "        cwd=PADDLEOCR_DIR, capture_output=True)",
    "iou_path = os.path.join(PADDLEOCR_DIR, 'ppocr/metrics/eval_det_iou.py')",
    "with open(iou_path, 'r') as f: _lines = f.readlines()",
    "_out = []; _fixed = 0",
    "for _l in _lines:",
    "    if 'Polygon(points).is_valid' in _l:",
    "        _s = ' ' * (len(_l) - len(_l.lstrip()))",
    "        _out.append(f'{_s}if hasattr(points, \"reshape\") and points.ndim == 1:\\n')",
    "        _out.append(f'{_s}    points = points.reshape(-1, 2)\\n')",
    "        _fixed += 1",
    "    _out.append(_l)",
    "with open(iou_path, 'w') as f: f.writelines(_out)",
    "print(f'✓ eval_det_iou.py patched ({_fixed} sites)')",
    "",
    "# ── 4. Ensure config has correct eval settings ──",
    "with open(CONFIG, 'r') as f: cfg = yaml.safe_load(f)",
    "cfg['Global']['eval_batch_step'] = [0, 50]",
    "cfg['Global']['cal_metric_during_train'] = True",
    "cfg['Global']['checkpoints'] = None",
    "# Reset DetResizeForTest to default (limit_type:max) — [:4] fix handles 6-element shapes",
    "for t in cfg['Eval']['dataset']['transforms']:",
    "    if 'DetResizeForTest' in t:",
    "        t['DetResizeForTest'] = None",
    "        break",
    "with open(CONFIG, 'w') as f: yaml.safe_dump(cfg, f, sort_keys=False)",
    "print('✓ Config verified (eval_batch_step=50, DetResizeForTest=default)')",
    "",
    "# ── 5. Auto-resume or fresh start ──",
    "latest_ckpt = os.path.join(CKPT_DIR, 'latest')",
    "if os.path.exists(latest_ckpt + '.pdparams'):",
    "    print(f'▶️  Resuming from: {latest_ckpt}')",
    "    !python tools/train.py -c {CONFIG} -o Global.checkpoints={latest_ckpt}",
    "else:",
    "    print('🆕 Starting fresh from pretrained model')",
    "    !python tools/train.py -c {CONFIG}",
]),

# ── Step 9: Export ────────────────────────────────────────────────────────
md("## Step 9 — Export Trained Model"),
code([
    "WORK_DIR      = '/content/workspace'",
    "PADDLEOCR_DIR = '/content/PaddleOCR'",
    "CONFIG        = os.path.join(WORK_DIR, 'det_icdar_finetune.yml')",
    "BEST_MODEL    = os.path.join(WORK_DIR, 'output/det_finetune/best_accuracy')",
    "INFER_OUT     = os.path.join(WORK_DIR, 'output/det_inference_finetuned')",
    "",
    "import os",
    "os.chdir(PADDLEOCR_DIR)",
    "",
    "!python tools/export_model.py \\",
    "    -c {CONFIG} \\",
    "    -o Global.pretrained_model={BEST_MODEL} \\",
    "       Global.save_inference_dir={INFER_OUT}",
    "",
    "print('Export complete!')",
]),

# ── Step 10: Save to Drive ────────────────────────────────────────────────
md("## Step 10 — Save Everything to Drive ✅\n"
   "Saves checkpoints and the final inference model back to your Google Drive."),
code([
    "import shutil, os",
    f"DRIVE_ROOT = '{DRIVE_ROOT}'",
    "WORK_DIR   = '/content/workspace'",
    "",
    "DRIVE_OUT = os.path.join(DRIVE_ROOT, 'output')",
    "os.makedirs(DRIVE_OUT, exist_ok=True)",
    "",
    "# Save training checkpoints",
    "src_ckpt = os.path.join(WORK_DIR, 'output/det_finetune')",
    "dst_ckpt = os.path.join(DRIVE_OUT, 'det_finetune')",
    "if os.path.exists(src_ckpt):",
    "    shutil.copytree(src_ckpt, dst_ckpt, dirs_exist_ok=True)",
    "    print(f'✅ Checkpoints saved to Drive')",
    "",
    "# Save inference model",
    "src_inf = os.path.join(WORK_DIR, 'output/det_inference_finetuned')",
    "dst_inf = os.path.join(DRIVE_OUT, 'det_inference_finetuned')",
    "if os.path.exists(src_inf):",
    "    shutil.copytree(src_inf, dst_inf, dirs_exist_ok=True)",
    "    print(f'✅ Inference model saved to Drive')",
    "",
    "print(f'\\nAll outputs at: {DRIVE_OUT}')",
    "print('Download det_inference_finetuned/ to your laptop when ready.')",
]),

md("""## ✅ All Done!

After the notebook completes:
1. Download `output/det_inference_finetuned/` from Drive to your laptop
2. Place it in `Rumsey_Map_OCR/output/det_inference_finetuned/`
3. Run `step4_evaluate_and_infer.py` on your laptop to see the improved results

**Expected improvement: 27% → 50-60%+ recall** 🎯
"""),

]

# ── Write .ipynb ──────────────────────────────────────────────────────────

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "cells": cells,
}

out_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "colab_detection_finetune.ipynb")
)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook generated: {out_path}")
print()
print("📋 Upload to Google Drive:")
print(f"  1. colab_detection_finetune.ipynb   → open in Colab")
print(f"  2. train_data/det/                  → My Drive/Rumsey_OCR/train_data/det/")
print(f"  3. training_setup/configs/det_icdar_finetune.yml → My Drive/Rumsey_OCR/det_icdar_finetune.yml")
