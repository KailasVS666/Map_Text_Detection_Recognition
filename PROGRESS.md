# 🗺️ Rumsey Map OCR — Progress Log
Last updated: 2026-02-20

---

## ✅ What's Been Done

### Step 1 — Dataset Preparation
- ICDAR 2024 dataset downloaded to `Rumsey_Map_OCR_Data/rumsey/`
- Word crops extracted to `train_data/rec/` (train_list.txt + val_list.txt)
- Scripts: `step1_extract_icdar_crops.py`

### Step 2 — Recognition Model Fine-tuning ✅ COMPLETE
- **Base model:** PP-OCRv4 English recognition (`en_PP-OCRv4_rec_train`)
- **Config:** `training_setup/configs/rec_icdar_finetune.yml`
- **Epochs:** 30
- **Best validation accuracy:** 62.61% (achieved at Epoch 21)
- **Checkpoint saved at:** `output/rec_finetune/best_accuracy.*`
- **Exported inference model:** `output/rec_inference_finetuned/`
- **Key fix for Windows:** `PaddleOCR_Official_Tools/ppocr/data/imaug/iaa_augment.py`
  - Mocks `torch` module to prevent `shm.dll` OSError on Windows

### Step 3 — Detection Model Fine-tuning ⏭️ NOT DONE YET
- Detection model fine-tuning was skipped; currently using baseline `ch_PP-OCRv4_det_infer`
- This is the main bottleneck for end-to-end recall

### Step 4 — Evaluation & Inference ✅ COMPLETE
- End-to-end recall on 10 sample maps: **27.3%** (limited by baseline detector)
- Full inference on 20 training maps:
  - **1,431 detections**
  - **90.8% average confidence**
- Results saved to: `results/map_text_results_finetuned.csv`

---

## 🔄 Current Status
> **Recognition model is fine-tuned and working.**
> Detection model is still the pre-trained baseline — this is the main limiting factor.

---

## 🚀 Next Steps (In Order)

### 1. Fine-tune Detection Model (HIGHEST IMPACT)
This will dramatically improve end-to-end recall (estimated 27% → 50-60%).

```powershell
# Set cuDNN path first (required every new terminal session)
$env:PATH = "C:\Users\sharj\AppData\Local\Programs\Python\Python312\Lib\site-packages\nvidia\cudnn\bin;" + $env:PATH

# Run from PaddleOCR_Official_Tools directory
cd C:\Users\sharj\Desktop\Rumsey_Map_OCR\PaddleOCR_Official_Tools
python tools/train.py -c ../training_setup/configs/det_icdar_finetune.yml
```
> ⚠️ Note: `det_icdar_finetune.yml` needs to be created first (ask AI to create it).

### 2. Hard Negative Mining
After detection fine-tuning, identify samples the recognition model gets wrong and retrain on those. Script: `step5_hard_negative_mining.py` (to be built).

### 3. Re-evaluate End-to-End
After both models are fine-tuned, re-run `step4_evaluate_and_infer.py` and check if recall improved.

---

## ⚙️ Environment Notes

### Running Training / Export (ALWAYS set cuDNN path first)
```powershell
$env:PATH = "C:\Users\sharj\AppData\Local\Programs\Python\Python312\Lib\site-packages\nvidia\cudnn\bin;" + $env:PATH
```

### Key Paths
| Item | Path |
|---|---|
| Fine-tuned rec model (training) | `output/rec_finetune/best_accuracy.*` |
| Fine-tuned rec model (inference) | `output/rec_inference_finetuned/` |
| Baseline det model | `inference/ch_PP-OCRv4_det_infer/` |
| Training config | `training_setup/configs/rec_icdar_finetune.yml` |
| Character dict | `PaddleOCR_Official_Tools/ppocr/utils/en_dict.txt` |
| Train data | `train_data/rec/` |

### Python
- Executable: `C:\Users\sharj\AppData\Local\Programs\Python\Python312\python.exe`
- cuDNN DLL location: `C:\Users\sharj\AppData\Local\Programs\Python\Python312\Lib\site-packages\nvidia\cudnn\bin\`
- PaddlePaddle GPU version installed, CUDA 11.7, GPU: RTX 30xx (Compute 8.6)

---

## 📌 Important: What's in .gitignore
The `PaddleOCR_Official_Tools/` folder is in `.gitignore`.
Any changes to files inside it must be force-added:
```bash
git add -f PaddleOCR_Official_Tools/path/to/file
```
