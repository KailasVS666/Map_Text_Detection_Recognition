# 📉 Project Progress Log: Hard Negative Mining Phase
**Date:** December 14, 2025
**Current Status:** 🟡 Phase 2 - Active Learning & Data Cleaning

---

## 🚀 1. The Objective
After achieving ~96% confidence on "clean" text, we identified a critical weakness:
1.  **Hallucinations:** The model mistakes mountain hachures and map symbols for text (e.g., "l", "i", "m").
2.  **Missed "Hard" Text:** The model has low confidence (50-85%) on cursive, vertical, or noisy text.

**Goal:** Create a specialized "Hard Example" dataset to fine-tune the model, teaching it to distinguish between *real* cursive text and *fake* map texture.

---

## 🛠️ 2. Technical Achievements (The "Crash" Fix)
We encountered significant version conflicts with `paddleocr 3.3.2` on Windows (DLL conflicts and model download loops).

**The Solution:**
Instead of using the high-level `PaddleOCR` wrapper (which was broken), we bypassed it by importing the underlying `TextSystem` engine directly from the working `core_pipeline`.

* **Script Created:** `tools/mine_hard_negatives.py`
* **Engine:** Local `TextSystem` (bypasses internet check/download)
* **Thresholds:** * Detection Threshold: `0.1` (Aggressive search)
    * Mining Range: `0.50` - `0.85` confidence

---

## 📊 3. Mining Results
* **Source:** 10 Random Training Maps
* **Execution Time:** ~4 minutes
* **Images Mined:** 282 "Confused" Examples
* **Location:** `training_hard_examples/`

### Diagnosis of Mined Data
The model's confusion falls into two categories:
1.  **Type A (False Positives):** Mountain shadings, river lines, random circles. -> **ACTION: DELETE**
2.  **Type B (Hard Text):** Single letters ('R', 'I'), cursive suffixes ('-ville', '-ing'), and vertical text. -> **ACTION: KEEP**

---

## 📋 4. Current Task: The "Purge"
**Action Item:** Manually filtering the `training_hard_examples/` folder.

**Filtering Rules:**
-   ✅ **KEEP:** Single letters (R, I, A), Numbers (10, 500), Cursive fragments, Blurry but legitimate words.
-   ❌ **DELETE:** Rocks, trees, hachures, map border lines, random noise.

---

## ⏭️ 5. Next Steps
Once filtering is complete:
1.  **Labeling:** Use a lightweight tool to generate ground truth labels for the survivors.
2.  **Dataset Merging:** Add these "Hard Examples" to the main training set.
3.  **Retraining:** Run a fine-tuning session to update the recognition model.