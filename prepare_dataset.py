import pandas as pd
import cv2
import os
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIGURATION ---
# Path to your stitched CSV
CSV_PATH = 'map_ocr_results_STITCHED.csv'

# Path to the folder containing your raw map images
# ⚠️ VERIFY THIS PATH: Based on your notebook, we assume 'data/val_images'
IMAGE_DIR = 'Rumsey_Map_OCR_Data/rumsey/icdar24-val-png/val_images'

# Where to save the training data for PaddleOCR
OUTPUT_DIR = 'PaddleOCR/train_data/rec'
# ---------------------

def prepare_dataset():
    # 1. Verification Checks
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Error: Image directory not found: {IMAGE_DIR}")
        print("   Please edit the 'IMAGE_DIR' variable in this script to point to your map tiles.")
        return

    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: CSV file not found: {CSV_PATH}")
        return

    print(f"📖 Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    # 2. Setup Output Directories
    # Crops will go into PaddleOCR/train_data/rec/crops
    crop_dir = os.path.join(OUTPUT_DIR, 'crops')
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure the parent directory is created

    data_lines = []
    print(f"✂️  Cropping text from {len(df)} records...")

    # 3. Crop Images
    for index, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row['image_file']
        text = str(row['text']).strip()
        
        if len(text) < 1: continue

        img_path = os.path.join(IMAGE_DIR, img_name)
        
        if not os.path.exists(img_path):
            continue

        # OpenCV needs to be able to read all images. This assumes images are in PNG/JPG format.
        img = cv2.imread(img_path)
        if img is None: continue

        try:
            # Parse bounding box string to list
            box = ast.literal_eval(row['bbox_coords'])
            box = np.array(box, dtype=np.int32)

            # Get coordinates (handle rotational/tilted boxes by using min/max)
            x_min = max(0, np.min(box[:, 0]))
            x_max = min(img.shape[1], np.max(box[:, 0]))
            y_min = max(0, np.min(box[:, 1]))
            y_max = min(img.shape[0], np.max(box[:, 1]))

            if x_max - x_min < 5 or y_max - y_min < 5:
                continue 

            # Crop the region
            crop = img[y_min:y_max, x_min:x_max]
            
            # Save crop to disk
            crop_filename = f"crop_{index}_{os.path.splitext(img_name)[0]}.jpg"
            crop_save_path = os.path.join(crop_dir, crop_filename)
            cv2.imwrite(crop_save_path, crop)

            # Record relative path and label for PaddleOCR
            # Format: rec/crops/filename.jpg [tab] Label
            rel_path = os.path.join('rec/crops', crop_filename).replace('\\', '/') # Use forward slash for PaddleOCR
            data_lines.append(f"{rel_path}\t{text}")

        except Exception as e:
            continue

    print(f"✅ Generated {len(data_lines)} valid text crops.")

    if len(data_lines) == 0:
        print("❌ No crops generated. Check image path, filenames, and contents.")
        return

    # 4. Split and Save Labels
    train_lines, val_lines = train_test_split(data_lines, test_size=0.1, random_state=42)

    # Write the standard label files PaddleOCR expects
    with open(os.path.join(OUTPUT_DIR, 'rec_gt_train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    
    with open(os.path.join(OUTPUT_DIR, 'rec_gt_val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))

    print(f"💾 Saved label files to {OUTPUT_DIR}")
    print(f"   - Training Samples: {len(train_lines)}")
    print(f"   - Validation Samples: {len(val_lines)}")

if __name__ == "__main__":
    prepare_dataset()