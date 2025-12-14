import pandas as pd
import cv2
import os
import numpy as np
import ast
import albumentations as A
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_CSV = 'map_ocr_results_STRICT_CLEAN.csv'
IMAGE_DIR = 'Rumsey_Map_OCR_Data/rumsey/icdar24-val-png/val_images'

# Output paths
OUTPUT_DIR = 'PaddleOCR/train_data/rec/aug_crops'
LABEL_FILE = 'PaddleOCR/train_data/rec_gt_augmented.txt'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- AUGMENTATION PIPELINE (Safe Mode) ---
# We use only core transforms that work in all versions
transform = A.Compose([
    # 1. Geometric: Simple rotation without complex padding args
    A.Rotate(limit=10, p=0.5), 
    
    # 2. Distortion: GridDistortion is great for "warped" old maps
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),

    # 3. Text Quality & Ink Simulation
    A.OneOf([
        # Blur = simulates ink bleeding
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        
        # Noise = simulates scan grain
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        
        # Brightness = simulates faded paper
        A.RandomBrightnessContrast(p=0.5),
        
        # CoarseDropout = simulates broken/flaking ink (holes in letters)
        A.CoarseDropout(max_holes=8, max_height=4, max_width=4, min_holes=1, p=0.3),
    ], p=0.8),
])

def generate_augmented_data():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found.")
        return

    print(f"📖 Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    label_lines = []
    
    print(f"🚀 Generating data from {len(df)} original samples...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Load Image
        img_path = os.path.join(IMAGE_DIR, row['image_file'])
        if not os.path.exists(img_path): continue
        
        full_img = cv2.imread(img_path)
        if full_img is None: continue

        try:
            # Parse BBox
            box = np.array(ast.literal_eval(row['bbox_coords']), dtype=np.int32)
            x_min, y_min = np.min(box, axis=0)
            x_max, y_max = np.max(box, axis=0)
            
            # Add padding
            h, w, _ = full_img.shape
            pad = 5
            y_min = max(0, y_min - pad); y_max = min(h, y_max + pad)
            x_min = max(0, x_min - pad); x_max = min(w, x_max + pad)
            
            crop = full_img[y_min:y_max, x_min:x_max]
            if crop.size == 0: continue

            # --- 1. SAVE ORIGINAL (Baseline) ---
            orig_name = f"orig_{idx}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, orig_name), crop)
            label_lines.append(f"rec/aug_crops/{orig_name}\t{row['text']}")

            # --- 2. GENERATE 3 AUGMENTED VERSIONS ---
            for i in range(3):
                # Apply Augmentation
                augmented = transform(image=crop)['image']
                
                aug_name = f"aug_{i}_{idx}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, aug_name), augmented)
                label_lines.append(f"rec/aug_crops/{aug_name}\t{row['text']}")

        except Exception as e:
            # Just skip bad crops without crashing
            continue

    # Write Master Label File
    with open(LABEL_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(label_lines))

    print("-" * 30)
    print(f"✅ Augmentation Complete!")
    print(f"📊 Total Training Samples Created: {len(label_lines)}")
    print(f"💾 Label File: {LABEL_FILE}")
    print("-" * 30)

if __name__ == "__main__":
    generate_augmented_data()