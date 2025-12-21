import os
import cv2
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
# Path to the folder containing the 200 large PNG tiles
BASE_DIR = r"C:\Users\sharj\Desktop\Rumsey_Map_OCR\Rumsey_Map_OCR_Data\rumsey\icdar24-train-png\train_images"
# Path to your annotations file
JSON_PATH = r"C:\Users\sharj\Desktop\Rumsey_Map_OCR\Rumsey_Map_OCR_Data\rumsey\icdar24-train-png\annotations.json"

OUTPUT_WORDS_DIR = "dataset_ready/train_words"
OUTPUT_CSV = "dataset_ready/train_words/labels.csv"

def main():
    if not os.path.exists(JSON_PATH):
        print(f"❌ Error: Could not find JSON at {JSON_PATH}")
        return
    if not os.path.exists(BASE_DIR):
        print(f"❌ Error: Could not find Image Folder at {BASE_DIR}")
        return

    os.makedirs(OUTPUT_WORDS_DIR, exist_ok=True)
    
    print(f"📖 Loading annotations...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    word_records = []

    print(f"✂️ Slicing map tiles into individual words...")
    for entry in tqdm(data, desc="Processing Tiles"):
        # 1. Get filename string directly
        img_filename = entry.get('image')
        if not img_filename:
            continue

        img_path = os.path.join(BASE_DIR, img_filename)
        image = cv2.imread(img_path)
        if image is None:
            continue

        # 2. Iterate through 'groups' (list of lists)
        groups = entry.get('groups', [])
        for g_idx, group in enumerate(groups):
            # Each 'group' is a list of word dictionaries
            for w_idx, word_item in enumerate(group):
                text = str(word_item.get('text', '')).strip()
                
                # Skip illegible or empty text
                if not text or text == "###" or word_item.get('illegible') is True: 
                    continue

                # 3. Calculate BBox from "vertices"
                vertices = word_item.get('vertices', [])
                if not vertices:
                    continue

                # Convert vertices to x, y, w, h
                vertices = np.array(vertices)
                x_min, y_min = np.min(vertices, axis=0)
                x_max, y_max = np.max(vertices, axis=0)
                
                x, y = int(max(0, x_min)), int(max(0, y_min))
                w, h = int(x_max - x_min), int(y_max - y_min)
                
                # Crop the word snippet
                crop = image[y:y+h, x:x+w]
                if crop.size == 0: 
                    continue

                # Save the snippet
                clean_name = img_filename.replace('.', '_')
                save_name = f"word_{clean_name}_g{g_idx}_w{w_idx}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_WORDS_DIR, save_name), crop)
                
                word_records.append({"filename": save_name, "words": text})

    # Save final labels
    if word_records:
        df = pd.DataFrame(word_records)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ SUCCESS! Extracted {len(word_records)} words.")
        print(f"📂 New Training Directory: {OUTPUT_WORDS_DIR}")
    else:
        print("\n❌ Extraction failed. No words matched the criteria.")

if __name__ == "__main__":
    main()