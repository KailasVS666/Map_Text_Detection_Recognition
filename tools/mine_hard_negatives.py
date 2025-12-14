import os
import random
import cv2
import logging
from paddleocr import PaddleOCR

# --- CONFIGURATION ---
IMAGE_FOLDER = r"C:\Users\sharj\Desktop\Rumsey_Map_OCR\Rumsey_Map_OCR_Data\rumsey\icdar24-train-png\train_images"
OUTPUT_FOLDER = "training_hard_examples"
NUM_MAPS_TO_SCAN = 10  # Scan 10 random maps

# 🎯 MINING SETTINGS
# We want the model's "weak" guesses.
# Anything between 50% and 85% is a "Hard Negative" or "Hard Positive".
MIN_CONF = 0.50
MAX_CONF = 0.85

def main():
    # Suppress Paddle logs
    logging.getLogger('ppocr').setLevel(logging.ERROR)

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # 1. Initialize OCR with ONLY the safe, standard arguments
    # We remove 'drop_score' and 'show_log' to prevent crashes.
    # We use 'det_db_thresh' to lower detection sensitivity (catch faint text).
    ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_thresh=0.1)

    # 2. Get Images
    try:
        all_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg'))]
    except FileNotFoundError:
        print(f"❌ Error: Image folder not found: {IMAGE_FOLDER}")
        return

    if not all_files:
        print("❌ No images found in the folder!")
        return
        
    selected_files = random.sample(all_files, min(len(all_files), NUM_MAPS_TO_SCAN))
    
    print(f"⛏️  Mining {len(selected_files)} random maps for 'hard' examples (Threshold: {MIN_CONF})...")
    
    total_saved = 0

    for idx, filename in enumerate(selected_files):
        print(f"   [{idx+1}/{NUM_MAPS_TO_SCAN}] Scanning {filename}...", end="\r")
        img_path = os.path.join(IMAGE_FOLDER, filename)
        
        # Run OCR
        try:
            result = ocr.ocr(img_path, cls=True)
        except Exception as e:
            continue

        if not result or result[0] is None:
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Process results
        for line in result[0]:
            box = line[0]
            text, conf = line[1]
            
            # 3. CAPTURE THE CONFUSION
            # If confidence is in the "Maybe" zone (50% - 85%)
            if MIN_CONF <= conf <= MAX_CONF:
                
                # Crop the confusion
                try:
                    x_min = max(0, int(min(p[0] for p in box)) - 5)
                    x_max = min(img.shape[1], int(max(p[0] for p in box)) + 5)
                    y_min = max(0, int(min(p[1] for p in box)) - 5)
                    y_max = min(img.shape[0], int(max(p[1] for p in box)) + 5)
                    
                    crop = img[y_min:y_max, x_min:x_max]
                    
                    # Sanitize filename
                    safe_text = "".join([c for c in text if c.isalnum()])[:15]
                    # Format: CONFIDENCE_TEXT_ID.jpg
                    save_name = f"{conf:.2f}_{safe_text}_{random.randint(1000,9999)}.jpg"
                    
                    cv2.imwrite(os.path.join(OUTPUT_FOLDER, save_name), crop)
                    total_saved += 1
                except Exception as e:
                    pass # Skip bad crops

    print(f"\n\n✅ DONE! Saved {total_saved} confused images to '{OUTPUT_FOLDER}/'")
    print("👉 Now open that folder. You will see exactly what confuses your model.")

if __name__ == "__main__":
    main()