import sys
import os
import cv2
import glob
import random
import logging

# --- 1. SETUP PATHS (Exact copy from your working script) ---
# We point to your specific tools folder
TOOLS_PATH = os.path.join(os.getcwd(), "PaddleOCR_Official_Tools")
sys.path.append(TOOLS_PATH)

try:
    # We import the low-level engine that works for you
    from tools.infer.predict_system import TextSystem
    from tools.infer.utility import parse_args
except ImportError:
    print("❌ Error: Could not find PaddleOCR tools. Run this from the 'Rumsey_Map_OCR' folder.")
    sys.exit(1)

# --- CONFIGURATION ---
IMAGE_FOLDER = r"C:\Users\sharj\Desktop\Rumsey_Map_OCR\Rumsey_Map_OCR_Data\rumsey\icdar24-train-png\train_images"
OUTPUT_FOLDER = "training_hard_examples"
NUM_MAPS_TO_SCAN = 10

# Mining Thresholds
MIN_CONF = 0.50
MAX_CONF = 0.85

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 2. Configure the Engine (Exact copy of your working config)
    # This uses your LOCAL models, so no downloading crashes!
    print("🔹 Initializing Local Engine...")
    
    args = parse_args()
    # Point to your local inference models
    args.det_model_dir = "./inference/ch_PP-OCRv4_det_infer/"
    args.rec_model_dir = "./output/rec_inference/"
    args.rec_char_dict_path = "PaddleOCR_Official_Tools/ppocr/utils/en_dict.txt"
    
    # Critical Settings
    args.use_angle_cls = False
    args.use_gpu = False  # CPU for stability
    
    # Initialize
    try:
        text_sys = TextSystem(args)
        print("✅ Engine Loaded Successfully!")
    except Exception as e:
        print(f"❌ Error loading engine: {e}")
        return

    # 3. Find Images
    image_files = glob.glob(os.path.join(IMAGE_FOLDER, "*.png")) + \
                  glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))
    
    if not image_files:
        print("❌ No images found in folder.")
        return

    selected_files = random.sample(image_files, min(len(image_files), NUM_MAPS_TO_SCAN))
    print(f"⛏️  Mining {len(selected_files)} random maps for 'Hard Examples'...")

    total_saved = 0

    # 4. Scan and Save Confusion
    for idx, img_file in enumerate(selected_files):
        fname = os.path.basename(img_file)
        print(f"   Scanning [{idx+1}/{NUM_MAPS_TO_SCAN}]: {fname}...", end="\r")

        img = cv2.imread(img_file)
        if img is None: continue

        try:
            # Run Inference
            preds = text_sys(img)
            dt_boxes = preds[0]  # Boxes
            rec_res = preds[1]   # Text + Confidence
            
            if dt_boxes is None or rec_res is None:
                continue

            for box, res in zip(dt_boxes, rec_res):
                text, score = res
                
                # THE LOGIC: If confidence is between 50% and 85%
                if MIN_CONF <= score <= MAX_CONF:
                    
                    # Crop logic
                    try:
                        # box is numpy array, convert to list of points
                        xs = [p[0] for p in box]
                        ys = [p[1] for p in box]
                        
                        x_min = max(0, int(min(xs)) - 5)
                        x_max = min(img.shape[1], int(max(xs)) + 5)
                        y_min = max(0, int(min(ys)) - 5)
                        y_max = min(img.shape[0], int(max(ys)) + 5)
                        
                        crop = img[y_min:y_max, x_min:x_max]
                        
                        if crop.size == 0: continue

                        # Save as: CONFIDENCE_TEXT_ID.jpg
                        safe_text = "".join([c for c in text if c.isalnum()])[:15]
                        save_name = f"{score:.2f}_{safe_text}_{random.randint(1000,9999)}.jpg"
                        
                        cv2.imwrite(os.path.join(OUTPUT_FOLDER, save_name), crop)
                        total_saved += 1
                    except:
                        pass

        except Exception as e:
            continue

    print(f"\n\n✅ DONE! Saved {total_saved} confused images to '{OUTPUT_FOLDER}/'")

if __name__ == "__main__":
    main()