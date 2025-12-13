import os
import cv2
from paddleocr import PaddleOCR

# --- CONFIGURATION ---
# Path to your custom trained model
REC_MODEL_DIR = 'PaddleOCR/output/rec_finetune/best_accuracy'
# Path to your map image
IMAGE_PATH = 'image_cadae5.jpg' 

print(f"🚀 Initializing OCR with custom model: {REC_MODEL_DIR}")

# Initialize PaddleOCR
# - use_angle_cls=True: Rotates text if needed (good for maps)
# - det=True: Uses the default pre-trained detector (Text Snake/DB)
# - rec=True: Uses YOUR custom trained model
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    rec_model_dir=REC_MODEL_DIR,
    use_gpu=True,
    show_log=False
)

if not os.path.exists(IMAGE_PATH):
    print(f"❌ Error: Image not found at {IMAGE_PATH}")
else:
    print(f"🔎 Scanning image: {IMAGE_PATH}...")
    # Run Inference
    # cls=True enables angle classification
    result = ocr.ocr(IMAGE_PATH, cls=True)

    print("-" * 50)
    # Result structure: [ [box, (text, score)], ... ]
    if result and result[0]:
        for idx, line in enumerate(result[0]):
            box = line[0]
            text, score = line[1]
            print(f"📍 Found: '{text}' (Conf: {score:.4f})")
    else:
        print("⚠️ No text detected.")
    print("-" * 50)