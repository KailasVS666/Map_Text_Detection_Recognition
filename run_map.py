import os
import logging
from paddleocr import PaddleOCR

# --- CONFIGURATION ---
REC_MODEL_DIR = 'PaddleOCR/output/rec_finetune/best_accuracy'
IMAGE_PATH = 'image_cadae5.jpg' 

# Suppress debug logs
logging.getLogger('ppocr').setLevel(logging.ERROR)

print(f"🚀 Initializing OCR with custom model: {REC_MODEL_DIR}")

# --- FIX: Use New API Arguments & Force CPU ---
# We use 'device="cpu"' to prevent the silent crash. 
# Once this works, you can try changing it back to 'gpu'.
ocr = PaddleOCR(
    lang='en',
    use_textline_orientation=True,             # Correct new argument
    text_recognition_model_dir=REC_MODEL_DIR,  # Correct new argument
    device='cpu'                               # Correct argument (replaces use_gpu)
)

if not os.path.exists(IMAGE_PATH):
    print(f"❌ Error: Image not found at {IMAGE_PATH}")
else:
    print(f"🔎 Scanning image: {IMAGE_PATH}...")
    
    try:
        # Run Inference
        result = ocr.ocr(IMAGE_PATH, cls=True)

        print("-" * 50)
        # Handle cases where result is None (no text found at all)
        if result and result[0]:
            count = 0
            for line in result[0]:
                if line:
                    text = line[1][0]
                    score = line[1][1]
                    
                    if score > 0.65:
                        print(f"✅ Found: '{text}' (Conf: {score:.4f})")
                        count += 1
                    else:
                        print(f"⚠️ Found: '{text}' (Conf: {score:.4f})")
            
            if count == 0:
                print("⚠️ No high-confidence text found.")
        else:
            print("⚠️ No text detected.")
        print("-" * 50)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Inference Error: {e}")