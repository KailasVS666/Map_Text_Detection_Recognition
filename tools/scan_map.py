import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import easyocr
from train import TransferCRNN, ALPHABET, IMG_WIDTH, IMG_HEIGHT, decode_prediction

# --- CONFIG ---
DEVICE = torch.device("cuda")
MODEL_PATH = "outputs/best_model.pth"
# Point this to ANY map image on your PC
TEST_IMAGE = r"C:\Users\sharj\Desktop\Rumsey_Map_OCR\Rumsey_Map_OCR_Data\rumsey\icdar24-train-png\train_images\0015111_h3_w6.png"

# --- 1. SETUP ---
print("🛰️ Loading Detector & Your Custom Model...")
reader = easyocr.Reader(['en'], gpu=True) 

model = TransferCRNN(len(ALPHABET)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def main():
    image = cv2.imread(TEST_IMAGE)
    display_img = image.copy()

    # --- 2. DETECT (Find Boxes) ---
    print("🔍 Scanning for text...")
    detection_results = reader.detect(image)[0][0] 

    # --- 3. RECOGNIZE (Read Boxes) ---
    for box in detection_results:
        x_min, x_max = int(min(box[0], box[1])), int(max(box[0], box[1]))
        y_min, y_max = int(min(box[2], box[3])), int(max(box[2], box[3]))
        
        crop = image[y_min:y_max, x_min:x_max]
        if crop.size == 0: continue
        
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        input_tensor = transform(pil_crop).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            preds = model(input_tensor)
            text = decode_prediction(preds)[0]

        cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(display_img, text, (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite("outputs/autonomous_scan.jpg", display_img)
    print("✅ Done! Result saved to: outputs/autonomous_scan.jpg")

if __name__ == "__main__":
    main()