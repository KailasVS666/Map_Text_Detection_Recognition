import os
import cv2
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import easyocr
from train import TransferCRNN, ALPHABET, IMG_WIDTH, IMG_HEIGHT, decode_prediction

# --- CONFIGURATION ---
DEVICE = torch.device("cuda")
MODEL_PATH = "outputs/best_model.pth"
INPUT_FOLDER = r"C:\Users\sharj\Desktop\Rumsey_Map_OCR\Rumsey_Map_OCR_Data\rumsey\icdar24-train-png\train_images"
OUTPUT_CSV = "outputs/map_ocr_database.csv"

# --- 1. INITIALIZATION ---
print("🚀 Initializing Export Engine...")
reader = easyocr.Reader(['en'], gpu=True)
model = TransferCRNN(len(ALPHABET)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def main():
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.png', '.jpg'))][:50]
    all_results = []

    print(f"📡 Processing {len(image_files)} maps into database...")

    for filename in tqdm(image_files):
        img_path = os.path.join(INPUT_FOLDER, filename)
        image = cv2.imread(img_path)
        
        # Detection
        detection_results = reader.detect(image)[0][0]

        for box in detection_results:
            x_min, x_max = int(min(box[0], box[1])), int(max(box[0], box[1]))
            y_min, y_max = int(min(box[2], box[3])), int(max(box[2], box[3]))
            
            crop = image[y_min:y_max, x_min:x_max]
            if crop.size == 0: continue
            
            # Recognition
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            input_tensor = transform(pil_crop).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                text = decode_prediction(model(input_tensor))[0]

            # Save metadata
            all_results.append({
                "filename": filename,
                "text": text,
                "x_min": x_min,
                "y_min": y_min,
                "width": x_max - x_min,
                "height": y_max - y_min
            })

    # --- 2. SAVE TO CSV ---
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Database created: {OUTPUT_CSV}")
    print(f"📝 Total entries extracted: {len(df)}")

if __name__ == "__main__":
    main()