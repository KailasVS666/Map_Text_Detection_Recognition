import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from train import TransferCRNN, ALPHABET, IMG_WIDTH, IMG_HEIGHT, decode_prediction
from tqdm import tqdm
import re

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "outputs/best_model.pth"
# UPDATE THIS TO YOUR FULL CROPS FOLDER
INPUT_DIR = "dataset_ready/train_words" 
OUTPUT_CSV = "outputs/final_map_database.csv"

def clean_text(text):
    """Post-processing to fix common map OCR hallucinations"""
    cleaned = re.sub(r'[#\.\*]', '', text)
    return cleaned.strip()

def export():
    # 1. Load Model safely
    model = TransferCRNN(len(ALPHABET)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # 2. Setup Transform
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    results = []
    image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"🚀 Generating Final Database for {len(image_files)} images...")

    for filename in tqdm(image_files):
        img_path = os.path.join(INPUT_DIR, filename)
        try:
            image = Image.open(img_path).convert('L')
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                preds = model(input_tensor)
                raw_text = decode_prediction(preds)[0]
                
            final_text = clean_text(raw_text)

            # Only export meaningful geographic data
            if len(final_text) >= 3:
                results.append({
                    "image_id": filename,
                    "extracted_text": final_text,
                    "raw_output": raw_text
                })
        except Exception:
            continue

    # 3. Save Final Project CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✨ Project Complete! Final data saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    export()