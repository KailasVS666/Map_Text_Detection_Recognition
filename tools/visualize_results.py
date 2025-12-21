import os
import cv2
import torch
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from train import TransferCRNN, ALPHABET, IMG_WIDTH, IMG_HEIGHT, decode_prediction

# --- CONFIG ---
DEVICE = torch.device("cuda")
MODEL_PATH = "outputs/best_model.pth"
# Pick one large tile from your ICDAR folder
TILE_PATH = r"C:\Users\sharj\Desktop\Rumsey_Map_OCR\Rumsey_Map_OCR_Data\rumsey\icdar24-train-png\train_images\0009008_h2_w3.png"
# Your original annotations to find the boxes
JSON_PATH = r"C:\Users\sharj\Desktop\Rumsey_Map_OCR\Rumsey_Map_OCR_Data\rumsey\icdar24-train-png\annotations.json"

def load_model():
    model = TransferCRNN(len(ALPHABET)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

def main():
    model = load_model()
    image = cv2.imread(TILE_PATH)
    display_img = image.copy()
    
    # Load JSON to find the boxes for this specific tile
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    tile_filename = os.path.basename(TILE_PATH)
    entry = next(item for item in data if item["image"] == tile_filename)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print(f"🧐 Processing tile: {tile_filename}...")

    for group in entry['groups']:
        for word_item in group:
            verts = np.array(word_item['vertices'])
            x_min, y_min = np.min(verts, axis=0).astype(int)
            x_max, y_max = np.max(verts, axis=0).astype(int)
            
            # Crop & Predict
            crop = image[y_min:y_max, x_min:x_max]
            if crop.size == 0: continue
            
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            input_tensor = transform(pil_crop).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                preds = model(input_tensor)
                text = decode_prediction(preds)[0]

            # Draw on Map
            cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(display_img, text, (x_min, y_min - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    output_path = "outputs/visual_result.jpg"
    cv2.imwrite(output_path, display_img)
    print(f"✅ Visualization saved to: {output_path}")

if __name__ == "__main__":
    main()