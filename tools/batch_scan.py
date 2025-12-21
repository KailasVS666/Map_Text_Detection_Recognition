import os
import cv2
import torch
import easyocr
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from train import TransferCRNN, ALPHABET, IMG_WIDTH, IMG_HEIGHT, decode_prediction

# --- CONFIG ---
DEVICE = torch.device("cuda")
MODEL_PATH = "outputs/best_model.pth"
INPUT_FOLDER = r"C:\Users\sharj\Desktop\Rumsey_Map_OCR\Rumsey_Map_OCR_Data\rumsey\icdar24-train-png\train_images"
OUTPUT_FOLDER = "outputs/batch_results"

# --- 1. SETUP ---
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
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
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.png', '.jpg'))][:10] # Test with 10
    print(f"🚀 Starting Batch Scan on {len(image_files)} images...")

    for filename in tqdm(image_files):
        img_path = os.path.join(INPUT_FOLDER, filename)
        image = cv2.imread(img_path)
        display_img = image.copy()

        # Detect
        detection_results = reader.detect(image)[0][0]

        # Recognize
        for box in detection_results:
            x_min, x_max = int(min(box[0], box[1])), int(max(box[0], box[1]))
            y_min, y_max = int(min(box[2], box[3])), int(max(box[2], box[3]))
            crop = image[y_min:y_max, x_min:x_max]
            if crop.size == 0: continue
            
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            input_tensor = transform(pil_crop).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                text = decode_prediction(model(input_tensor))[0]

            cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(display_img, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"res_{filename}"), display_img)

    print(f"✅ Batch complete. Results in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()