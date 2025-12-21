import torch
from torchvision import transforms
from PIL import Image
import os
from train import TransferCRNN, ALPHABET, IMG_WIDTH, IMG_HEIGHT, decode_prediction

def test_single_image(img_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = Image.open(img_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        preds = model(image)
        # We need to reshape for the decoder: [Time, Batch, Class]
        decoded = decode_prediction(preds)
    return decoded[0]

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransferCRNN(len(ALPHABET)).to(device)
model.load_state_dict(torch.load("outputs/best_model.pth"))

# Pick 5 images from your dataset to test
test_folder = "dataset_ready/train_words"
sample_files = os.listdir(test_folder)[:5]

print("\n🔍 --- MODEL INFERENCE TEST ---")
for file in sample_files:
    if file.endswith(".csv"): continue
    full_path = os.path.join(test_folder, file)
    prediction = test_single_image(full_path, model, device)
    print(f"📷 Image: {file} | 🤖 Prediction: '{prediction}'")