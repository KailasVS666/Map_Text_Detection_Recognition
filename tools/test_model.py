import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# --- MODEL CLASS (Must match train.py exactly) ---
class CRNN(nn.Module):
    def __init__(self, num_chars):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((4, 1))
        )
        self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(256 * 2, num_chars + 1)

    def forward(self, x):
        features = self.cnn(x)
        features = features.squeeze(2)
        features = features.permute(2, 0, 1)
        rnn_out, _ = self.rnn(features)
        return self.fc(rnn_out)

ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
rev_char_map = {i + 1: char for i, char in enumerate(ALPHABET)}

def decode(logits):
    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=2)
    conf, tokens = torch.max(probs, dim=2)
    
    token_seq = tokens[:, 0]
    conf_seq = conf[:, 0]
    
    result = []
    total_conf = 0
    char_count = 0
    prev = 0
    
    for i, t in enumerate(token_seq):
        if t != prev and t != 0:
            char = rev_char_map.get(t.item(), '')
            result.append(char)
            total_conf += conf_seq[i].item()
            char_count += 1
        prev = t
        
    final_text = "".join(result)
    avg_conf = (total_conf / char_count * 100) if char_count > 0 else 0
    
    return final_text, avg_conf

def run_test(img_path):
    device = torch.device("cpu")
    model = CRNN(len(ALPHABET))
    model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(img_path).convert('L')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        preds = model(image)
        text, confidence = decode(preds)
        print(f"\n🖼️  Image: {os.path.basename(img_path)}")
        print(f"🔍 AI Prediction: '{text}'")
        print(f"📊 Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    # LOOK IN YOUR 'dataset_ready/val' FOLDER AND PICK ANY FILE
    test_file = "C:/Users/sharj/Desktop/Rumsey_Map_OCR/dataset_ready/val/0.76_Agen_1938.jpg"
    
    if os.path.exists(test_file):
        run_test(test_file)
    else:
        print(f"File {test_file} not found. Check the filename in dataset_ready/val/")