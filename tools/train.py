import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd

# --- CONFIGURATION ---
TRAIN_DIR = "dataset_ready/train"
VAL_DIR = "dataset_ready/val"
BATCH_SIZE = 8  # Smaller batch for more stable fine-tuning
EPOCHS = 50 
LEARNING_RATE = 0.0001 # Lower rate for Transfer Learning
IMG_HEIGHT = 32
IMG_WIDTH = 128
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "

# --- 1. DATASET CLASS ---
class OCRDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.transform = transform
        self.data = pd.read_csv(os.path.join(dir_path, "labels.csv"))
        self.data = self.data[self.data['filename'].apply(lambda x: os.path.exists(os.path.join(dir_path, x)))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.dir_path, row['filename'])
        text = str(row['words'])
        image = Image.open(img_path).convert('L') 
        if self.transform:
            image = self.transform(image)
        label = text_to_labels(text)
        return image, label, len(label)

# --- 2. UTILS ---
char_map = {char: i + 1 for i, char in enumerate(ALPHABET)}
rev_char_map = {i + 1: char for i, char in enumerate(ALPHABET)}

def text_to_labels(text):
    return torch.IntTensor([char_map.get(c, 0) for c in text])

def decode_prediction(logits):
    tokens = torch.argmax(logits, dim=2)
    decoded_strings = []
    for i in range(tokens.size(1)):
        token_seq = tokens[:, i]
        result = []
        prev = 0
        for t in token_seq:
            if t != prev and t != 0:
                result.append(rev_char_map.get(t.item(), ''))
            prev = t
        decoded_strings.append("".join(result))
    return decoded_strings

# --- 3. THE OPTIMIZED MODEL (Transfer Learning) ---
class TransferCRNN(nn.Module):
    def __init__(self, num_chars):
        super(TransferCRNN, self).__init__()
        # Load a pre-trained ResNet-18 (Already knows shapes/textures)
        # Using weights=models.ResNet18_Weights.IMAGENET1K_V1 for the latest PyTorch versions
        resnet = models.resnet18(pretrained=True)
        
        # Modify the first layer to accept Grayscale (1 channel) instead of RGB (3)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        self.backbone[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # The output of ResNet18 is 512 channels
        self.rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_chars + 1)

    def forward(self, x):
        # CNN Feature Extraction
        features = self.backbone(x) # [B, 512, 1, 4]
        
        # Prepare for RNN
        features = features.permute(0, 3, 1, 2).flatten(2) # [B, 4, 512]
        
        rnn_out, _ = self.rnn(features) # [B, 4, 512]
        
        # CTC needs [Time, Batch, Classes]
        output = self.fc(rnn_out).permute(1, 0, 2)
        return output

# --- 4. MAIN LOOP ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Optimized Transfer Learning on {device}")
    os.makedirs("outputs", exist_ok=True)

    # Standard clean transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_loader = DataLoader(OCRDataset(TRAIN_DIR, transform=transform), batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=lambda b: (torch.stack([x[0] for x in b]), torch.cat([x[1] for x in b]), torch.IntTensor([x[2] for x in b])))
    val_loader = DataLoader(OCRDataset(VAL_DIR, transform=transform), batch_size=BATCH_SIZE, shuffle=False, 
                            collate_fn=lambda b: (torch.stack([x[0] for x in b]), torch.cat([x[1] for x in b]), torch.IntTensor([x[2] for x in b])))

    model = TransferCRNN(len(ALPHABET)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for images, labels, label_lengths in train_loader:
            images, labels, label_lengths = images.to(device), labels.to(device), label_lengths.to(device)
            optimizer.zero_grad()
            preds = model(images)
            input_lengths = torch.full((preds.size(1),), preds.size(0), dtype=torch.long).to(device)
            loss = criterion(preds.log_softmax(2), labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, labels, label_lengths in val_loader:
                images, labels, label_lengths = images.to(device), labels.to(device), label_lengths.to(device)
                preds = model(images)
                input_lengths = torch.full((preds.size(1),), preds.size(0), dtype=torch.long).to(device)
                val_loss += criterion(preds.log_softmax(2), labels, input_lengths, label_lengths).item()

        avg_train, avg_val = train_loss/len(train_loader), val_loss/len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), "outputs/best_model.pth")
            print(f"   ✨ Best Model Saved! Sample: '{decode_prediction(preds)[0]}'")

if __name__ == "__main__":
    main()