import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import v2 
from PIL import Image
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
TRAIN_DIR = "dataset_ready/train_words"
VAL_DIR = "dataset_ready/val"
BATCH_SIZE = 64         # Optimized for RTX 3050 6GB
EPOCHS = 50             # High epoch count allows Scheduler to work
LEARNING_RATE = 0.0001
IMG_HEIGHT = 32
IMG_WIDTH = 256
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
NUM_WORKERS = 4

# --- 1. DATASET CLASS ---
class OCRDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.transform = transform
        df = pd.read_csv(os.path.join(dir_path, "labels.csv"))
        self.data = df[df['filename'].apply(lambda x: os.path.exists(os.path.join(dir_path, str(x))))]
        print(f"📦 Loaded {len(self.data)} images from {dir_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.dir_path, str(row['filename']))
        text = str(row['words'])
        image = Image.open(img_path).convert('L') 
        if self.transform:
            image = self.transform(image)
        label = torch.IntTensor([char_map.get(c, 0) for c in text])
        return image, label, len(label)

# --- 2. UTILS ---
char_map = {char: i + 1 for i, char in enumerate(ALPHABET)}
rev_char_map = {i + 1: char for i, char in enumerate(ALPHABET)}

def decode_prediction(logits):
    tokens = torch.argmax(logits, dim=2)
    decoded_strings = []
    for i in range(tokens.size(1)):
        token_seq = tokens[:, i]
        result = [rev_char_map.get(t.item(), '') for t, p in zip(token_seq, torch.cat([torch.tensor([0]).to(token_seq.device), token_seq[:-1]])) if t != 0 and t != p]
        decoded_strings.append("".join(result))
    return decoded_strings

def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    return torch.stack(images), torch.cat(labels), torch.IntTensor(lengths)

# --- 3. THE MODEL ---
class TransferCRNN(nn.Module):
    def __init__(self, num_chars):
        super(TransferCRNN, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-3]) 
        self.backbone[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.rnn = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_chars + 1)

    def forward(self, x):
        features = self.backbone(x)
        features = features.mean(2) 
        features = features.permute(0, 2, 1) 
        rnn_out, _ = self.rnn(features)
        return self.fc(rnn_out).permute(1, 0, 2)

# --- 4. MAIN ---
def main():
    device = torch.device("cuda")
    print(f"🚀 Training with Auto-Slowdown on {torch.cuda.get_device_name(0)}")
    os.makedirs("outputs", exist_ok=True)

    # --- AUGMENTATION ---
    train_transform = v2.Compose([
        v2.RandomRotation(degrees=10),
        v2.RandomPerspective(distortion_scale=0.2, p=0.5),
        v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
        v2.Resize((IMG_HEIGHT, IMG_WIDTH)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5,), (0.5,))
    ])

    val_transform = v2.Compose([
        v2.Resize((IMG_HEIGHT, IMG_WIDTH)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5,), (0.5,))
    ])

    train_loader = DataLoader(OCRDataset(TRAIN_DIR, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(OCRDataset(VAL_DIR, val_transform), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = TransferCRNN(len(ALPHABET)).to(device)
    
    # Load previous best weights to start from where we left off (Optional but recommended)
    if os.path.exists("outputs/best_model.pth"):
        print("🔄 Resuming from previous best model...")
        model.load_state_dict(torch.load("outputs/best_model.pth", weights_only=True))
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- SCHEDULER: The "Auto-Slowdown" Feature ---
    # If val_loss doesn't improve for 3 epochs, cut LR by 50%
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, labels, label_lengths in pbar:
            images, labels, label_lengths = images.to(device), labels.to(device), label_lengths.to(device)
            optimizer.zero_grad()
            preds = model(images)
            input_lengths = torch.full((preds.size(1),), preds.size(0), dtype=torch.long).to(device)
            loss = criterion(preds.log_softmax(2), labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = sum(criterion(model(im.to(device)).log_softmax(2), la.to(device), torch.full((im.size(0),), 16, dtype=torch.long).to(device), ll.to(device)).item() for im, la, ll in val_loader) / len(val_loader)
        
        # --- UPDATE SCHEDULER ---
        scheduler.step(val_loss)

        print(f"✅ Epoch {epoch+1} | Val Loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "outputs/best_model.pth")
            print(f"   ✨ New Best Saved! Sample: '{decode_prediction(preds)[0]}'")

if __name__ == "__main__":
    main()