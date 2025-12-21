import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from train import TransferCRNN, OCRDataset, ALPHABET, IMG_WIDTH, IMG_HEIGHT, decode_prediction

# --- CONFIG ---
DEVICE = torch.device("cuda")
MODEL_PATH = "outputs/best_model.pth"
VAL_DIR = "dataset_ready/val"

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if len(s2) == 0: return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def main():
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = OCRDataset(VAL_DIR, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = TransferCRNN(len(ALPHABET)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    total_cer, total_wer, total_chars, total_words = 0, 0, 0, 0
    comparisons = []
    skipped_count = 0

    print(f"🧪 Auditing {len(dataset)} validation samples (Filtering length < 3)...")

    for i, (images, labels, label_lengths) in enumerate(tqdm(loader)):
        images = images.to(DEVICE)
        
        # Get Ground Truth
        gt_indices = labels.numpy().flatten()
        gt_text = "".join([ALPHABET[i-1] for i in gt_indices if i > 0])
        
        # Get Prediction
        with torch.no_grad():
            preds = model(images)
            pred_text = decode_prediction(preds)[0]

        # --- NORMALIZATION ---
        gt_clean = gt_text.lower().replace(".", "").replace("#", "").strip()
        pred_clean = pred_text.lower().replace(".", "").replace("#", "").strip()

        # --- FILTER: Skip single chars/noise ---
        if len(gt_clean) < 3:
            skipped_count += 1
            continue

        # Log first valid comparisons for auditing
        if len(comparisons) < 10:
            comparisons.append(f"🎯 Target: '{gt_text}' | 🚀 Pred: '{pred_text}'")

        # Calculate metrics on CLEANED text
        dist = levenshtein_distance(pred_clean, gt_clean)
        total_cer += dist
        total_chars += len(gt_clean)
        
        if pred_clean != gt_clean:
            total_wer += 1
        total_words += 1

    print(f"\n🔍 --- AUDIT LOG (First 10 Valid) ---")
    for comp in comparisons:
        print(comp)
    
    print(f"\nℹ️ Skipped {skipped_count} short/noise samples.")

    if total_words > 0:
        cer = (total_cer / total_chars) * 100
        wer = (total_wer / total_words) * 100
        print(f"\n🎯 --- FINAL ACCURACY REPORT (Phase 8) ---")
        print(f"✅ Character Accuracy: {100 - cer:.2f}%")
        print(f"✅ Word Accuracy:      {100 - wer:.2f}%")
        print(f"❌ CER: {cer:.2f}% | WER: {wer:.2f}%")
    else:
        print("❌ No valid samples found (all were < 3 chars). Check your validation data.")

if __name__ == "__main__":
    main()