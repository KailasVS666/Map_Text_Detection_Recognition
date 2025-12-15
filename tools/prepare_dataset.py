import os
import shutil
import random
import csv

# --- CONFIGURATION ---
SOURCE_IMG_DIR = "training_hard_examples"
SOURCE_LABEL_FILE = "training_hard_examples/labels.txt"
OUTPUT_DIR = "dataset_ready"
SPLIT_RATIO = 0.9  # 90% Training, 10% Validation

def setup_dirs():
    """Creates the folder structure: dataset_ready/train and dataset_ready/val"""
    if os.path.exists(OUTPUT_DIR):
        response = input(f"⚠️  Folder '{OUTPUT_DIR}' already exists. Delete and remake? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(OUTPUT_DIR)
        else:
            print("❌ Operation cancelled.")
            exit()
    
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "val"), exist_ok=True)

def main():
    print(f"🔹 Preparing Dataset from {SOURCE_IMG_DIR}...")
    
    # 1. Read the labels
    data = []
    if not os.path.exists(SOURCE_LABEL_FILE):
        print("❌ Error: labels.txt not found.")
        return

    with open(SOURCE_LABEL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                filename, text = parts
                # Verify image actually exists before adding to list
                if os.path.exists(os.path.join(SOURCE_IMG_DIR, filename)):
                    data.append((filename, text))

    # 2. Shuffle and Split
    random.seed(42) # Fixed seed for reproducibility
    random.shuffle(data)
    
    split_idx = int(len(data) * SPLIT_RATIO)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"✅ Found {len(data)} valid pairs.")
    print(f"   - Training:   {len(train_data)} images")
    print(f"   - Validation: {len(val_data)} images")

    setup_dirs()

    # 3. Copy files and generate CSVs
    def process_set(dataset, split_name):
        csv_path = os.path.join(OUTPUT_DIR, split_name, "labels.csv")
        print(f"   --> Processing {split_name}...")
        
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "words"]) # Header
            
            for filename, text in dataset:
                # Copy Image
                src = os.path.join(SOURCE_IMG_DIR, filename)
                dst = os.path.join(OUTPUT_DIR, split_name, filename)
                shutil.copy2(src, dst)
                
                # Write to CSV
                writer.writerow([filename, text])

    process_set(train_data, "train")
    process_set(val_data, "val")

    print(f"\n🎉 Done! Data is ready in '{OUTPUT_DIR}/'")
    print(f"   Use '{OUTPUT_DIR}/train' for training.")
    print(f"   Use '{OUTPUT_DIR}/val' for evaluating.")

if __name__ == "__main__":
    main()