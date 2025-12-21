import os
import easyocr
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
# Point this to a folder where you have UNLABELED map crops
UNLABELED_DIR = "dataset_ready/unlabeled_crops" 
OUTPUT_CSV = "dataset_ready/train/pseudo_labels.csv"

def main():
    if not os.path.exists(UNLABELED_DIR):
        print(f"❌ Error: {UNLABELED_DIR} not found. Please put unlabeled crops there.")
        return

    reader = easyocr.Reader(['en']) # The "Teacher"
    images = [f for f in os.listdir(UNLABELED_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    results = []
    print(f"🧠 Teacher is labeling {len(images)} images...")

    for img_name in tqdm(images):
        img_path = os.path.join(UNLABELED_DIR, img_name)
        # detail=0 gives just the text string
        prediction = reader.readtext(img_path, detail=0)
        
        if prediction:
            text = prediction[0]
            results.append({"filename": img_name, "words": text})

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Success! Generated {len(results)} pseudo-labels in {OUTPUT_CSV}")

if __name__ == "__main__":
    main()