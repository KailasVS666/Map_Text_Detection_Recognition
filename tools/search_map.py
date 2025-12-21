import os
import pandas as pd
from rapidfuzz import process, fuzz
from PIL import Image

# --- CONFIG ---
DF_PATH = "outputs/final_map_database.csv"
IMAGE_DIR = "dataset_ready/train_words" # Folder containing the 33,117 images

df = pd.read_csv(DF_PATH)
unique_words = df['extracted_text'].astype(str).unique().tolist()

def search_map():
    print("🗺️ --- RUMSEY MAP SEARCH & VIEW ENGINE ---")
    while True:
        query = input("\nEnter location name (or 'exit'): ").strip()
        if query.lower() == 'exit': break
        
        matches = process.extract(query, unique_words, scorer=fuzz.WRatio, limit=5)
        
        print(f"\n🔍 Results for '{query}':")
        results_map = {}
        for i, (match_text, score, _) in enumerate(matches):
            count = len(df[df['extracted_text'] == match_text])
            results_map[i+1] = match_text
            print(f"[{i+1}] {match_text:15} | Match: {score:.1f}% | Occurrences: {count}")

        choice = input("\nEnter result number to view image (or press Enter to skip): ")
        if choice.isdigit() and int(choice) in results_map:
            selected_text = results_map[int(choice)]
            # Get the first image_id associated with this text
            img_id = df[df['extracted_text'] == selected_text]['image_id'].iloc[0]
            img_path = os.path.join(IMAGE_DIR, img_id)
            
            if os.path.exists(img_path):
                print(f"🖼️ Opening: {img_id}")
                Image.open(img_path).show()
            else:
                print(f"❌ Could not find image at {img_path}")

if __name__ == "__main__":
    search_map()