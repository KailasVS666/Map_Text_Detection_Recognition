import pandas as pd
import os

# --- PATHS ---
VAL_CSV = "dataset_ready/val/labels.csv"

def sync():
    if not os.path.exists(VAL_CSV):
        print("❌ Error: labels.csv not found!")
        return

    print("🔄 Synchronizing labels with actual filenames...")
    df = pd.read_csv(VAL_CSV)

    def extract_truth(filename):
        # Example: '0.76_Agen_1938.jpg' -> split by '_' -> index 1 is 'Agen'
        parts = str(filename).split('_')
        if len(parts) >= 2:
            return parts[1]
        return parts[0] # Fallback if filename format is different

    df['words'] = df['filename'].apply(extract_truth)
    df.to_csv(VAL_CSV, index=False)
    
    print("✅ Done! Validation labels now match the map text.")
    print("\nSample of new labels:")
    print(df.head(5))

if __name__ == "__main__":
    sync()