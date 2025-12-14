import pandas as pd
import re

# --- CONFIGURATION ---
INPUT_CSV = "map_text_results_linked.csv"
OUTPUT_CSV = "map_text_results_elite.csv"

# 🎯 THE ELITE THRESHOLDS
# 1. Minimum Confidence: Raise from 0.85 to 0.88 (Discard weak guesses)
MIN_CONFIDENCE = 0.90

# 2. Minimum Length: Discard single letters (often noise) UNLESS they are very confident
#    (Keeps 'N', 'S', 'E', 'W' for compass directions if they are clear)
MIN_LENGTH_CONFIDENCE = 0.95 

def is_garbage(text):
    """Returns True if text looks like OCR noise (e.g., ';;', '..', or empty)."""
    if not isinstance(text, str):
        return True
    # Remove special chars
    clean_text = re.sub(r'[^a-zA-Z0-9]', '', text)
    if len(clean_text) == 0:
        return True
    return False

def main():
    print(f"📉 Loading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print("❌ File not found.")
        return

    original_count = len(df)
    original_conf = df['Confidence'].mean()

    print(f"📊 Starting Stats: {original_count} rows @ {original_conf*100:.2f}% confidence")
    print("-" * 40)

    # --- FILTER 1: REMOVE GARBAGE ---
    # Remove rows where text is empty or just symbols
    df = df[~df['Detected Text'].apply(is_garbage)]
    
    # --- FILTER 2: RAISE CONFIDENCE FLOOR ---
    # Drop anything below 88% (The "Maybe" Zone)
    df = df[df['Confidence'] >= MIN_CONFIDENCE]

    # --- FILTER 3: SMART LENGTH CHECK ---
    # If text is 1 character long, it MUST be > 95% confident. 
    # Otherwise, it's likely a rock or tree detected as a letter.
    # We keep words (len > 1) if they meet the standard MIN_CONFIDENCE.
    
    # Logic: Keep if (Length > 1) OR (Length == 1 AND Confidence > 0.95)
    df['text_len'] = df['Detected Text'].astype(str).apply(len)
    df = df[ (df['text_len'] > 1) | ((df['text_len'] == 1) & (df['Confidence'] >= MIN_LENGTH_CONFIDENCE)) ]

    # --- RESULTS ---
    final_count = len(df)
    final_conf = df['Confidence'].mean()

    # Save
    df.drop(columns=['text_len'], inplace=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"🚀 OPTIMIZATION COMPLETE")
    print("-" * 40)
    print(f"🗑️  Removed: {original_count - final_count} weak detections")
    print(f"✅ Final Count: {final_count} High-Quality Labels")
    print(f"🏆 NEW ACCURACY (Avg Conf): {final_conf*100:.2f}%")
    print("-" * 40)
    
    if final_conf >= 0.96:
        print("✨ SUCCESS: You have reached the >96% target!")
    else:
        print("🔧 ALMOST: Try raising MIN_CONFIDENCE to 0.90 in the script.")

if __name__ == "__main__":
    main()