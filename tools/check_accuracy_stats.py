import pandas as pd
import random

# --- CONFIGURATION ---
CSV_FILE = "map_text_results_linked.csv"

def main():
    print("📊 Loading data...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print("❌ CSV not found.")
        return

    total_words = len(df)
    avg_conf = df['Confidence'].mean()
    
    # 1. High Confidence Count
    high_conf_df = df[df['Confidence'] >= 0.90]
    high_conf_count = len(high_conf_df)
    
    # 2. Low Confidence Count (Potential Errors)
    low_conf_df = df[df['Confidence'] < 0.85]
    low_conf_count = len(low_conf_df)

    print("-" * 40)
    print(f"🔍 AUTOMATIC STATS FOR {total_words} DETECTIONS")
    print("-" * 40)
    print(f"✅ Average Model Confidence:  {avg_conf*100:.2f}%")
    print(f"💎 High Confidence (>90%):    {high_conf_count} ({high_conf_count/total_words*100:.1f}%)")
    print(f"⚠️ Low Confidence (<85%):     {low_conf_count} ({low_conf_count/total_words*100:.1f}%)")
    print("-" * 40)
    
    print("\n🕵️‍♂️ MANUAL SAMPLING (The 'Real' Accuracy)")
    print("Check these 5 random examples against your maps:")
    print("-" * 40)
    
    sample = df.sample(5)
    for i, row in sample.iterrows():
        print(f"File: {row['Filename']}")
        print(f"  AI Read: '{row['Detected Text']}'")
        print(f"  Certainty: {row['Confidence']*100:.1f}%")
        print("")

    print("👉 RULE OF THUMB: If 4 out of these 5 are correct, your accuracy is approx 80%.")
    print("👉 If 5 out of 5 are correct, you are likely above 90% accuracy.")

if __name__ == "__main__":
    main()