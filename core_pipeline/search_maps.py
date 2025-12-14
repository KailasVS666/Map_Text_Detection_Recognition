import pandas as pd
import argparse
import os

# --- CONFIGURATION ---
# We use your new ELITE dataset for the best results
CSV_FILE = "map_text_results_elite.csv"

def search(query):
    if not os.path.exists(CSV_FILE):
        print("❌ Error: Elite data file not found. Run boost_accuracy.py first.")
        return

    print(f"🔎 Searching for '{query}'...")
    df = pd.read_csv(CSV_FILE)
    
    # Case-insensitive search
    # We convert the column to string to handle any edge cases
    mask = df['Detected Text'].astype(str).str.contains(query, case=False, na=False)
    results = df[mask]

    if len(results) == 0:
        print(f"❌ No matches found for '{query}'")
        return

    print(f"✅ Found {len(results)} matches:")
    print("-" * 60)
    
    for i, row in results.iterrows():
        text = row['Detected Text']
        file = row['Filename']
        conf = float(row['Confidence']) * 100
        box = row['Box Coordinates']
        
        print(f"📍 Map: {file}")
        print(f"   Label: '{text}'") 
        print(f"   Confidence: {conf:.1f}%")
        print(f"   Location: {box}")
        print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search your historic map database.")
    parser.add_argument("query", type=str, help="The word to find (e.g., 'River', 'City')")
    args = parser.parse_args()
    
    search(args.query)