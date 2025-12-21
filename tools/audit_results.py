import pandas as pd
import os

# Path to your final database
csv_path = "outputs/final_map_database.csv"

if not os.path.exists(csv_path):
    print(f"❌ Could not find {csv_path}. Did the export finish?")
else:
    df = pd.read_csv(csv_path)
    print("🔍 --- FINAL DATABASE AUDIT (50 RANDOM SAMPLES) ---")
    
    # We use .sample() to get a broad look at the model's performance across the whole map
    print(df.sample(min(50, len(df)))[['extracted_text', 'image_id']])
    
    print(f"\n📈 Total searchable words extracted: {len(df)}")