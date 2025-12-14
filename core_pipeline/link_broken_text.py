import pandas as pd
import ast
import math
import os

# --- CONFIGURATION ---
INPUT_CSV = "map_text_results.csv"
OUTPUT_CSV = "map_text_results_linked.csv"

# Tuning Parameters
# How far apart can letters be? (2.5x the letter height is a good standard)
MAX_HORIZONTAL_GAP_RATIO = 1.0
# How much can they shift up/down? (0.5x the letter height)
MAX_VERTICAL_SHIFT_RATIO = 0.2 

def get_box_height(box):
    """Calculates height of the box."""
    ys = [p[1] for p in box]
    return max(ys) - min(ys)

def get_box_center(box):
    """Calculates the center (x, y)."""
    x = [p[0] for p in box]
    y = [p[1] for p in box]
    return sum(x)/4, sum(y)/4

def merge_boxes(box1, box2):
    """Merges two boxes into one big box."""
    all_points = box1 + box2
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    return [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]

def process_map_data(df):
    """Merges broken text for a single map."""
    # Convert string boxes back to lists if needed
    if isinstance(df.iloc[0]['Box Coordinates'], str):
        df['Box Coordinates'] = df['Box Coordinates'].apply(ast.literal_eval)
        
    # Sort by Left-X coordinate to process left-to-right
    df['x_min'] = df['Box Coordinates'].apply(lambda b: min(p[0] for p in b))
    df = df.sort_values('x_min').reset_index(drop=True)

    merged_data = []
    used_indices = set()

    for i in range(len(df)):
        if i in used_indices:
            continue

        current_box = df.iloc[i]['Box Coordinates']
        current_text = str(df.iloc[i]['Detected Text'])
        current_conf = float(df.iloc[i]['Confidence'])
        current_h = get_box_height(current_box)

        # Iteratively look for the "next letter"
        has_merged = True
        while has_merged:
            has_merged = False
            best_merge_idx = -1
            
            # Check other unused boxes
            for j in range(len(df)):
                if i == j or j in used_indices:
                    continue
                
                next_box = df.iloc[j]['Box Coordinates']
                next_h = get_box_height(next_box)
                
                # 1. Height Check (Must be roughly same size)
                if abs(current_h - next_h) > (current_h * 0.5):
                    continue

                # 2. Distance Check (Right edge of A -> Left edge of B)
                curr_x_max = max(p[0] for p in current_box)
                next_x_min = min(p[0] for p in next_box)
                gap = next_x_min - curr_x_max
                
                allowed_gap = current_h * MAX_HORIZONTAL_GAP_RATIO
                
                # If gap is positive (next is to the right) and small enough
                if 0 < gap < allowed_gap:
                     # 3. Vertical Alignment Check
                     curr_y = get_box_center(current_box)[1]
                     next_y = get_box_center(next_box)[1]
                     
                     if abs(curr_y - next_y) < (current_h * MAX_VERTICAL_SHIFT_RATIO):
                         best_merge_idx = j
                         break # Found the closest neighbor
            
            # Merge Logic
            if best_merge_idx != -1:
                target = df.iloc[best_merge_idx]
                current_text += " " + str(target['Detected Text'])
                current_conf = (current_conf + float(target['Confidence'])) / 2
                current_box = merge_boxes(current_box, target['Box Coordinates'])
                
                used_indices.add(best_merge_idx)
                has_merged = True # Loop again to find the *next* letter

        # Save result
        merged_data.append({
            'Filename': df.iloc[i]['Filename'],
            'Detected Text': current_text,
            'Confidence': f"{current_conf:.4f}",
            'Box Coordinates': current_box
        })
        used_indices.add(i)

    return pd.DataFrame(merged_data)

def main():
    print("🔗 Loading CSV...")
    if not os.path.exists(INPUT_CSV):
        print("❌ Error: map_text_results.csv not found.")
        return

    full_df = pd.read_csv(INPUT_CSV)
    all_merged_dfs = []
    unique_files = full_df['Filename'].unique()
    
    print(f"🔄 Linking text in {len(unique_files)} maps...")
    
    for idx, filename in enumerate(unique_files):
        print(f"   [{idx+1}/{len(unique_files)}] Processing {filename}...", end="\r")
        
        # Get data for just this map
        map_df = full_df[full_df['Filename'] == filename].copy()
        
        # Run merging algorithm
        linked_df = process_map_data(map_df)
        all_merged_dfs.append(linked_df)

    # Save to new CSV
    final_df = pd.concat(all_merged_dfs)
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n\n✅ Done! Saved to: {OUTPUT_CSV}")
    print(f"📊 Rows Reduced: {len(full_df)} -> {len(final_df)} ({(len(full_df)-len(final_df))} merges)")

if __name__ == "__main__":
    main()