import pandas as pd
import cv2
import os
import random

# --- CONFIGURATION ---
CSV_FILE = "map_text_results.csv"
IMAGE_FOLDER = r"C:\Users\sharj\Desktop\Rumsey_Map_OCR\Rumsey_Map_OCR_Data\rumsey\icdar24-train-png\train_images"
OUTPUT_FOLDER = "visualized_results"

# Create output folder
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def main():
    print("📊 Loading CSV data...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print("❌ CSV file not found! Run batch_process_maps.py first.")
        return

    # Pick 3 random maps to visualize
    unique_files = df['Filename'].unique().tolist()
    sample_files = random.sample(unique_files, 3)

    print(f"🖌️  Visualizing results for: {sample_files}")

    for filename in sample_files:
        img_path = os.path.join(IMAGE_FOLDER, filename)
        
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {filename}")
            continue

        # Load Image
        img = cv2.imread(img_path)
        
        # Get all text for this image
        map_data = df[df['Filename'] == filename]

        for index, row in map_data.iterrows():
            text = row['Detected Text']
            score = float(row['Confidence'])
            box = eval(row['Box Coordinates']) # Convert string back to list

            # Draw Box (Green)
            points = [(int(pt[0]), int(pt[1])) for pt in box]
            for i in range(4):
                cv2.line(img, points[i], points[(i+1)%4], (0, 255, 0), 2)

            # Draw Text (Red)
            # Put text slightly above the box
            cv2.putText(img, f"{text} ({score:.2f})", (points[0][0], points[0][1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save Result
        save_path = os.path.join(OUTPUT_FOLDER, f"viz_{filename}")
        cv2.imwrite(save_path, img)
        print(f"✅ Saved visualization: {save_path}")

    print(f"\n✨ Done! Check the '{OUTPUT_FOLDER}' folder to see your AI in action.")

if __name__ == "__main__":
    main()