import os
import cv2
import sys

# --- CONFIGURATION ---
IMAGE_FOLDER = "training_hard_examples"
LABEL_FILE = "training_hard_examples/labels.txt"

def main():
    print(f"🔹 Starting Labeling Session for: {IMAGE_FOLDER}")
    
    # 1. Load existing labels (to support resuming)
    existing_labels = {}
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    existing_labels[parts[0]] = parts[1]
        print(f"✅ Loaded {len(existing_labels)} existing labels. Resuming...")

    # 2. Get list of images
    try:
        images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png'))]
    except FileNotFoundError:
        print(f"❌ Error: Folder '{IMAGE_FOLDER}' not found.")
        return

    # Filter out already labeled images
    to_process = [img for img in images if img not in existing_labels]
    
    if not to_process:
        print("🎉 All images are already labeled! You are done.")
        return

    print(f"📝 You have {len(to_process)} images left to label.")
    print("-------------------------------------------------")
    print("👉 INSTRUCTIONS:")
    print("   1. Look at the popup window.")
    print("   2. Type the text exactly as you see it in the TERMINAL.")
    print("   3. Press ENTER to save and go to next.")
    print("   4. Type 'del' to delete the image if it's actually garbage.")
    print("   5. Type 'exit' to stop and save progress.")
    print("-------------------------------------------------")

    with open(LABEL_FILE, "a", encoding="utf-8") as f:
        for i, filename in enumerate(to_process):
            img_path = os.path.join(IMAGE_FOLDER, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"⚠️ Could not load {filename}, skipping.")
                continue

            # Show the image (Enlarged 2x for easier reading)
            display_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Labeling - Check Terminal to Type", display_img)
            cv2.waitKey(1) # Force window update

            # Get user input
            print(f"[{i+1}/{len(to_process)}] Text for {filename}: ", end="")
            user_input = input().strip()

            if user_input.lower() == 'exit':
                print("💾 Progress saved. Exiting...")
                break
            
            if user_input.lower() == 'del':
                print(f"🗑️ Deleting {filename}...")
                # Close image window before deleting to avoid lock issues on Windows
                # (Optional safety, though Python usually handles it)
                try:
                    os.remove(img_path)
                except PermissionError:
                    print("⚠️ File locked. Could not delete immediately.")
                continue

            # Save valid label
            if user_input:
                f.write(f"{filename}\t{user_input}\n")
                f.flush() # Ensure it writes to disk immediately
            else:
                print("⚠️ Empty input. Skipping image (not saved).")

    cv2.destroyAllWindows()
    print("\n✅ Session Finished.")

if __name__ == "__main__":
    main()