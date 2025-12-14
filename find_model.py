import os

# Start searching from the current directory
search_root = "." 
print(f"🔎 Searching for '.pdparams' or '.pdmodel' files in: {os.path.abspath(search_root)}")

found = False
for root, dirs, files in os.walk(search_root):
    # Look for model files
    for file in files:
        if file.endswith("best_accuracy.pdparams") or file.endswith("inference.pdmodel"):
            print(f"\n✅ FOUND MODEL FILE:")
            print(f"   📂 Path: {root}")
            print(f"   📄 File: {file}")
            found = True

if not found:
    print("\n❌ No model files found in this folder or subfolders.")