import os
import urllib.request
import tarfile
import shutil

# --- CONFIGURATION ---
# We use a distinct folder for the code to keep it clean
REPO_DIR = "PaddleOCR_Engine"
MODEL_DIR = "PaddleOCR/pretrain_models"
URL_MODEL = "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar"

def setup_environment():
    # 1. Clone PaddleOCR Source Code (if not exists)
    if not os.path.exists(REPO_DIR):
        print(f"⬇️ Cloning PaddleOCR Repository into {REPO_DIR}...")
        # We use git command here
        os.system(f"git clone https://github.com/PaddlePaddle/PaddleOCR.git {REPO_DIR}")
    else:
        print(f"✅ {REPO_DIR} already exists.")

    # 2. Create Model Directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 3. Download Pre-trained Weights
    tar_name = os.path.basename(URL_MODEL)
    tar_path = os.path.join(MODEL_DIR, tar_name)
    
    if not os.path.exists(tar_path):
        print(f"⬇️ Downloading Pre-trained Model ({tar_name})...")
        urllib.request.urlretrieve(URL_MODEL, tar_path)
        print("   Download complete.")
        
        # Extract
        print("📦 Extracting model...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=MODEL_DIR)
        print("✅ Model extracted.")
    else:
        print("✅ Pre-trained model already downloaded.")

    # 4. Install Training Dependencies
    print("🔧 Installing dependencies for training...")
    os.system(f"pip install -r {REPO_DIR}/requirements.txt")
    
    print("-" * 30)
    print("🚀 Training Setup Complete!")
    print(f"1. Code: {REPO_DIR}/")
    print(f"2. Model: {MODEL_DIR}/en_PP-OCRv4_rec_train/")
    print("-" * 30)

if __name__ == "__main__":
    setup_environment()