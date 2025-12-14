import sys
import os
import cv2
import csv
import glob

# 1. Add the tools folder to Python's path so we can import the working engine
TOOLS_PATH = os.path.join(os.getcwd(), "PaddleOCR_Official_Tools")
sys.path.append(TOOLS_PATH)

# 2. Import the exact system that worked in your terminal
try:
    from tools.infer.predict_system import TextSystem
    from tools.infer.utility import parse_args
except ImportError:
    print("❌ Error: Could not find PaddleOCR tools. Make sure you are in the 'Rumsey_Map_OCR' folder.")
    sys.exit(1)

def main():
    # --- CONFIGURATION ---
    IMAGE_FOLDER = r"C:\Users\sharj\Desktop\Rumsey_Map_OCR\Rumsey_Map_OCR_Data\rumsey\icdar24-train-png\train_images"
    OUTPUT_CSV = "map_text_results.csv"
    
    # Paths to your models
    DET_MODEL = "./inference/ch_PP-OCRv4_det_infer/"
    REC_MODEL = "./output/rec_inference/"
    DICT_PATH = "PaddleOCR_Official_Tools/ppocr/utils/en_dict.txt"
    
    print("🔄 Initializing the Robust Engine (TextSystem)...")
    
    # 3. Configure the engine manually (Same as command line args)
    args = parse_args()
    args.det_model_dir = DET_MODEL
    args.rec_model_dir = REC_MODEL
    args.rec_char_dict_path = DICT_PATH
    args.use_angle_cls = False
    args.use_gpu = False  # Keep CPU for maximum stability
    
    # Initialize the engine
    text_sys = TextSystem(args)
    
    # 4. Find Images
    image_files = glob.glob(os.path.join(IMAGE_FOLDER, "*.png")) + \
                  glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))
    
    print(f"📂 Found {len(image_files)} images. Starting processing...")
    
    # 5. Process and Save
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Detected Text", "Confidence", "Box Coordinates"])
        
        for index, img_file in enumerate(image_files):
            fname = os.path.basename(img_file)
            print(f"[{index+1}/{len(image_files)}] Processing {fname}...", end="\r")
            
            # Read image
            img = cv2.imread(img_file)
            if img is None:
                continue
            
            # Run inference
            try:
                # FIX: Capture all return values first
                preds = text_sys(img)
                
                # Extract only the first two (Boxes and Results)
                dt_boxes = preds[0]
                rec_res = preds[1]
                
            except Exception as e:
                print(f"\n   ❌ Error on {fname}: {e}")
                continue
            
            # Save results
            if dt_boxes is not None and rec_res is not None:
                for box, res in zip(dt_boxes, rec_res):
                    text, score = res
                    if score >= 0.85: # Confidence threshold
                        writer.writerow([fname, text, f"{score:.4f}", box.tolist()])
                        
    print(f"\n✅ Success! Results saved to: {os.path.abspath(OUTPUT_CSV)}")

if __name__ == "__main__":
    main()