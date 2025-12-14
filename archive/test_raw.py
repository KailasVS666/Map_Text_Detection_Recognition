import os
import sys
import cv2
import numpy as np
import yaml
import paddle

sys.path.append(os.path.abspath("PaddleOCR_Engine"))

from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model

# --- CONFIGURATION ---
CONFIG_PATH = "my_config.yml"
MODEL_PATH = "PaddleOCR/output/rec_finetune/best_accuracy" 
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "PaddleOCR/output/rec_finetune/latest"

IMAGE_DIR = "PaddleOCR/train_data/rec/aug_crops"

def init_model():
    print(f"📖 Loading Config: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    dict_path = config['Global']['character_dict_path']
    with open(dict_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        char_num = len(lines)
    if config['Global']['use_space_char']: char_num += 1
    class_num = char_num + 1 
    
    config['Architecture']['Head']['out_channels_list'] = {'CTCLabelDecode': class_num}

    print(f"🏗️ Building Model...")
    model = build_model(config['Architecture'])
    
    print(f"⚖️ Loading Weights from: {MODEL_PATH}")
    load_model(config, model, model_type=config['Architecture']['model_type'])
    model.eval()
    
    post_process_class = build_post_process(config['PostProcess'], config['Global'])
    return model, post_process_class

def predict_image(model, post_processor, image_path):
    if not os.path.exists(image_path): return

    print(f"\n📄 Testing: {os.path.basename(image_path)}")
    img = cv2.imread(image_path)
    if img is None: return

    # --- 1. Preprocessing (The Winner: Scale 0-255) ---
    h, w = 48, 320 
    img = cv2.resize(img, (w, h))
    # Correct shape: (C, H, W). NO division by 255.
    img = img.transpose((2, 0, 1)).astype('float32')
    img = img[np.newaxis, :] 
    tensor_img = paddle.to_tensor(img)

    # --- 2. Inference ---
    preds = model(tensor_img)
    if isinstance(preds, dict): preds = preds['ctc']
    preds = paddle.nn.functional.softmax(preds)
    res = post_processor(preds.numpy()) 

    # --- 3. Safety Mode Output ---
    # We print the raw structure to debug the "too many values" error
    if len(res) > 0:
        result_item = res[0]
        # Check if it's a tuple or list and print cleanly
        if isinstance(result_item, (list, tuple)):
            print(f"   📦 Raw Item: {result_item}")
            if len(result_item) >= 1:
                print(f"   ✨ Text: '{result_item[0]}'")
            if len(result_item) >= 2:
                print(f"   📊 Score: {result_item[1]}")
        else:
            print(f"   📦 Unknown Format: {result_item}")
    else:
        print("   ⚠️ No output decoded.")

if __name__ == "__main__":
    model, post_processor = init_model()
    
    print("-" * 40)
    print("🚀 Final Test Run (Raw Scaling)")
    print("-" * 40)

    for i in range(10):
        img_name = f"orig_{i}.jpg"
        full_path = os.path.join(IMAGE_DIR, img_name)
        if os.path.exists(full_path):
            predict_image(model, post_processor, full_path)