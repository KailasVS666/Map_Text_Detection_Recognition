import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION ---
OUTPUT_DIR = "dataset_ready/train" 
LABEL_FILE = "dataset_ready/train/labels.csv"
NUM_IMAGES = 1000
IMG_HEIGHT = 32
IMG_WIDTH = 128

# List of common 1800s map words
WORDS = ["River", "Mountain", "County", "Bridge", "Valley", "Road", "Lake", "Village", "Creek", "Town", 
         "Island", "North", "South", "East", "West", "Saint", "Mount", "Hill", "Station", "Church"]

def add_noise(img):
    """Add salt-and-pepper noise to mimic old paper texture"""
    noise = np.random.randint(0, 40, (IMG_HEIGHT, IMG_WIDTH), dtype='uint8')
    img = cv2.subtract(img, noise) # Darker noise for ink-bleed effect
    return img

def generate_text_img(text):
    # Create white canvas
    img = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), color=255)
    draw = ImageDraw.Draw(img)
    
    # Try to use a Serif font found on Windows
    try:
        font = ImageFont.truetype("times.ttf", random.randint(18, 22))
    except:
        font = ImageFont.load_default()

    # FIX: Use textbbox instead of textsize for Pillow 10+
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    # Draw text in the center
    draw.text(((IMG_WIDTH-w)/2, (IMG_HEIGHT-h)/2 - bbox[1]), text, font=font, fill=0)
    
    img_cv = np.array(img)
    
    # Subtle Rotation
    matrix = cv2.getRotationMatrix2D((IMG_WIDTH/2, IMG_HEIGHT/2), random.uniform(-1.5, 1.5), 1)
    img_cv = cv2.warpAffine(img_cv, matrix, (IMG_WIDTH, IMG_HEIGHT), borderValue=255)
    
    return add_noise(img_cv)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"🎨 Generating {NUM_IMAGES} synthetic images...")
    
    # Check if labels.csv exists, if not create header
    file_exists = os.path.isfile(LABEL_FILE)
    
    with open(LABEL_FILE, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("filename,words\n")
            
        for i in range(NUM_IMAGES):
            word = random.choice(WORDS)
            if random.random() > 0.8:
                word += f" {random.randint(1, 50)}"
                
            filename = f"synth_{i}.jpg"
            img = generate_text_img(word)
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img)
            f.write(f"{filename},{word}\n")

    print(f"✅ Success! 1000 images added to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()