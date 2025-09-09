

import cv2
import numpy as np
from PIL import Image
import os
import zipfile

# ========= CONFIG =========
INPUT_IMAGE = "/home/inzamul/Desktop/data_image.jpg"   # <-- replace with your file name
OUTPUT_DIR = "digits_28x28"
ZIP_NAME = "digits_28x28.zip"
# ==========================

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load image in grayscale
img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)

# Threshold (black digits on white background)
_, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

# Find contours (digits)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours row by row, left to right
contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1]//50, cv2.boundingRect(c)[0]))

idx = 0
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    
    # Skip tiny noise
    if w < 5 or h < 5:
        continue
    
    digit = thresh[y:y+h, x:x+w]

    # Resize digit to 20x20
    digit_resized = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    
    # Pad to 28x28
    padded = np.pad(digit_resized, ((4,4),(4,4)), "constant", constant_values=0)
    
    # Save each digit
    out_path = os.path.join(OUTPUT_DIR, f"digit_{idx:03d}.jpg")
    Image.fromarray(padded).save(out_path)
    idx += 1

# Create ZIP file
with zipfile.ZipFile(ZIP_NAME, 'w') as zipf:
    for file in os.listdir(OUTPUT_DIR):
        zipf.write(os.path.join(OUTPUT_DIR, file), file)

print(f"✅ Done! Extracted {idx} digits into {OUTPUT_DIR} and zipped into {ZIP_NAME}")

