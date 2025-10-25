"""
Debug recognition by comparing crop extraction
"""
import cv2
import numpy as np
from pathlib import Path

# Load image
img = cv2.imread('dataset/base/HAL.2015.page_42.pdf_125176.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Box 1 coordinates
# C#:     (0,13) (37,13) (37,27) (0,27)
# Python: (0,15) (37,15) (37,27) (0,27)

# Extract Python crop
py_bbox = [(0,15), (37,15), (37,27), (0,27)]
py_pts = np.array(py_bbox, dtype=np.int32)
py_x_min, py_x_max = py_pts[:, 0].min(), py_pts[:, 0].max()
py_y_min, py_y_max = py_pts[:, 1].min(), py_pts[:, 1].max()
py_crop = img_gray[py_y_min:py_y_max, py_x_min:py_x_max]

print(f"Python crop: ({py_x_min},{py_y_min},{py_x_max},{py_y_max})")
print(f"Python crop size: {py_crop.shape}")
print(f"Python crop pixels (first row):\n{py_crop[0, :10]}")

# Extract C# crop
cs_bbox = [(0,13), (37,13), (37,27), (0,27)]
cs_pts = np.array(cs_bbox, dtype=np.int32)
cs_x_min, cs_x_max = cs_pts[:, 0].min(), cs_pts[:, 0].max()
cs_y_min, cs_y_max = cs_pts[:, 1].min(), cs_pts[:, 1].max()
cs_crop = img_gray[cs_y_min:cs_y_max, cs_x_min:cs_x_max]

print(f"\nC# crop: ({cs_x_min},{cs_y_min},{cs_x_max},{cs_y_max})")
print(f"C# crop size: {cs_crop.shape}")
print(f"C# crop pixels (first row):\n{cs_crop[0, :10]}")

# Save crops for visual inspection
cv2.imwrite('debug_crop_python.png', py_crop)
cv2.imwrite('debug_crop_csharp.png', cs_crop)
print("\nCrops saved to debug_crop_*.png")

# Calculate difference
print(f"\nCoordinate difference: yMin delta = {cs_y_min - py_y_min} pixels")

# Check if crops are identical
if py_crop.shape == cs_crop.shape:
    diff = np.abs(py_crop.astype(int) - cs_crop.astype(int))
    print(f"Crops identical? {np.all(diff == 0)}")
    if not np.all(diff == 0):
        print(f"Max pixel difference: {diff.max()}")
        print(f"Mean pixel difference: {diff.mean():.2f}")
else:
    print("Crops have different shapes!")

# Now test recognition preprocessing
print("\n=== TESTING RECOGNITION PREPROCESSING ===")

# Python preprocessing (from ocr_process.py)
def python_preprocess(crop_gray, imgH=64, imgW=200):
    """Python recognition preprocessing"""
    h, w = crop_gray.shape
    ratio = w / h

    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = math.ceil(imgH * ratio)

    resized = cv2.resize(crop_gray, (resized_w, imgH), interpolation=cv2.INTER_LINEAR)

    # Normalize to [-1, 1]
    normalized = (resized.astype('float32') / 255.0 - 0.5) / 0.5

    # Pad with last column
    if resized_w < imgW:
        padded = np.zeros((imgH, imgW), dtype='float32')
        padded[:, :resized_w] = normalized
        for col in range(resized_w, imgW):
            padded[:, col] = normalized[:, -1]
        return padded
    else:
        return normalized

import math
py_preprocessed = python_preprocess(py_crop)
cs_preprocessed = python_preprocess(cs_crop)

print(f"Python preprocessed shape: {py_preprocessed.shape}")
print(f"C# preprocessed shape: {cs_preprocessed.shape}")
print(f"Python preprocessed range: [{py_preprocessed.min():.3f}, {py_preprocessed.max():.3f}]")
print(f"C# preprocessed range: [{cs_preprocessed.min():.3f}, {cs_preprocessed.max():.3f}]")

# Calculate preprocessing difference
diff_preproc = np.abs(py_preprocessed - cs_preprocessed)
print(f"Preprocessing difference - max: {diff_preproc.max():.6f}, mean: {diff_preproc.mean():.6f}")
