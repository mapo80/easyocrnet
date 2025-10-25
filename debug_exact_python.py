"""
Compare Python and C# OCR results EXACTLY using actual Python code
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Import local craft_utils
sys.path.insert(0, '.')
from craft_utils import group_text_box
from ocr_process import detector_preprocess, detector_postprocess
import onnxruntime as ort

# Load image
image_path = Path('dataset/base/HAL.2015.page_42.pdf_125176.png')
img = cv2.imread(str(image_path))

# Step 1: Detection
det_session = ort.InferenceSession('models/cpu/detection.onnx', providers=['CPUExecutionProvider'])
det_input, ratio, size_heatmap = detector_preprocess(img)
det_output = det_session.run(None, {det_session.get_inputs()[0].name: det_input})[0]
bboxes = detector_postprocess(det_output, ratio)

print(f"=== RAW DETECTIONS: {len(bboxes)} ===")
for i, bbox in enumerate(bboxes):
    pts = np.array(bbox).astype(np.int32)
    x_min, x_max = int(pts[:, 0].min()), int(pts[:, 0].max())
    y_min, y_max = int(pts[:, 1].min()), int(pts[:, 1].max())
    print(f"{i+1}. ({x_min},{y_min},{x_max},{y_max})")

# Step 2: Group text boxes
polys = [np.array(bbox).astype(np.int32).reshape((-1)) for bbox in bboxes]
horizontal_list, free_list = group_text_box(
    polys,
    slope_ths=0.1,
    ycenter_ths=0.5,
    height_ths=0.5,
    width_ths=0.5,  # CRITICAL: This is what readtext() uses!
    add_margin=0.1
)

print(f"\n=== AFTER GROUPING (before min_size filter): {len(horizontal_list) + len(free_list)} ===")
print(f"Horizontal boxes: {len(horizontal_list)}")
for i, region in enumerate(horizontal_list):
    print(f"{i+1}. xMin={region[0]}, xMax={region[1]}, yMin={region[2]}, yMax={region[3]}")

print(f"\nFree-form boxes: {len(free_list)}")
for i, poly in enumerate(free_list):
    pts = np.array(poly).astype(np.int32)
    x_min, x_max = int(pts[:, 0].min()), int(pts[:, 0].max())
    y_min, y_max = int(pts[:, 1].min()), int(pts[:, 1].max())
    print(f"{i+1}. ({x_min},{y_min},{x_max},{y_max})")

# Step 3: Apply min_size filter
min_size = 20
horizontal_list_filtered = [box for box in horizontal_list
                            if max(box[1] - box[0], box[3] - box[2]) > min_size]
free_list_filtered = [box for box in free_list
                      if max(np.max([c[0] for c in box]) - np.min([c[0] for c in box]),
                             np.max([c[1] for c in box]) - np.min([c[1] for c in box])) > min_size]

print(f"\n=== AFTER MIN_SIZE FILTER: {len(horizontal_list_filtered) + len(free_list_filtered)} ===")
print(f"Horizontal boxes: {len(horizontal_list_filtered)}")
for i, region in enumerate(horizontal_list_filtered):
    print(f"{i+1}. ({region[0]},{region[2]}) ({region[1]},{region[2]}) ({region[1]},{region[3]}) ({region[0]},{region[3]})")

print(f"\nFree-form boxes: {len(free_list_filtered)}")
for i, poly in enumerate(free_list_filtered):
    pts = np.array(poly).astype(np.int32)
    bbox_str = ' '.join(f"({int(p[0])},{int(p[1])})" for p in poly)
    print(f"{i+1}. {bbox_str}")

print(f"\n=== TOTAL FINAL BOXES: {len(horizontal_list_filtered) + len(free_list_filtered)} ===")

# Now read C# output and compare
print("\n" + "="*80)
print("=== C# OUTPUT ===")
with open('dataset/base/HAL.2015.page_42.pdf_125176.png.ocr.csharp.txt', 'r') as f:
    csharp_lines = f.readlines()

print(f"C# has {len(csharp_lines)} boxes")
for i, line in enumerate(csharp_lines):
    bbox_str = line.split('|')[0].strip()
    text = line.split('|')[1].strip()
    print(f"{i+1}. {bbox_str} | {text}")
