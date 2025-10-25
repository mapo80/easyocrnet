"""
Debug script to analyze grouping differences between Python and C#
"""
import sys
sys.path.insert(0, 'torchfree_ocr')

from pathlib import Path
from torchfree_ocr import Reader
import craft_utils

# Load image
image_path = Path('dataset/base/HAL.2015.page_42.pdf_125176.png')

# Initialize reader
reader = Reader(['en'])

# Get raw detections
print("=== RAW DETECTIONS (before grouping) ===")
img, img_cv_grey = reader.detector.load_image(str(image_path))
det_output, ratio = reader.detector.detector_preprocess(img, text_threshold=0.7,
                                                         link_threshold=0.4,
                                                         low_text=0.4,
                                                         poly=False)

raw_bboxes = reader.detector.detector_postprocess(det_output, ratio)
print(f"Raw detections: {len(raw_bboxes)}")
for i, bbox in enumerate(raw_bboxes[:5]):
    print(f"{i+1}. {bbox}")

# Apply grouping
print("\n=== GROUPING (group_text_box) ===")
print(f"Config: slope_ths={0.1}, ycenter_ths={0.5}, height_ths={0.5}, width_ths={0.5}, add_margin={0.1}")

horizontal_list, free_list = craft_utils.group_text_box(
    raw_bboxes,
    slope_ths=0.1,
    ycenter_ths=0.5,
    height_ths=0.5,
    width_ths=0.5,
    width_ths_for_merge=1.0,  # This is critical!
    add_margin=0.1
)

print(f"\nHorizontal list: {len(horizontal_list)}")
for i, region in enumerate(horizontal_list):
    print(f"{i+1}. {region}")

print(f"\nFree list: {len(free_list)}")
for i, poly in enumerate(free_list):
    print(f"{i+1}. {poly}")

print(f"\nTotal grouped: {len(horizontal_list) + len(free_list)}")

# Now test recognition
print("\n=== FULL PIPELINE (with grouping) ===")
results = reader.readtext(str(image_path), detail=1)
print(f"Final results: {len(results)}")
for i, (bbox, text, conf) in enumerate(results[:5]):
    # Convert bbox to string format
    bbox_str = ' '.join(f"({int(p[0])},{int(p[1])})" for p in bbox)
    print(f"{i+1}. {bbox_str} | {text} | {conf:.4f}")
