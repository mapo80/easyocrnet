"""
Analyze parity between C# and Python OCR results with tolerance
"""
import sys

def parse_bbox(bbox_str):
    """Parse bbox string: (x1,y1) (x2,y2) (x3,y3) (x4,y4)"""
    points = []
    parts = bbox_str.split('(')[1:]  # Skip first empty part
    for part in parts:
        coords = part.split(')')[0].split(',')
        if len(coords) == 2:
            points.append((int(coords[0]), int(coords[1])))
    return points

def bbox_to_rect(points):
    """Convert 4 points to rectangle (xmin, ymin, xmax, ymax)"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))

def boxes_match(bbox1, bbox2, tolerance=5):
    """Check if two bboxes match within tolerance"""
    rect1 = bbox_to_rect(bbox1)
    rect2 = bbox_to_rect(bbox2)

    # Check if all coordinates are within tolerance
    for c1, c2 in zip(rect1, rect2):
        if abs(c1 - c2) > tolerance:
            return False
    return True

def compute_iou(bbox1, bbox2):
    """Compute Intersection over Union"""
    rect1 = bbox_to_rect(bbox1)
    rect2 = bbox_to_rect(bbox2)

    # Intersection
    x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
    y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
    intersection = x_overlap * y_overlap

    # Union
    area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def parse_line(line):
    """Parse OCR result line"""
    parts = line.strip().split('|')
    if len(parts) >= 3:
        bbox_str = parts[0].strip()
        text = parts[1].strip()
        conf_str = parts[2].strip().replace(',', '.')  # Handle both , and . as decimal
        conf = float(conf_str)
        bbox = parse_bbox(bbox_str)
        return bbox, text, conf
    return None, None, None

# Load results
csharp_file = 'dataset/base/HAL.2015.page_42.pdf_125176.png.ocr.csharp.txt'
python_file = 'dataset/base/HAL.2015.page_42.pdf_125176.png.ocr.python.txt'

with open(csharp_file, 'r') as f:
    csharp_lines = f.readlines()

with open(python_file, 'r') as f:
    python_lines = f.readlines()

print("=" * 80)
print(f"C# results: {len(csharp_lines)}")
print(f"Python results: {len(python_lines)}")
print("=" * 80)

# Try to match with different strategies
print("\n### STRATEGY 1: Order-based matching ###")
order_matches = 0
for i in range(min(len(csharp_lines), len(python_lines))):
    cs_bbox, cs_text, cs_conf = parse_line(csharp_lines[i])
    py_bbox, py_text, py_conf = parse_line(python_lines[i])

    if cs_bbox and py_bbox:
        iou = compute_iou(cs_bbox, py_bbox)
        match_5px = boxes_match(cs_bbox, py_bbox, tolerance=5)

        if match_5px:
            order_matches += 1
            print(f"{i+1}. ✓ MATCH (IoU={iou:.2f}): {py_text}")
        else:
            print(f"{i+1}. ✗ DIFF (IoU={iou:.2f}):")
            print(f"   C#: {cs_text} {bbox_to_rect(cs_bbox)}")
            print(f"   Py: {py_text} {bbox_to_rect(py_bbox)}")

print(f"\nOrder-based accuracy: {order_matches}/{min(len(csharp_lines), len(python_lines))} = {order_matches/min(len(csharp_lines), len(python_lines))*100:.1f}%")

# Try IoU-based best matching
print("\n### STRATEGY 2: IoU-based best matching ###")
csharp_results = [parse_line(line) for line in csharp_lines]
python_results = [parse_line(line) for line in python_lines]

matched_cs = set()
matched_py = set()
matches = []

# For each Python result, find best C# match
for py_idx, (py_bbox, py_text, py_conf) in enumerate(python_results):
    if not py_bbox:
        continue

    best_iou = 0
    best_cs_idx = -1

    for cs_idx, (cs_bbox, cs_text, cs_conf) in enumerate(csharp_results):
        if not cs_bbox or cs_idx in matched_cs:
            continue

        iou = compute_iou(cs_bbox, py_bbox)
        if iou > best_iou:
            best_iou = iou
            best_cs_idx = cs_idx

    if best_iou > 0.5:  # IoU threshold
        matches.append((py_idx, best_cs_idx, best_iou))
        matched_cs.add(best_cs_idx)
        matched_py.add(py_idx)

print(f"Matched pairs: {len(matches)}/{len(python_results)}")

for py_idx, cs_idx, iou in matches:
    py_bbox, py_text, py_conf = python_results[py_idx]
    cs_bbox, cs_text, cs_conf = csharp_results[cs_idx]

    rect_match = boxes_match(cs_bbox, py_bbox, tolerance=5)
    text_match = py_text == cs_text

    status = "✓" if rect_match and text_match else "~" if rect_match else "✗"
    print(f"{status} IoU={iou:.2f} | Py[{py_idx}] <-> C#[{cs_idx}]")
    if not text_match:
        print(f"  Text diff: '{cs_text}' vs '{py_text}'")

print(f"\nIoU-based accuracy: {len(matches)}/{len(python_results)} = {len(matches)/len(python_results)*100:.1f}%")

# Unmatched boxes
print("\n### UNMATCHED BOXES ###")
print(f"Unmatched C# boxes: {len(csharp_results) - len(matched_cs)}")
for cs_idx, (cs_bbox, cs_text, cs_conf) in enumerate(csharp_results):
    if cs_idx not in matched_cs and cs_bbox:
        print(f"  C#[{cs_idx}]: {cs_text} {bbox_to_rect(cs_bbox)}")

print(f"Unmatched Python boxes: {len(python_results) - len(matched_py)}")
for py_idx, (py_bbox, py_text, py_conf) in enumerate(python_results):
    if py_idx not in matched_py and py_bbox:
        print(f"  Py[{py_idx}]: {py_text} {bbox_to_rect(py_bbox)}")
