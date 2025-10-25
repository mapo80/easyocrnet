"""
Debug Henry Hub merge issue
"""
import numpy as np

# Simulate grouping for line 3 (y~46-56)
# Raw boxes from detection
raw_boxes = [
    {'coords': [0, 30, 46, 54], 'name': 'Box1'},
    {'coords': [48, 68, 46, 54], 'name': 'Box2'},
    {'coords': [76, 100, 46, 54], 'name': 'Henry-1'},  # "H" or "Hcmr"
    {'coords': [102, 122, 46, 56], 'name': 'Henry-2'},  # "Hub"
    {'coords': [32, 46, 48, 54], 'name': 'Box-mid'},
]

# Convert to horizontal_list format
horizontal_list = []
for box in raw_boxes:
    xmin, xmax, ymin, ymax = box['coords']
    ycenter = 0.5 * (ymin + ymax)
    height = ymax - ymin
    horizontal_list.append({
        'data': [xmin, xmax, ymin, ymax, ycenter, height],
        'name': box['name']
    })

# Sort by ycenter first (grouping by line), then by xmin
horizontal_list_sorted = sorted(horizontal_list, key=lambda x: (x['data'][4], x['data'][0]))

print("=== HORIZONTAL LIST (sorted by ycenter, xmin) ===")
for item in horizontal_list_sorted:
    d = item['data']
    print(f"{item['name']}: xMin={d[0]}, xMax={d[1]}, yMin={d[2]}, yMax={d[3]}, yCenter={d[4]:.1f}, height={d[5]}")

# First, group by line (ycenter threshold)
print("\n=== PHASE 1: GROUP BY LINE ===")
ycenter_ths = 0.5
combined_list = []
new_box = []
b_height = []
b_ycenter = []

for item in horizontal_list_sorted:
    box = item['data']
    xmin, xmax, ymin, ymax, ycenter, height = box

    if len(new_box) == 0:
        print(f"\nStarting new line with {item['name']} (yCenter={ycenter:.1f})")
        b_height = [height]
        b_ycenter = [ycenter]
        new_box.append(item)
    else:
        avg_ycenter = np.mean(b_ycenter)
        avg_height = np.mean(b_height)
        ycenter_diff = abs(avg_ycenter - ycenter)
        threshold = ycenter_ths * avg_height

        print(f"\nChecking {item['name']} (yCenter={ycenter:.1f}):")
        print(f"  yCenter diff: {ycenter_diff:.1f} vs threshold {threshold:.1f}")

        if ycenter_diff < threshold:
            print(f"  ✓ SAME LINE - add to current group")
            b_height.append(height)
            b_ycenter.append(ycenter)
            new_box.append(item)
        else:
            print(f"  ✗ NEW LINE - finalize current group")
            combined_list.append(new_box)
            print(f"  → Finalized line: {[b['name'] for b in new_box]}")
            new_box = [item]
            b_height = [height]
            b_ycenter = [ycenter]

if len(new_box) > 0:
    combined_list.append(new_box)
    print(f"\n→ Finalized last line: {[b['name'] for b in new_box]}")

print(f"\n=== PHASE 2: MERGE BOXES ON SAME LINE ===")
height_ths = 0.5
width_ths = 0.5

for line_idx, boxes in enumerate(combined_list):
    print(f"\n--- Line {line_idx + 1}: {[b['name'] for b in boxes]} ---")

    # Sort by xMin
    boxes_sorted = sorted(boxes, key=lambda x: x['data'][0])

    merged_box = []
    new_box = []
    b_height = []
    x_max = 0

    for item in boxes_sorted:
        box = item['data']
        xmin, xmax, ymin, ymax, ycenter, height = box

        if len(new_box) == 0:
            print(f"\n  Starting new group with {item['name']}")
            b_height = [height]
            x_max = xmax
            new_box.append(item)
        else:
            avg_height = np.mean(b_height)
            height_diff = abs(avg_height - height)
            distance = xmin - x_max

            # Merge conditions
            height_condition = height_diff < height_ths * avg_height
            width_condition = distance < width_ths * (ymax - ymin)

            print(f"\n  Checking {item['name']}:")
            print(f"    Height: {height} vs avg {avg_height:.1f}, diff={height_diff:.1f}, threshold={height_ths * avg_height:.1f} → {height_condition}")
            print(f"    Distance: {distance} vs threshold={width_ths * (ymax - ymin):.1f} (width_ths × box.height) → {width_condition}")

            if height_condition and width_condition:
                print(f"    ✓ MERGE into current group")
                b_height.append(height)
                x_max = xmax
                new_box.append(item)
            else:
                print(f"    ✗ START NEW GROUP")
                merged_box.append(new_box)
                print(f"    → Finalized: {[b['name'] for b in new_box]}")
                new_box = [item]
                b_height = [height]
                x_max = xmax

    if len(new_box) > 0:
        merged_box.append(new_box)
        print(f"\n  → Finalized last: {[b['name'] for b in new_box]}")

    print(f"\n  RESULT: {len(merged_box)} merged groups")
    for i, group in enumerate(merged_box):
        names = [b['name'] for b in group]
        # Calculate final bbox
        min_x = min(b['data'][0] for b in group)
        max_x = max(b['data'][1] for b in group)
        min_y = min(b['data'][2] for b in group)
        max_y = max(b['data'][3] for b in group)
        print(f"    Group {i+1}: {names} → ({min_x},{min_y},{max_x},{max_y})")
