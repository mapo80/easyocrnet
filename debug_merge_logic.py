"""
Debug merge logic - simulate Python grouping on first line boxes
"""
import numpy as np

# Raw detections from line 1 (y ~ 16-26)
boxes_line1 = [
    {'coords': (0, 16, 14, 26), 'name': 'Box2-O'},
    {'coords': (14, 16, 36, 24), 'name': 'Box3-il'},
    {'coords': (42, 16, 72, 24), 'name': 'Box4-AW'},
    {'coords': (282, 16, 292, 24), 'name': 'Box5-num'},
    {'coords': (320, 16, 344, 24), 'name': 'Box6-num2'},
]

# Convert to horizontal_list format: [xMin, xMax, yMin, yMax, yCenter, height]
horizontal_list = []
for box in boxes_line1:
    xmin, ymin, xmax, ymax = box['coords']
    ycenter = 0.5 * (ymin + ymax)
    height = ymax - ymin
    horizontal_list.append({
        'data': [xmin, xmax, ymin, ymax, ycenter, height],
        'name': box['name']
    })

print("=== HORIZONTAL LIST (sorted by xMin) ===")
horizontal_list_sorted = sorted(horizontal_list, key=lambda x: x['data'][0])
for item in horizontal_list_sorted:
    d = item['data']
    print(f"{item['name']}: xMin={d[0]}, xMax={d[1]}, yMin={d[2]}, yMax={d[3]}, height={d[5]}")

# Simulate merge algorithm
print("\n=== MERGE SIMULATION ===")
height_ths = 0.5
width_ths = 1.0

merged_box = []
new_box = []
b_height = []
x_max = 0

for item in horizontal_list_sorted:
    box = item['data']
    xmin, xmax, ymin, ymax, ycenter, height = box

    if len(new_box) == 0:
        print(f"\n1. Starting new group with {item['name']}")
        b_height = [height]
        x_max = xmax
        new_box.append(item)
    else:
        avg_height = np.mean(b_height)
        height_diff = abs(avg_height - height)
        distance = xmin - x_max

        # Merge conditions
        height_condition = height_diff < height_ths * avg_height
        width_condition = distance < width_ths * (ymax - ymin)  # Note: uses current box height!

        print(f"\n{len(new_box)+1}. Checking {item['name']}:")
        print(f"   Height: {height:.1f} vs avg {avg_height:.1f}, diff={height_diff:.1f}, threshold={height_ths * avg_height:.1f} → {height_condition}")
        print(f"   Distance: {distance} vs threshold={width_ths * (ymax - ymin):.1f} → {width_condition}")

        if height_condition and width_condition:
            print(f"   ✓ MERGE {item['name']} into current group")
            b_height.append(height)
            x_max = xmax
            new_box.append(item)
        else:
            print(f"   ✗ START NEW GROUP with {item['name']}")
            merged_box.append(new_box)
            print(f"   → Finalized group: {[b['name'] for b in new_box]}")
            b_height = [height]
            x_max = xmax
            new_box = [item]

if len(new_box) > 0:
    merged_box.append(new_box)
    print(f"\n   → Finalized last group: {[b['name'] for b in new_box]}")

print(f"\n=== FINAL MERGED GROUPS: {len(merged_box)} ===")
for i, group in enumerate(merged_box):
    names = [b['name'] for b in group]
    print(f"Group {i+1}: {names}")
