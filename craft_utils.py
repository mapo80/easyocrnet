"""
CRAFT (Character Region Awareness For Text detection) utilities.
Extracted from easyocr library - MIT License.
"""
import numpy as np
import cv2
import math
from scipy.ndimage import label


def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text, estimate_num_chars=False):
    """Extract bounding boxes from CRAFT text and link score maps."""
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    # Labeling method
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4
    )

    det = []
    mapper = []
    for k in range(1, nLabels):
        # Size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # Thresholding
        if np.max(textmap[labels == k]) < text_threshold:
            continue

        # Make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255

        if estimate_num_chars:
            _, character_locs = cv2.threshold(
                (textmap - linkmap) * segmap / 255., text_threshold, 1, 0
            )
            _, n_chars = label(character_locs)
            mapper.append(n_chars)
        else:
            mapper.append(k)

        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # Remove link area

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1

        # Boundary check
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # Make box
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # Align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # Make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)

    return det, labels, mapper


def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False, estimate_num_chars=False):
    """
    Get detection boxes from CRAFT score maps.

    Args:
        textmap: Text region score map
        linkmap: Link score map
        text_threshold: Text confidence threshold
        link_threshold: Link confidence threshold
        low_text: Low text threshold
        poly: Return polygons instead of boxes
        estimate_num_chars: Estimate number of characters

    Returns:
        boxes, polys, mapper
    """
    boxes, labels, mapper = getDetBoxes_core(
        textmap, linkmap, text_threshold, link_threshold, low_text, estimate_num_chars
    )

    if poly:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys, mapper


def getPoly_core(boxes, labels, mapper, linkmap):
    """Get polygon representation of detected boxes."""
    # This is a simplified version - full implementation is complex
    # For our use case, we don't need poly mode
    return [None] * len(boxes)


def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    """
    Adjust bounding box coordinates by scaling ratios.

    Args:
        polys: List of polygons/boxes
        ratio_w: Width scaling ratio
        ratio_h: Height scaling ratio
        ratio_net: Network ratio (default 2 for CRAFT)

    Returns:
        Adjusted polygons
    """
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


def normalizeMeanVariance(img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    """
    Normalize image with ImageNet mean and variance.

    Args:
        img: Input image (RGB format)
        mean: Mean values for normalization
        variance: Std values for normalization

    Returns:
        Normalized image
    """
    img = img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1.0):
    """
    Resize image while preserving aspect ratio and pad to multiples of 32.

    Args:
        img: Input image
        square_size: Maximum size
        interpolation: OpenCV interpolation method
        mag_ratio: Magnification ratio

    Returns:
        resized_img, ratio, size_heatmap
    """
    height, width, channel = img.shape

    # Magnify image size
    target_size = mag_ratio * max(height, width)

    # Set original image size
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # Make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)

    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap


def group_text_box(polys, slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5,
                    width_ths=1.0, add_margin=0.05, sort_output=True):
    """
    Group and merge text boxes based on their position and size.

    Args:
        polys: List of polygons (flattened format)
        slope_ths: Slope threshold for horizontal classification
        ycenter_ths: Y-center threshold for grouping
        height_ths: Height threshold for merging
        width_ths: Width threshold for merging
        add_margin: Margin to add around boxes
        sort_output: Sort by y-center

    Returns:
        horizontal_list, free_list
    """
    horizontal_list, free_list, combined_list, merged_list = [], [], [], []

    # Classify boxes as horizontal or free-form based on slope
    for poly in polys:
        slope_up = (poly[3] - poly[1]) / np.maximum(10, (poly[2] - poly[0]))
        slope_down = (poly[5] - poly[7]) / np.maximum(10, (poly[4] - poly[6]))

        if max(abs(slope_up), abs(slope_down)) < slope_ths:
            x_max = max([poly[0], poly[2], poly[4], poly[6]])
            x_min = min([poly[0], poly[2], poly[4], poly[6]])
            y_max = max([poly[1], poly[3], poly[5], poly[7]])
            y_min = min([poly[1], poly[3], poly[5], poly[7]])
            horizontal_list.append([x_min, x_max, y_min, y_max, 0.5 * (y_min + y_max), y_max - y_min])
        else:
            height = np.linalg.norm([poly[6] - poly[0], poly[7] - poly[1]])
            width = np.linalg.norm([poly[2] - poly[0], poly[3] - poly[1]])
            margin = int(1.44 * add_margin * min(width, height))

            theta13 = abs(np.arctan((poly[1] - poly[5]) / np.maximum(10, (poly[0] - poly[4]))))
            theta24 = abs(np.arctan((poly[3] - poly[7]) / np.maximum(10, (poly[2] - poly[6]))))

            x1 = poly[0] - np.cos(theta13) * margin
            y1 = poly[1] - np.sin(theta13) * margin
            x2 = poly[2] + np.cos(theta24) * margin
            y2 = poly[3] - np.sin(theta24) * margin
            x3 = poly[4] + np.cos(theta13) * margin
            y3 = poly[5] + np.sin(theta13) * margin
            x4 = poly[6] - np.cos(theta24) * margin
            y4 = poly[7] + np.sin(theta24) * margin

            free_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    if sort_output:
        horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

    # Group boxes by line (similar y_center)
    new_box = []
    for poly in horizontal_list:
        if len(new_box) == 0:
            b_height = [poly[5]]
            b_ycenter = [poly[4]]
            new_box.append(poly)
        else:
            if abs(np.mean(b_ycenter) - poly[4]) < ycenter_ths * np.mean(b_height):
                b_height.append(poly[5])
                b_ycenter.append(poly[4])
                new_box.append(poly)
            else:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                combined_list.append(new_box)
                new_box = [poly]
    combined_list.append(new_box)

    # Merge boxes on the same line
    for boxes in combined_list:
        if len(boxes) == 1:
            box = boxes[0]
            margin = int(add_margin * min(box[1] - box[0], box[5]))
            merged_list.append([box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin])
        else:
            boxes = sorted(boxes, key=lambda item: item[0])

            merged_box, new_box = [], []
            for box in boxes:
                if len(new_box) == 0:
                    b_height = [box[5]]
                    x_max = box[1]
                    new_box.append(box)
                else:
                    if (abs(np.mean(b_height) - box[5]) < height_ths * np.mean(b_height)) and \
                       ((box[0] - x_max) < width_ths * (box[3] - box[2])):
                        b_height.append(box[5])
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        b_height = [box[5]]
                        x_max = box[1]
                        merged_box.append(new_box)
                        new_box = [box]
            if len(new_box) > 0:
                merged_box.append(new_box)

            for mbox in merged_box:
                if len(mbox) != 1:  # Adjacent boxes in same line
                    x_min = min(mbox, key=lambda x: x[0])[0]
                    x_max = max(mbox, key=lambda x: x[1])[1]
                    y_min = min(mbox, key=lambda x: x[2])[2]
                    y_max = max(mbox, key=lambda x: x[3])[3]

                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    margin = int(add_margin * (min(box_width, box_height)))

                    merged_list.append([x_min - margin, x_max + margin, y_min - margin, y_max + margin])
                else:  # Non-adjacent box in same line
                    box = mbox[0]
                    box_width = box[1] - box[0]
                    box_height = box[3] - box[2]
                    margin = int(add_margin * (min(box_width, box_height)))

                    merged_list.append([box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin])

    return merged_list, free_list
