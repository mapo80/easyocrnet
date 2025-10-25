#!/usr/bin/env python3
"""Process images with OCR using ONNX models directly."""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple
import time

try:
    import cv2
    import numpy as np
    import onnxruntime as ort
    from craft_utils import (
        getDetBoxes, adjustResultCoordinates, normalizeMeanVariance,
        resize_aspect_ratio, group_text_box
    )
except ImportError as e:
    print(f"Error: {e}. Run: pip install opencv-python onnxruntime numpy scipy")
    sys.exit(1)


def detector_preprocess(img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Preprocess image for detection model with aspect ratio preservation."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized, ratio, size_heatmap = resize_aspect_ratio(
        img, square_size=2560, interpolation=cv2.INTER_LINEAR, mag_ratio=1.0
    )
    img_norm = normalizeMeanVariance(img_resized)
    img_input = np.transpose(img_norm, (2, 0, 1))
    return img_input[None, ...].astype(np.float32), ratio, size_heatmap


def detector_postprocess(score_map: np.ndarray, ratio: float) -> List[List[List[int]]]:
    """Extract bounding boxes using CRAFT post-processing."""
    score_text = score_map[0, :, :, 0]
    score_link = score_map[0, :, :, 1]

    boxes, _, _ = getDetBoxes(
        score_text, score_link,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        poly=False,
        estimate_num_chars=False
    )

    if len(boxes) == 0:
        return []

    # Scale back to original image coordinates (torchfree-ocr uses ratio_w = ratio_h = 1 / target_ratio)
    ratio_w = ratio_h = 1 / ratio
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)

    # Convert to list format
    bboxes = []
    for box in boxes:
        box = np.array(box).reshape(-1, 2)
        bboxes.append(box.tolist())

    return bboxes


def recognizer_preprocess(crop: np.ndarray, img_h: int = 64, img_w: int = None) -> np.ndarray:
    """
    Preprocess cropped region for recognition.
    Matches torchfree-ocr's AlignCollate preprocessing exactly.
    """
    # Convert to grayscale
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = crop.shape

    # Calculate resize width maintaining aspect ratio
    ratio = w / float(h)
    resized_w = img_w if int(img_h * ratio) > img_w else int(img_h * ratio)

    # Resize with LINEAR interpolation (matches torchfree-ocr exactly)
    resized = cv2.resize(crop, (resized_w, img_h), interpolation=cv2.INTER_LINEAR)

    # Normalize: convert to [0,1] then [-1,1]
    resized = resized.astype(np.float32) / 255.0
    resized = (resized - 0.5) / 0.5

    # Create padded output
    padded = np.zeros((img_h, img_w), dtype=np.float32)
    padded[:, :resized_w] = resized

    # Pad the rest with the last column repeated
    if resized_w < img_w:
        last_col = resized[:, -1:]
        padded[:, resized_w:] = np.tile(last_col, (1, img_w - resized_w))

    # Add batch and channel dimensions: (1, 1, H, W)
    return padded[None, None, :, :].astype(np.float32)


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def custom_mean(x):
    """Calculate confidence score like torchfree-ocr."""
    return x.prod()**(2.0/np.sqrt(len(x)))


def contrast_grey(img):
    """Calculate contrast of grayscale image."""
    high = np.percentile(img, 90)
    low = np.percentile(img, 10)
    return (high-low)/np.maximum(10, high+low), high, low


def adjust_contrast_grey(img, target=0.4):
    """Adjust contrast of grayscale image to target level (matches torchfree exactly)."""
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200./np.maximum(10, high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0), np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img


def calculate_confidence(output: np.ndarray, charset: str) -> float:
    """Calculate confidence score from model output."""
    # Apply softmax
    preds_prob = softmax(output, axis=2)

    # Get predicted indices
    preds_index = np.argmax(output, axis=2)[0]

    # Collect max probabilities for non-blank, non-duplicate positions
    max_probs_list = []
    prev_idx = None
    for i, idx in enumerate(preds_index):
        if idx != 0 and idx != prev_idx:  # not blank and not duplicate
            max_probs_list.append(preds_prob[0, i, idx])
        prev_idx = idx

    if len(max_probs_list) > 0:
        return custom_mean(np.array(max_probs_list))
    else:
        return 0.0


def decode_recognition(output: np.ndarray, charset: str, ignore_idx: list = [0]) -> str:
    """
    Decode recognition output using CTC greedy decoding.
    Matches torchfree-ocr's recognizer_predict logic exactly.

    Args:
        output: Raw model output (logits) of shape (batch, time, num_classes)
        charset: Character set string (newline separated)
        ignore_idx: Indices to filter out (default [0] for blank)

    Returns:
        Decoded text string
    """
    # Greedy decoding: take argmax directly (like torchfree)
    # Do NOT filter before argmax - that changes the results!
    preds_index = np.argmax(output, axis=2).reshape(-1)

    # Build character list with [blank] at index 0
    charset_list = ['[blank]'] + list(charset)

    # Decode using CTC rules: remove consecutive duplicates and blanks
    preds_size = output.shape[1]
    text_index = preds_index[:preds_size]

    # Remove consecutive duplicates
    mask_not_repeated = np.insert(~(text_index[1:] == text_index[:-1]), 0, True)

    # Remove blank tokens (index 0)
    mask_not_blank = ~np.isin(text_index, np.array(ignore_idx))

    # Combine masks
    mask_combined = mask_not_repeated & mask_not_blank

    # Get characters
    valid_indices = text_index[mask_combined.nonzero()]
    text = ''.join([charset_list[idx] for idx in valid_indices if idx < len(charset_list)])

    return text


def load_charset(charset_name: str, charset_dir: str = 'character') -> str:
    """
    Load character set matching torchfree-ocr.
    For English Gen2 model, charset is hardcoded.
    For other languages, load from file.
    """
    # For English, use the exact charset from torchfree config
    if charset_name == 'en':
        # From recognition_models['gen2']['english_g2']['characters']
        return '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    # For other languages, load from file
    charset_path = Path(charset_dir) / f'{charset_name}_charset.txt'
    if not charset_path.exists():
        # Fallback to old naming convention
        charset_path = Path(charset_dir) / f'{charset_name}_char.txt'

    if not charset_path.exists():
        raise FileNotFoundError(f"Charset file not found: {charset_path}")

    with open(charset_path, 'r', encoding='utf-8') as f:
        # Read all characters (one per line for it_charset.txt)
        chars = f.read().replace('\n', '')

    return chars


def clean_crop_noise(crop: np.ndarray, top_threshold: float = 0.10) -> np.ndarray:
    """
    Remove noise from top of crop by finding first row with significant content.
    This fixes issues where detector includes fragments from lines above.

    Args:
        crop: Grayscale crop image
        top_threshold: Minimum fraction of dark pixels to consider as content

    Returns:
        Cleaned crop with noise removed from top
    """
    if crop is None or crop.size == 0:
        return crop

    h, w = crop.shape

    # Calculate darkness (fraction of pixels < 200) for each row
    for i in range(min(h // 3, h)):  # Check only top third
        row = crop[i, :]
        darkness = np.sum(row < 200) / len(row)

        # Found first row with significant content
        if darkness > top_threshold:
            if i > 0:
                # Remove noise rows from top
                return crop[i:, :]
            break

    return crop


def normalize_punctuation(text: str) -> str:
    """
    Normalize punctuation and apostrophes to standard forms.
    Converts typographic quotes to ASCII standard.
    """
    # Normalize typographic apostrophes to ASCII
    text = text.replace(''', "'")  # U+2019 → U+0027
    text = text.replace(''', "'")  # U+2018 → U+0027
    text = text.replace('`', "'")  # Backtick → apostrophe

    # Normalize quotes
    text = text.replace('"', '"')  # U+201C → U+0022
    text = text.replace('"', '"')  # U+201D → U+0022

    # Normalize dashes
    text = text.replace('–', '-')  # En dash → hyphen
    text = text.replace('—', '-')  # Em dash → hyphen

    return text


def separate_compound_words(text: str) -> str:
    """
    Separate common compound words in Italian where apostrophes were missed.
    Fixes segmentation errors from the detector.
    """
    import re

    # Common patterns where apostrophe is missing
    compound_fixes = {
        # Article + noun contractions (double consonant)
        r'\bdallu': "dall'u",     # dall'università
        r'\bdellu': "dell'u",     # dell'università
        r'\bnellu': "nell'u",     # nell'università
        r'\bsullu': "sull'u",     # sull'ultimo
        r'\ballU': "all'U",       # all'Università

        # Article + noun contractions (single consonant - OCR error)
        r'\bdalu': "dall'u",      # daluscita → dall'uscita
        r'\bdelu': "dell'u",      # deluscita → dell'uscita
        r'\bnelu': "nell'u",      # neluscita → nell'uscita

        # Verbs with missing apostrophe (very common!)
        r'\bcè\b': "c'è",         # cè → c'è (verbo essere)
        r'\bcera\b': "c'era",     # cera → c'era
        r'\bcerano\b': "c'erano", # cerano → c'erano

        # Pronoun contractions
        r'\bcenè\b': "ce n'è",    # ce nè → ce n'è
        r'\bcene\b': "ce ne",     # ce ne (if without accent)

        # Common expressions
        r'\bunarm': "un'arm",     # un'armoniosa
        r'\bunaltr': "un'altr",   # un'altra
        r'\bunult': "un'ult",     # un'ultima
        r'\bunennesimo\b': "un'ennesimo",  # unennesimo → un'ennesimo

        # Verb + pronoun
        r'\bcomè\b': "com'è",     # comè → com'è

        # Dall'/Dell' variations already handled by ľ processing
    }

    for pattern, replacement in compound_fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def fix_contextual_errors(text: str) -> str:
    """
    Fix common contextual OCR errors specific to Italian text.
    These are errors observed in real OCR output that have specific patterns.
    """
    import re

    # Dictionary of common OCR mistakes and their corrections
    contextual_fixes = {
        # Common typos and misrecognitions
        r'\bprogerto\b': "progetto",      # progerto → progetto
        r'\bdallalto\b': "dall'alto",     # dallalto → dall'alto
        r'\bbensi\b': "bensì",            # bensi → bensì (missing accent)
        r'\bcaffe\b': "caffè",            # caffe → caffè (missing accent)
        r'\bcitta\b': "città",            # citta → città (missing accent)
        r'\bpiu\b': "più",                # piu → più (missing accent)
        r'\bgia\b': "già",                # gia → già (missing accent)
        r'\bperche\b': "perché",          # perche → perché (missing accent)
        r'\bcosi\b': "così",              # cosi → così (missing accent)
        r'\bsara\b': "sarà",              # sara → sarà (missing accent)
        r'\bLucía\b': "Lucia",            # Lucía → Lucia (wrong accent)

        # Repeated article errors
        r"l'l'": "l'",                    # l'l' → l' (doubled article)

        # Punctuation fixes
        r'"\s*,': '",',                   # " , → ",
        r'"\s*\.': '."',                  # " . → ."
        r'"\s*;': '";',                   # " ; → ";
        r'"\s*:': '":',                   # " : → ":

        # Wrong punctuation (OCR confuses ; and :)
        r';([a-zà-ù])': r', \1',          # ; → , (before lowercase)
        r':([A-Z])(?![A-Z])': r'. \1',    # : → . (before single uppercase)
    }

    for pattern, replacement in contextual_fixes.items():
        # Use case-insensitive for word boundaries, normal for punctuation
        if pattern.startswith(r'\b'):
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        else:
            text = re.sub(pattern, replacement, text)

    return text


# Global SymSpell instance for Italian (lazy initialization)
_italian_symspell = None

def get_italian_symspell():
    """
    Get or initialize the Italian SymSpell instance.
    Uses lazy initialization to load dictionary only when needed.
    """
    global _italian_symspell

    if _italian_symspell is None:
        try:
            from symspellpy import SymSpell, Verbosity

            _italian_symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

            # Load Italian dictionary (try complete first, fallback to basic)
            dict_path = Path(__file__).parent / "dictionaries" / "it_complete.txt"
            if not dict_path.exists():
                dict_path = Path(__file__).parent / "dictionaries" / "it_basic.txt"

            if not dict_path.exists():
                print(f"Warning: Italian dictionary not found, spell checking disabled")
                return None

            # Load dictionary (format: word frequency)
            _italian_symspell.load_dictionary(str(dict_path), term_index=0, count_index=1)
            print(f"Loaded Italian dictionary: {dict_path.name}")

        except ImportError:
            print("Warning: symspellpy not installed, spell checking disabled")
            return None
        except Exception as e:
            print(f"Warning: Error loading spell checker: {e}")
            return None

    return _italian_symspell


def spell_check_italian(text: str, enable_spell_check: bool = True) -> str:
    """
    Apply spell checking to Italian text using SymSpell.

    Args:
        text: Input text to spell check
        enable_spell_check: Whether spell checking is enabled (default True)

    Returns:
        Corrected text

    Performance: ~0.2 μs per word lookup with SymSpell

    Note: Spell checking is conservative and requires a complete Italian dictionary
    to work effectively. With the current basic dictionary, it's disabled by default
    to avoid over-correction. Enable with --spell-check flag when you have a
    complete frequency dictionary.
    """
    if not enable_spell_check:
        return text

    symspell = get_italian_symspell()
    if symspell is None:
        return text

    from symspellpy import Verbosity
    import re

    # Split text into words while preserving punctuation and spacing
    # Pattern matches: word characters, apostrophes in words, or any other character
    tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]|\s+", text)

    corrected_tokens = []
    corrections_made = 0

    for token in tokens:
        # Only spell check actual words (not punctuation or whitespace)
        if re.match(r"\w+(?:'\w+)?", token):
            # Skip very short words (likely correct or function words)
            if len(token) <= 2:
                corrected_tokens.append(token)
                continue

            # Look up word in dictionary (max edit distance 2)
            suggestions = symspell.lookup(token.lower(), Verbosity.CLOSEST, max_edit_distance=2)

            if suggestions and suggestions[0].distance > 0:
                # Found a potential correction
                original_freq = suggestions[0].count if suggestions[0].distance == 0 else 0
                correction = suggestions[0].term
                correction_freq = suggestions[0].count

                # Only apply correction if:
                # 1. Edit distance is 1 (very close match)
                # 2. OR correction is much more common (10x frequency)
                should_correct = False
                if suggestions[0].distance == 1:
                    should_correct = True
                elif suggestions[0].distance == 2 and correction_freq > original_freq * 10:
                    should_correct = True

                if should_correct:
                    # Preserve original capitalization pattern
                    if token[0].isupper():
                        correction = correction.capitalize()
                    if token.isupper():
                        correction = correction.upper()

                    corrected_tokens.append(correction)
                    corrections_made += 1
                else:
                    corrected_tokens.append(token)
            else:
                # Keep original (either perfect match or no suggestion found)
                corrected_tokens.append(token)
        else:
            # Keep punctuation and whitespace as-is
            corrected_tokens.append(token)

    return ''.join(corrected_tokens)


def postprocess_italian_text(text: str, enable_spell_check: bool = False) -> str:
    """
    Apply post-processing corrections for Italian text.
    Fixes common character substitutions from the recognition model to match actual image content.

    The model produces visually similar but incorrect Unicode characters:
    - ľ (L with caron) appears where apostrophes should be
    - Ī (I with macron) appears instead of lowercase 'l'
    - Ē (E with macron) appears instead of 'È' (E with grave accent)

    Args:
        text: Input text to process
        enable_spell_check: Whether to apply spell checking (default False)
                          Disabled by default due to small dictionary. Enable with --spell-check
    """
    import re

    # Step 1: Normalize punctuation
    text = normalize_punctuation(text)

    # Step 2: Fix Unicode character substitutions
    # Fix 2a: ľ (U+013E) → apostrophe + missing 'l' for double consonants
    # "delľ" → "dell'", "dalľ" → "dall'", "alľ" → "all'", "nelľ" → "nell'"
    # Special case: when ľ appears after 'l', add another 'l' before apostrophe
    text = re.sub(r'([aeiou])lľ', r"\1ll'", text)  # delľ → dell', etc.
    text = re.sub(r'([aeiou])ľ', r"\1l'", text)    # aľ → al' (single consonant)

    # General case for other consonants
    text = re.sub(r'([bcdfgmnpqrstvz])ľ', r"\1'", text)

    # Fix 2b: ľ at start of word (after space/punctuation) → "l'"
    # " ľunico" → " l'unico", " ľora" → " l'ora"
    text = re.sub(r'(\s)ľ([a-z])', r"\1l'\2", text)
    text = re.sub(r'^ľ([a-z])', r"l'\1", text)  # At start of string

    # Fix 2c: iĪ → il (I with macron after 'i' → lowercase 'l')
    text = text.replace('iĪ', 'il')

    # Fix 2d: Standalone Ī → l (in case it appears alone)
    text = text.replace('Ī', 'l')

    # Fix 2e: Ē (E with macron) → È (E with grave accent)
    # The image shows "È" not plain "E"
    text = text.replace('Ē', 'È')

    # Step 3: Separate compound words with missing apostrophes
    text = separate_compound_words(text)

    # Step 4: Fix contextual errors (specific word patterns)
    text = fix_contextual_errors(text)

    # Step 5: Apply spell checking (optional, disabled by default)
    if enable_spell_check:
        text = spell_check_italian(text, enable_spell_check=True)

    return text


def run_ocr(image_path: Path, detector_path: str, recognizer_path: str, charset: str,
            min_size: int = 20, slope_ths: float = 0.1, ycenter_ths: float = 0.5,
            height_ths: float = 0.5, width_ths: float = 0.5, add_margin: float = 0.1,
            lang: str = 'en', enable_spell_check: bool = False) -> List[Tuple]:
    """Run OCR on image - matches torchfree-ocr with per-crop imgW."""
    import math

    det_session = ort.InferenceSession(detector_path, providers=['CPUExecutionProvider'])
    rec_session = ort.InferenceSession(recognizer_path, providers=['CPUExecutionProvider'])

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    orig_h, orig_w = img.shape[:2]

    # Convert to grayscale for recognition (do this BEFORE cropping for pixel-perfect match)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Detection
    det_input, ratio, size_heatmap = detector_preprocess(img)
    det_output = det_session.run(None, {det_session.get_inputs()[0].name: det_input})[0]
    bboxes = detector_postprocess(det_output, ratio)

    if not bboxes:
        return []

    # Step 2: Group text boxes
    polys = [np.array(bbox).astype(np.int32).reshape((-1)) for bbox in bboxes]
    horizontal_list, free_list = group_text_box(
        polys, slope_ths=slope_ths, ycenter_ths=ycenter_ths,
        height_ths=height_ths, width_ths=width_ths, add_margin=add_margin
    )

    # Step 3: Filter by min_size
    if min_size:
        horizontal_list = [box for box in horizontal_list
                          if max(box[1] - box[0], box[3] - box[2]) > min_size]

    # Step 4: Extract all crops
    crops = []
    bbox_coords = []
    
    for region in horizontal_list:
        x_min, x_max, y_min, y_max = region[:4]
        x_min = int(np.clip(x_min, 0, orig_w))
        x_max = int(np.clip(x_max, 0, orig_w))
        y_min = int(np.clip(y_min, 0, orig_h))
        y_max = int(np.clip(y_max, 0, orig_h))

        crop = img_gray[y_min:y_max, x_min:x_max]
        crops.append(crop)
        bbox_coords.append([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])

    for free_box in free_list:
        pts = np.array(free_box, dtype=np.int32)
        x_coords, y_coords = pts[:, 0], pts[:, 1]
        x_min = int(np.clip(x_coords.min(), 0, orig_w))
        x_max = int(np.clip(x_coords.max(), 0, orig_w))
        y_min = int(np.clip(y_coords.min(), 0, orig_h))
        y_max = int(np.clip(y_coords.max(), 0, orig_h))

        crop = img_gray[y_min:y_max, x_min:x_max]
        crops.append(crop)
        bbox_coords.append(free_box)

    if len(crops) == 0:
        return []

    # Step 5: First pass - process all crops
    imgH = 64
    first_pass_results = []

    for bbox, crop in zip(bbox_coords, crops):
        if crop is None or crop.size == 0:
            first_pass_results.append((bbox, "", 0.0, crop))
            continue

        # Crop is already grayscale (numpy array)
        # Note: Crop cleaning disabled - was too aggressive and caused more problems
        h, w = crop.shape

        # Calculate imgW for THIS crop only
        crop_ratio = w / float(h)
        imgW = math.ceil(crop_ratio) * imgH
        resized_w = imgW if int(imgH * crop_ratio) > imgW else int(imgH * crop_ratio)

        # Resize with cv2.INTER_LINEAR (proven to work for English 100%)
        resized_img = cv2.resize(crop, (resized_w, imgH), interpolation=cv2.INTER_LINEAR)

        # Normalize to [-1, 1]
        img_array = resized_img.astype(np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5

        # Pad with last column repeated
        padded = np.zeros((imgH, imgW), dtype=np.float32)
        padded[:, :resized_w] = img_array

        if resized_w < imgW:
            last_col = img_array[:, -1:]
            padded[:, resized_w:] = np.tile(last_col, (1, imgW - resized_w))

        input_tensor = padded[None, None, :, :].astype(np.float32)

        # Inference (batch size 1)
        rec_output = rec_session.run(None, {rec_session.get_inputs()[0].name: input_tensor})[0]

        # Decode and calculate confidence
        text = decode_recognition(rec_output, charset)

        # Apply post-processing for Italian
        if lang == 'it':
            text = postprocess_italian_text(text, enable_spell_check=enable_spell_check)

        confidence = calculate_confidence(rec_output, charset)
        first_pass_results.append((bbox, text, confidence, crop))

    # Step 6: Second pass for low confidence results (like torchfree)
    contrast_ths = 0.1
    adjust_contrast = 0.5
    low_conf_indices = [i for i, (_, _, conf, _) in enumerate(first_pass_results) if conf < contrast_ths]

    second_pass_results = {}
    for idx in low_conf_indices:
        bbox, _, _, crop = first_pass_results[idx]

        # Replicate torchfree second pass EXACTLY:
        # 1. First resize with cv2 (like get_image_list does)
        h, w = crop.shape
        crop_ratio = w / float(h)
        imgW = math.ceil(crop_ratio) * imgH
        resized_w = imgW if int(imgH * crop_ratio) > imgW else int(imgH * crop_ratio)
        crop_resized_cv2 = cv2.resize(crop, (resized_w, imgH), interpolation=cv2.INTER_LINEAR)

        # 2. Apply contrast adjustment to the RESIZED crop (like AlignCollate does)
        resized_img = adjust_contrast_grey(crop_resized_cv2, target=adjust_contrast)

        # Normalize to [-1, 1]
        img_array = resized_img.astype(np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5

        # Pad with last column repeated
        padded = np.zeros((imgH, imgW), dtype=np.float32)
        padded[:, :resized_w] = img_array

        if resized_w < imgW:
            last_col = img_array[:, -1:]
            padded[:, resized_w:] = np.tile(last_col, (1, imgW - resized_w))

        input_tensor = padded[None, None, :, :].astype(np.float32)

        # Inference
        rec_output = rec_session.run(None, {rec_session.get_inputs()[0].name: input_tensor})[0]

        # Decode and calculate confidence
        text2 = decode_recognition(rec_output, charset)

        # Apply post-processing for Italian
        if lang == 'it':
            text2 = postprocess_italian_text(text2, enable_spell_check=enable_spell_check)

        confidence2 = calculate_confidence(rec_output, charset)
        second_pass_results[idx] = (text2, confidence2)

    # Step 7: Merge results - choose better result for each detection
    results = []
    for i, (bbox, text1, conf1, _) in enumerate(first_pass_results):
        if i in second_pass_results:
            text2, conf2 = second_pass_results[i]
            # Choose result with higher confidence
            if conf1 > conf2:
                results.append((bbox, text1, conf1))
            else:
                results.append((bbox, text2, conf2))
        else:
            results.append((bbox, text1, conf1))

    return results


def get_confidence_color(confidence: float) -> Tuple[int, int, int]:
    """BGR color based on confidence."""
    if confidence >= 0.7:
        return (0, 255, 0)
    elif confidence >= 0.4:
        return (0, 255, 255)
    else:
        return (0, 0, 255)


def save_text_file(image_path: Path, results: list) -> Path:
    """Save OCR results to text file."""
    output_path = image_path.parent / f"{image_path.name}.ocr.python.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for bbox, text, confidence in results:
            bbox_str = " ".join(f"({int(p[0])},{int(p[1])})" for p in bbox)
            # Replace newlines in text with spaces to maintain one line per detection
            text_clean = text.replace('\n', ' ').replace('\r', ' ')
            f.write(f"{bbox_str} | {text_clean} | {confidence:.4f}\n")

    return output_path


def save_bbox_image(image_path: Path, results: list, draw_text: bool = True, thickness: int = 2, scale: float = 1.0) -> Path:
    """Save image with colored bounding boxes."""
    output_path = image_path.parent / f"{image_path.name}.ocr.bbox.png"

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    if scale != 1.0:
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    for bbox, text, confidence in results:
        color = get_confidence_color(confidence)
        pts = np.array([[int(p[0] * scale), int(p[1] * scale)] for p in bbox], dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

        if draw_text and text.strip():
            x, y = pts[0]
            label = f"{text} ({confidence:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 * scale
            text_thickness = max(1, int(1 * scale))
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

            cv2.rectangle(image, (x, y - text_h - baseline - 5), (x + text_w + 5, y), color, -1)
            cv2.putText(image, label, (x + 2, y - 5), font, font_scale, (0, 0, 0), text_thickness)

    cv2.imwrite(str(output_path), image)
    return output_path


def process_image(image_path: Path, detector_path: str, recognizer_path: str, charset: str, mode: str, **kwargs):
    """Process single image."""
    # Extract OCR parameters from kwargs
    lang = kwargs.get('lang', 'en')
    enable_spell_check = kwargs.get('enable_spell_check', False)
    results = run_ocr(image_path, detector_path, recognizer_path, charset, lang=lang, enable_spell_check=enable_spell_check)

    outputs = []
    if mode in ['text', 'all']:
        text_file = save_text_file(image_path, results)
        outputs.append(str(text_file.name))

    if mode in ['visualize', 'all']:
        bbox_file = save_bbox_image(image_path, results, kwargs.get('draw_text', True),
                                     kwargs.get('thickness', 2), kwargs.get('scale', 1.0))
        outputs.append(str(bbox_file.name))

    return results, outputs


def main():
    parser = argparse.ArgumentParser(description="Process images with OCR")
    parser.add_argument("--dataset", type=Path, default=Path("dataset/base"), help="Dataset directory")
    parser.add_argument("--models", type=Path, default=Path("models/cpu"), help="Models directory")
    parser.add_argument("--lang", type=str, default="en", help="Language code")
    parser.add_argument("--mode", choices=['text', 'visualize', 'all'], default='all', help="Processing mode")
    parser.add_argument("--no-text", action="store_true", help="Don't draw text on bbox images")
    parser.add_argument("--thickness", type=int, default=2, help="Bbox line thickness")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for bbox images")
    parser.add_argument("--json", type=Path, help="Save JSON report")
    parser.add_argument("--extensions", type=str, default=".png,.jpg,.jpeg", help="Image extensions")
    parser.add_argument("--spell-check", action="store_true", help="Enable spell checking for Italian (experimental, requires complete dictionary)")

    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset not found: {args.dataset}")
        sys.exit(1)

    if not args.models.exists():
        print(f"Error: Models directory not found: {args.models}")
        sys.exit(1)

    detector_path = args.models / "detection.onnx"

    lang_to_model = {
        'en': 'english_g2_rec.onnx',
        'it': 'latin_g2_rec.onnx',
        'fr': 'latin_g2_rec.onnx',
        'de': 'latin_g2_rec.onnx',
        'es': 'latin_g2_rec.onnx',
    }

    lang_to_charset = {
        'en': 'en',  # English uses hardcoded charset
        'it': 'it',  # Italian uses charset from it_charset.txt (351 chars with accents)
        'fr': 'latin',
        'de': 'latin',
        'es': 'latin',
    }

    recognizer_path = args.models / lang_to_model.get(args.lang, 'english_g2_rec.onnx')
    charset_name = lang_to_charset.get(args.lang, 'latin')

    if not detector_path.exists():
        print(f"Error: Detector model not found: {detector_path}")
        sys.exit(1)

    if not recognizer_path.exists():
        print(f"Error: Recognizer model not found: {recognizer_path}")
        sys.exit(1)

    try:
        charset = load_charset(charset_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    extensions = [ext.strip() for ext in args.extensions.split(",")]
    image_files = []
    for ext in extensions:
        for img in args.dataset.glob(f"*{ext}"):
            if ".ocr." not in img.name:
                image_files.append(img)

    image_files = sorted(image_files)

    if not image_files:
        print(f"Error: No images found")
        sys.exit(1)

    print(f"Dataset: {args.dataset}")
    print(f"Models: {args.models}")
    print(f"Mode: {args.mode}")
    print(f"Images: {len(image_files)}")
    print("-" * 50)

    all_results = []
    start_time = time.time()

    for image_path in image_files:
        print(f"\n{image_path.name}")

        try:
            results, outputs = process_image(
                image_path, str(detector_path), str(recognizer_path), charset, args.mode,
                lang=args.lang,
                enable_spell_check=args.spell_check,
                draw_text=not args.no_text,
                thickness=args.thickness,
                scale=args.scale
            )

            for output in outputs:
                print(f"  → {output}")

            if args.json:
                all_results.append({
                    "image": image_path.name,
                    "path": str(image_path),
                    "num_detections": len(results),
                    "results": [
                        {
                            "text": text,
                            "confidence": round(float(confidence), 4),
                            "bbox": [[int(x), int(y)] for x, y in bbox]
                        }
                        for bbox, text, confidence in results
                    ]
                })

        except Exception as e:
            print(f"  ✗ Error: {e}")

    elapsed = time.time() - start_time

    if args.json:
        metadata = {
            "dataset": str(args.dataset),
            "language": args.lang,
            "mode": args.mode,
            "total_images": len(image_files),
            "total_time_seconds": round(elapsed, 3)
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"metadata": metadata, "results": all_results}, f, indent=2, ensure_ascii=False)
        print(f"\n✓ JSON saved: {args.json}")

    print(f"\nTotal time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
