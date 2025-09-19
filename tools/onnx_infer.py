import os
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from easyocr.craft_utils import adjustResultCoordinates, getDetBoxes
from easyocr.imgproc import normalizeMeanVariance

DETECTOR_ARGS = {
    'text_threshold': 0.7,
    'link_threshold': 0.4,
    'low_text': 0.4,
    'poly': False,
}


def load_charset(name: str, charset_dir: str = 'character') -> str:
    path = os.path.join(charset_dir, f'{name}_char.txt')
    with open(path, encoding='utf-8') as f:
        return f.read()


def detector_preprocess(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (800, 608))
    x = np.transpose(normalizeMeanVariance(img), (2, 0, 1))
    return x[None, ...]


def detector_postprocess(out: np.ndarray) -> Tuple[int, int, int, int] | None:
    score_text = out[0, :, :, 0]
    score_link = out[0, :, :, 1]
    boxes, polys, _ = getDetBoxes(
        score_text,
        score_link,
        text_threshold=DETECTOR_ARGS['text_threshold'],
        link_threshold=DETECTOR_ARGS['link_threshold'],
        low_text=DETECTOR_ARGS['low_text'],
        poly=DETECTOR_ARGS['poly'],
        estimate_num_chars=False,
    )
    if len(boxes) == 0:
        return None
    det_h, det_w = score_text.shape
    boxes = adjustResultCoordinates(boxes, 800 / det_w, 608 / det_h)
    boxes = np.array(boxes).reshape(-1, 2)
    x_min = boxes[:, 0].min()
    y_min = boxes[:, 1].min()
    x_max = boxes[:, 0].max()
    y_max = boxes[:, 1].max()
    return int(x_min), int(y_min), int(x_max), int(y_max)


def recognizer_preprocess(crop: np.ndarray) -> np.ndarray:
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = crop.shape
    scale = 64.0 / h
    new_w = min(1000, int(np.ceil(w * scale)))
    crop = cv2.resize(crop, (new_w, 64))
    if new_w < 1000:
        pad = np.tile(crop[:, -1:], (1, 1000 - new_w))
        crop = np.concatenate([crop, pad], axis=1)
    crop = crop.astype(np.float32)
    crop = (crop / 255.0 - 0.5) / 0.5
    crop = crop[None, None, :, :]
    return crop


def run(
    detector_path: str,
    recognizer_path: str,
    image_path: str,
    charset: str,
    providers: List[str] | None = None,
) -> Tuple[str, List[str]]:
    """Run ONNX detector+recognizer and return text and providers used."""
    providers = providers or ['CPUExecutionProvider']
    det = ort.InferenceSession(detector_path, providers=providers)
    rec = ort.InferenceSession(recognizer_path, providers=providers)
    det_input = det.get_inputs()[0].name
    rec_input = rec.get_inputs()[0].name
    providers_used = det.get_providers()

    img = cv2.imread(image_path)
    det_in = detector_preprocess(img)
    det_out = det.run(None, {det_input: det_in})[0]
    bbox = detector_postprocess(det_out)
    if bbox is None:
        return '', providers_used
    x_min, y_min, x_max, y_max = bbox
    crop = img[y_min:y_max, x_min:x_max]
    rec_in = recognizer_preprocess(crop)
    out = rec.run(None, {rec_input: rec_in})[0]

    charset_txt = load_charset(charset)
    prev = 0
    text = ''
    for t in range(out.shape[1]):
        idx = int(out[0, t].argmax())
        if idx > 0 and idx != prev:
            ci = idx - 1
            if ci < len(charset_txt):
                text += charset_txt[ci]
        prev = idx
    return text, providers_used
