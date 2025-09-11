import glob
import glob
import os
import difflib

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


def load_charset(name: str) -> str:
    path = os.path.join('character', f'{name}_char.txt')
    with open(path, encoding='utf-8') as f:
        return f.read()


def detector_preprocess(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (800, 608))
    x = np.transpose(normalizeMeanVariance(img), (2, 0, 1))
    return x[None, ...]


def detector_postprocess(out: np.ndarray):
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


def run_onnx(img_path: str, charset: str = 'en') -> str:
    img = cv2.imread(img_path)
    det = ort.InferenceSession('models/EasyOCRDetector.onnx')
    rec = ort.InferenceSession('models/EasyOCRRecognizer.onnx')

    det_in = detector_preprocess(img)
    det_out = det.run(None, {'image': det_in})[0]
    bbox = detector_postprocess(det_out)
    if bbox is None:
        return ''
    x_min, y_min, x_max, y_max = bbox
    crop = img[y_min:y_max, x_min:x_max]
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
    out = rec.run(None, {'image': crop})[0]
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
    return text


def main() -> None:
    def derive_charset(name: str) -> str:
        name = os.path.basename(name).split('.')[0].lower()
        if name in {
            'english',
            'example',
            'example2',
            'example3',
            'easyocr_framework',
            'width_ths',
        }:
            return 'en'
        if name == 'french':
            return 'fr'
        if name == 'japanese':
            return 'ja'
        if name == 'korean':
            return 'ko'
        if name == 'chinese':
            return 'ch_sim'
        if name == 'thai':
            return 'th'
        return 'en'

    image_files = glob.glob('examples/*.[pj][pn]g') + glob.glob('examples/*.jpeg')
    for img_file in image_files:
        base, _ = os.path.splitext(img_file)
        charset = derive_charset(base)
        text = run_onnx(img_file, charset)
        with open(base + '.onnx.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        python_path = base + '.python.txt'
        python_txt = (
            open(python_path, encoding='utf-8').read() if os.path.exists(python_path) else ''
        )
        diff_lines = list(
            difflib.unified_diff(
                python_txt.splitlines(),
                text.splitlines(),
                fromfile='python',
                tofile='onnx',
                lineterm='',
            )
        )
        if diff_lines:
            with open(base + '.diff.txt', 'w', encoding='utf-8') as df:
                df.write('\n'.join(diff_lines))
        print(f'{img_file}: ONNX="{text}" | matches baseline? {text == python_txt}')


if __name__ == '__main__':
    main()
