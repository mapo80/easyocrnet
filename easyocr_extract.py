import glob
import os
from typing import Dict

import easyocr


def derive_lang(name: str) -> str:
    name = os.path.basename(name).split('.')[0].lower()
    if name.startswith('generated'):
        return 'it'
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


def main() -> None:
    readers: Dict[str, easyocr.Reader] = {}
    model_overrides = {
        'it': {
            'recog_network': 'latin_g2',
        },
    }

    for img_file in glob.glob('examples/*.[pj][pn]g'):
        base, _ = os.path.splitext(img_file)
        lang = derive_lang(base)
        if lang not in readers:
            overrides = model_overrides.get(lang, {})
            readers[lang] = easyocr.Reader([lang], gpu=False, **overrides)
        result = readers[lang].readtext(img_file, detail=0, paragraph=False)
        text = '\n'.join(line.strip() for line in result if line.strip())
        if text and not text.endswith('\n'):
            text += '\n'
        with open(base + '.easyocr.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'{img_file}: {text}')


if __name__ == '__main__':
    main()
