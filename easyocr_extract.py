import glob
import os
from typing import Dict

import easyocr


def derive_lang(name: str) -> str:
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


def main() -> None:
    readers: Dict[str, easyocr.Reader] = {}
    for img_file in glob.glob('examples/*.[pj][pn]g'):
        base, _ = os.path.splitext(img_file)
        lang = derive_lang(base)
        if lang not in readers:
            readers[lang] = easyocr.Reader([lang], gpu=False)
        result = readers[lang].readtext(img_file, detail=0)
        text = ' '.join(result)
        with open(base + '.python.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'{img_file}: {text}')


if __name__ == '__main__':
    main()
