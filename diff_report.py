import glob, os, difflib

for onnx_file in glob.glob('examples/*.onnx.txt'):
    base = onnx_file[:-len('.onnx.txt')]
    py_file = base + '.python.txt'
    with open(onnx_file, encoding='utf-8') as f:
        onnx_txt = f.read().strip()
    with open(py_file, encoding='utf-8') as f:
        py_txt = f.read().strip()
    diff = list(difflib.unified_diff(py_txt.splitlines(), onnx_txt.splitlines(), fromfile='python', tofile='onnx', lineterm=''))
    diff_path = base + '.diff.txt'
    with open(diff_path, 'w', encoding='utf-8') as df:
        df.write('\n'.join(diff))
    print(f'{os.path.basename(base)}: match={not diff}')
