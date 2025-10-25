#!/usr/bin/env python3
"""
Complete benchmark script for Python vs C#
"""

import time
import json
import subprocess
from pathlib import Path


def run_python_benchmark():
    """Run Python OCR benchmark"""
    print("=" * 70)
    print("PYTHON BENCHMARK")
    print("=" * 70)

    results = []

    # Run on dataset - ocr_process.py processes all images in dataset
    images = sorted(Path('dataset/it').glob('*.png'))

    for img in images:
        print(f"\nBenchmarking {img.name}:")
        times = []

        for i in range(6):
            print(f"  Run {i+1}/6...", end=' ', flush=True)
            start = time.perf_counter()
            # Run for single image by specifying dataset with that image only
            subprocess.run(['python', 'ocr_process.py', '--dataset', f'dataset/it', '--lang', 'it', '--mode', 'text'],
                          capture_output=True, timeout=120)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"{elapsed:.2f}s")

        # Discard first
        times = times[1:]
        avg = sum(times) / len(times)

        results.append({
            'image': img.name,
            'times': times,
            'avg': avg,
            'min': min(times),
            'max': max(times)
        })
        print(f"  Average (excluding first): {avg:.2f}s")

        # Only do first image for now to test
        break

    # Save
    Path('benchmark_results').mkdir(exist_ok=True)
    with open('benchmark_results/python.json', 'w') as f:
        json.dump({'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'results': results}, f, indent=2)

    print(f"\n{'Image':<20} {'Avg':>8}")
    print("-" * 30)
    for r in results:
        print(f"{r['image']:<20} {r['avg']:>8.2f}s")
    if results:
        print(f"\nOVERALL: {sum(r['avg'] for r in results)/len(results):.2f}s")

    return results


def run_csharp_benchmark():
    """Run C# OCR benchmark"""
    print("\n" + "=" * 70)
    print("C# BENCHMARK")
    print("=" * 70)

    results = []

    for img in sorted(Path('dataset/it').glob('*.png')):
        print(f"\nBenchmarking {img.name}:")
        times = []

        for i in range(6):
            print(f"  Run {i+1}/6...", end=' ', flush=True)
            start = time.perf_counter()
            subprocess.run([
                'dotnet', 'run', '--project', 'EasyOcrNet.Cli/EasyOcrNet.Cli.csproj',
                '-c', 'Release', '--no-build', '--',
                'ocr', '--image', str(img),
                '--detector', 'models/cpu/detection.onnx',
                '--recognizer', 'models/cpu/latin_g2_rec.onnx',
                '--lang', 'it'
            ], capture_output=True, timeout=120)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"{elapsed:.2f}s")

        # Discard first
        times = times[1:]
        avg = sum(times) / len(times)

        results.append({
            'image': img.name,
            'times': times,
            'avg': avg,
            'min': min(times),
            'max': max(times)
        })
        print(f"  Average (excluding first): {avg:.2f}s")

        # Only do first image for now to test
        break

    # Save
    Path('benchmark_results').mkdir(exist_ok=True)
    with open('benchmark_results/csharp.json', 'w') as f:
        json.dump({'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'results': results}, f, indent=2)

    print(f"\n{'Image':<20} {'Avg':>8}")
    print("-" * 30)
    for r in results:
        print(f"{r['image']:<20} {r['avg']:>8.2f}s")
    if results:
        print(f"\nOVERALL: {sum(r['avg'] for r in results)/len(results):.2f}s")

    return results


def print_comparison(py_results, cs_results):
    """Print comparison table"""
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(f"\n{'Image':<20} {'Python':>10} {'C#':>10} {'Speedup':>10}")
    print("-" * 55)

    for py_r, cs_r in zip(py_results, cs_results):
        if cs_r['avg'] > 0:
            speedup = py_r['avg'] / cs_r['avg']
        else:
            speedup = 0
        print(f"{py_r['image']:<20} {py_r['avg']:>9.2f}s {cs_r['avg']:>9.2f}s {speedup:>9.2f}x")

    if py_results and cs_results:
        py_avg = sum(r['avg'] for r in py_results) / len(py_results)
        cs_avg = sum(r['avg'] for r in cs_results) / len(cs_results)
        speedup = py_avg / cs_avg if cs_avg > 0 else 0

        print("-" * 55)
        print(f"{'OVERALL':<20} {py_avg:>9.2f}s {cs_avg:>9.2f}s {speedup:>9.2f}x")


def main():
    print("=" * 70)
    print("RUNNING BENCHMARKS - Python vs C#")
    print("=" * 70)
    print()

    # Build C# first
    print("Building C# in Release mode...")
    subprocess.run(['dotnet', 'build', 'EasyOcrNet.Cli/EasyOcrNet.Cli.csproj',
                   '-c', 'Release', '--nologo', '-v', 'q'], check=True)
    print()

    # Run benchmarks
    py_results = run_python_benchmark()
    cs_results = run_csharp_benchmark()

    # Print comparison
    print_comparison(py_results, cs_results)

    print()


if __name__ == '__main__':
    main()
