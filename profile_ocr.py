#!/usr/bin/env python3
"""
OCR Profiling Script - Detailed timing for each phase
Identifies performance bottlenecks in the OCR pipeline
"""

import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List


def profile_csharp_ocr(image_path: str, iterations: int = 3) -> Dict:
    """
    Profile C# OCR with detailed phase timing

    Requires modified C# code with timing output
    """
    print(f"\n{'='*70}")
    print(f"PROFILING: {Path(image_path).name}")
    print(f"{'='*70}\n")

    all_timings = []

    for i in range(iterations):
        print(f"Run {i+1}/{iterations}...", end=' ', flush=True)

        start = time.perf_counter()

        # Run C# OCR with verbose timing
        result = subprocess.run(
            [
                'dotnet', 'run', '--project', 'EasyOcrNet.Cli/EasyOcrNet.Cli.csproj',
                '-c', 'Release', '--no-build', '--',
                'ocr', '--image', image_path,
                '--detector', 'models/cpu/detection.onnx',
                '--recognizer', 'models/cpu/latin_g2_rec.onnx',
                '--lang', 'it',
                '--profile'  # Enable profiling output
            ],
            capture_output=True,
            text=True,
            timeout=120
        )

        elapsed = time.perf_counter() - start
        print(f"{elapsed:.2f}s")

        # Parse timing output from C# (if available)
        timings = parse_timing_output(result.stdout)
        if timings:
            all_timings.append(timings)

    # Calculate averages
    if all_timings:
        avg_timings = calculate_averages(all_timings)
        return avg_timings

    return None


def parse_timing_output(output: str) -> Dict:
    """Parse timing information from C# output"""
    timings = {}

    for line in output.split('\n'):
        # Expected format: "[TIMING] Phase: 123.45ms"
        if '[TIMING]' in line:
            parts = line.split(':')
            if len(parts) == 2:
                phase = parts[0].replace('[TIMING]', '').strip()
                time_str = parts[1].replace('ms', '').strip()
                try:
                    timings[phase] = float(time_str)
                except ValueError:
                    pass

    return timings


def calculate_averages(all_timings: List[Dict]) -> Dict:
    """Calculate average timing for each phase"""
    if not all_timings:
        return {}

    # Collect all phase names
    all_phases = set()
    for timing in all_timings:
        all_phases.update(timing.keys())

    # Calculate averages
    averages = {}
    for phase in all_phases:
        values = [t.get(phase, 0) for t in all_timings if phase in t]
        if values:
            averages[phase] = {
                'avg': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }

    return averages


def print_profile_report(timings: Dict):
    """Print detailed profiling report"""
    print(f"\n{'='*70}")
    print("PROFILING REPORT")
    print(f"{'='*70}\n")

    if not timings:
        print("No timing data available. Enable --profile flag in C# code.")
        return

    # Sort by average time (descending)
    sorted_phases = sorted(timings.items(),
                          key=lambda x: x[1]['avg'],
                          reverse=True)

    print(f"{'Phase':<40} {'Avg (ms)':>10} {'Min':>10} {'Max':>10} {'%':>8}")
    print("-" * 80)

    total_time = sum(t['avg'] for t in timings.values())

    for phase, stats in sorted_phases:
        percentage = (stats['avg'] / total_time * 100) if total_time > 0 else 0
        print(f"{phase:<40} {stats['avg']:>10.2f} {stats['min']:>10.2f} "
              f"{stats['max']:>10.2f} {percentage:>7.1f}%")

    print("-" * 80)
    print(f"{'TOTAL':<40} {total_time:>10.2f}")

    # Identify top bottlenecks
    print(f"\n{'='*70}")
    print("TOP 3 BOTTLENECKS")
    print(f"{'='*70}\n")

    for i, (phase, stats) in enumerate(sorted_phases[:3], 1):
        percentage = (stats['avg'] / total_time * 100) if total_time > 0 else 0
        print(f"{i}. {phase}")
        print(f"   Time: {stats['avg']:.2f}ms ({percentage:.1f}% of total)")
        print(f"   Range: {stats['min']:.2f}ms - {stats['max']:.2f}ms")
        print()


def estimate_optimization_impact():
    """Estimate impact of various optimizations"""
    print(f"\n{'='*70}")
    print("OPTIMIZATION OPPORTUNITIES")
    print(f"{'='*70}\n")

    optimizations = [
        {
            'name': 'Batch Recognition Inference',
            'impact': '30-40%',
            'difficulty': 'Medium',
            'priority': 'HIGH',
            'phases': ['Recognition Inference (per crop)']
        },
        {
            'name': 'ONNX Session Optimization',
            'impact': '5-10%',
            'difficulty': 'Easy',
            'priority': 'HIGH',
            'phases': ['Detection Inference', 'Recognition Inference']
        },
        {
            'name': 'Parallel Crop Preprocessing',
            'impact': '5-8%',
            'difficulty': 'Easy',
            'priority': 'MEDIUM',
            'phases': ['Crop Preprocessing']
        },
        {
            'name': 'Memory Pooling',
            'impact': '3-5%',
            'difficulty': 'Medium',
            'priority': 'MEDIUM',
            'phases': ['All tensor allocations']
        },
        {
            'name': 'SIMD Vectorization',
            'impact': '3-5%',
            'difficulty': 'Hard',
            'priority': 'LOW',
            'phases': ['Image normalization', 'Score map processing']
        }
    ]

    for opt in optimizations:
        print(f"[{opt['priority']}] {opt['name']}")
        print(f"    Impact: {opt['impact']} | Difficulty: {opt['difficulty']}")
        print(f"    Target phases: {', '.join(opt['phases'])}")
        print()


def main():
    """Run profiling on Italian dataset"""

    print("="*70)
    print("C# OCR PERFORMANCE PROFILING")
    print("="*70)

    # Find first image in dataset
    images = sorted(Path('dataset/it').glob('*.png'))

    if not images:
        print("ERROR: No images found in dataset/it/")
        return 1

    # Profile first image
    image_path = str(images[0])

    print(f"\nNote: This requires C# code to output timing information")
    print(f"Add '[TIMING] Phase Name: XXX.XXms' to stdout for each phase")
    print()

    timings = profile_csharp_ocr(image_path, iterations=3)

    if timings:
        print_profile_report(timings)
    else:
        print("\n⚠️  No timing data captured from C#")
        print("Next steps:")
        print("1. Add timing output to C# code")
        print("2. Use System.Diagnostics.Stopwatch for each phase")
        print("3. Output format: Console.WriteLine($\"[TIMING] Phase: {elapsed}ms\");")

    # Always show optimization recommendations
    estimate_optimization_impact()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
