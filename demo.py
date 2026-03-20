#!/usr/bin/env python3
"""Demo: 64×64 photonic matrix multiply."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from photonic.accelerator import PhotonicAccelerator
from photonic.benchmark import AccelerationBenchmark

def main():
    print("Photonic AI Simulator Demo — 64×64 Matrix Multiply")
    print("="*55)

    acc = PhotonicAccelerator(size=64)
    bench = AccelerationBenchmark(acc)
    result = bench.run(size=64)
    bench.print_report(result)

    print(f"MZ switches in mesh: {acc.num_switches()}")
    print(f"Programmable params: {acc.num_parameters()}")

if __name__ == "__main__":
    main()
