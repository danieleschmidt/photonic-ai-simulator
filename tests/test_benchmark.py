"""Tests for AccelerationBenchmark."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from photonic.accelerator import PhotonicAccelerator
from photonic.benchmark import AccelerationBenchmark, BenchmarkResult


def make_bench(size=4):
    acc = PhotonicAccelerator(size)
    return AccelerationBenchmark(acc)


def test_run_returns_result():
    bench = make_bench(4)
    result = bench.run(4)
    assert isinstance(result, BenchmarkResult)


def test_run_matrix_size():
    bench = make_bench(4)
    result = bench.run(4)
    assert result.matrix_size == 4


def test_speedup_factor():
    bench = make_bench(4)
    result = bench.run(4)
    # Photonic should have > 1x ops/J vs electronic
    assert result.speedup_factor > 1.0


def test_error_finite():
    bench = make_bench(4)
    result = bench.run(4)
    assert 0.0 <= result.error_frobenius < 10.0


def test_timings_positive():
    bench = make_bench(4)
    result = bench.run(4)
    assert result.photonic_time_ms >= 0
    assert result.electronic_time_ms >= 0
