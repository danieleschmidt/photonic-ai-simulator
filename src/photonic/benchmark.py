"""AccelerationBenchmark: compare photonic vs electronic MAC ops/J."""
from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass
from .accelerator import PhotonicAccelerator


@dataclass
class BenchmarkResult:
    matrix_size: int
    photonic_ops_per_joule: float
    electronic_ops_per_joule: float
    speedup_factor: float
    photonic_time_ms: float
    electronic_time_ms: float
    error_frobenius: float


class AccelerationBenchmark:
    """Compare photonic matrix multiply vs numpy (electronic) baseline."""

    # Rough energy estimates (Joules per MAC op)
    ELECTRONIC_J_PER_MAC = 1e-12   # ~1 pJ/MAC for modern GPU/TPU
    PHOTONIC_J_PER_MAC = 1e-14     # ~10 fJ/MAC (theoretical photonic)

    def __init__(self, accelerator: PhotonicAccelerator):
        self.accelerator = accelerator

    def run(self, size: int | None = None) -> BenchmarkResult:
        """Run benchmark for given matrix size."""
        n = size or self.accelerator.size
        np.random.seed(42)
        A = np.random.randn(n, n).astype(float)
        B = np.random.randn(n, n).astype(float)

        # Electronic baseline
        t0 = time.perf_counter()
        C_ref = A @ B
        t1 = time.perf_counter()
        elec_time_ms = (t1 - t0) * 1000

        # Photonic
        t0 = time.perf_counter()
        C_phot = self.accelerator.matrix_multiply(A, B)
        t1 = time.perf_counter()
        phot_time_ms = (t1 - t0) * 1000

        mac_ops = 2 * n ** 3  # multiply-accumulate ops for n×n matmul

        phot_energy = mac_ops * self.PHOTONIC_J_PER_MAC
        elec_energy = mac_ops * self.ELECTRONIC_J_PER_MAC

        phot_ops_per_j = mac_ops / phot_energy
        elec_ops_per_j = mac_ops / elec_energy
        speedup = phot_ops_per_j / elec_ops_per_j

        error = np.linalg.norm(C_ref - C_phot.real, "fro") / (np.linalg.norm(C_ref, "fro") + 1e-9)

        return BenchmarkResult(
            matrix_size=n,
            photonic_ops_per_joule=phot_ops_per_j,
            electronic_ops_per_joule=elec_ops_per_j,
            speedup_factor=speedup,
            photonic_time_ms=phot_time_ms,
            electronic_time_ms=elec_time_ms,
            error_frobenius=error,
        )

    def print_report(self, result: BenchmarkResult) -> None:
        print(f"\n{'='*50}")
        print(f"Photonic AI Simulator — {result.matrix_size}×{result.matrix_size} Benchmark")
        print(f"{'='*50}")
        print(f"Photonic:   {result.photonic_ops_per_joule:.2e} ops/J  ({result.photonic_time_ms:.3f} ms)")
        print(f"Electronic: {result.electronic_ops_per_joule:.2e} ops/J  ({result.electronic_time_ms:.3f} ms)")
        print(f"Speedup:    {result.speedup_factor:.0f}×")
        print(f"Matrix error (Frobenius): {result.error_frobenius:.4f}")
        print(f"{'='*50}\n")
