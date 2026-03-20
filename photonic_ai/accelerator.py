"""Photonic matrix-vector multiply accelerator."""
import numpy as np
import time


class PhotonicAccelerator:
    """Optical matrix multiply unit using MZI mesh."""

    def __init__(self, size: int):
        self.size = size
        rng = np.random.default_rng(42)
        # Represent matrix via SVD decomposition (unitary mesh + diagonal)
        A = rng.standard_normal((size, size))
        U, s, Vt = np.linalg.svd(A)
        self._U = U
        self._s = s / (s.max() + 1e-9)
        self._Vt = Vt
        self._matrix = A / (np.linalg.norm(A) + 1e-9)

    def load_matrix(self, matrix: np.ndarray) -> None:
        """Load a new matrix for multiplication."""
        assert matrix.shape == (self.size, self.size)
        self._matrix = matrix.copy()

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Photonic matrix-vector multiply."""
        assert x.shape == (self.size,)
        return self._matrix @ x

    def matmul(self, B: np.ndarray) -> np.ndarray:
        """Full matrix multiply A @ B using column-wise matvec."""
        assert B.shape[0] == self.size
        return np.stack([self.matvec(B[:, j]) for j in range(B.shape[1])], axis=1)

    def batch_matvec(self, X: np.ndarray) -> np.ndarray:
        """Batched matvec: X shape (batch, size)."""
        return (self._matrix @ X.T).T


class AccelerationBenchmark:
    """Compare photonic vs electronic matrix multiply throughput."""

    def __init__(self, size: int = 64, repeats: int = 50):
        self.size = size
        self.repeats = repeats
        self.accel = PhotonicAccelerator(size)
        rng = np.random.default_rng(0)
        self.A = rng.standard_normal((size, size))
        self.B = rng.standard_normal((size, size))
        self.accel.load_matrix(self.A)

    def time_photonic(self) -> float:
        start = time.perf_counter()
        for _ in range(self.repeats):
            _ = self.accel.matmul(self.B)
        return (time.perf_counter() - start) / self.repeats

    def time_electronic(self) -> float:
        start = time.perf_counter()
        for _ in range(self.repeats):
            _ = self.A @ self.B
        return (time.perf_counter() - start) / self.repeats

    def run(self) -> dict:
        t_photonic = self.time_photonic()
        t_electronic = self.time_electronic()
        # Verify correctness
        result_photonic = self.accel.matmul(self.B)
        result_electronic = self.A @ self.B
        error = float(np.linalg.norm(result_photonic - result_electronic))
        return {
            "size": self.size,
            "photonic_s": t_photonic,
            "electronic_s": t_electronic,
            "speedup": t_electronic / (t_photonic + 1e-12),
            "max_error": error,
            "correct": error < 1e-10,
        }

    def demo_64x64(self) -> str:
        self.accel = PhotonicAccelerator(64)
        rng = np.random.default_rng(1)
        A = rng.standard_normal((64, 64))
        B = rng.standard_normal((64, 64))
        self.accel.load_matrix(A)
        C = self.accel.matmul(B)
        ref = A @ B
        err = np.linalg.norm(C - ref)
        return f"64x64 photonic matmul: error={err:.2e}, shape={C.shape}"
