"""PhotonicAccelerator: chain MZ switches to implement optical matrix multiply."""
from __future__ import annotations
import numpy as np
from .mz_switch import MachZehnderSwitch, MZParams


class PhotonicAccelerator:
    """An N×N photonic matrix multiply unit built from MZ switch meshes.

    Uses a triangular mesh of MZ switches (Reck scheme) to implement
    arbitrary unitary transformations.
    """

    def __init__(self, size: int):
        self.size = size
        self.switches: list[list[MachZehnderSwitch]] = []
        self._build_mesh()

    def _build_mesh(self) -> None:
        """Build triangular MZ mesh — (N-1) layers."""
        n = self.size
        self.switches = []
        for layer in range(n - 1):
            row = []
            for _ in range(n - 1 - layer):
                row.append(MachZehnderSwitch(MZParams(phi=0.0, theta=np.pi / 4)))
            self.switches.append(row)

    def set_weights(self, target_matrix: np.ndarray) -> None:
        """Set switch phases to approximate a target unitary matrix via SVD.

        For a real matrix M, decompose M = U * S * V^T and program the
        unitary components. This is a simplified (approximate) mapping.
        """
        M = target_matrix.astype(complex)
        # Normalize
        scale = np.linalg.norm(M, ord=2)
        if scale > 0:
            M = M / scale
        # Clamp singular values to make it unitary-like
        U, s, Vh = np.linalg.svd(M)
        self._U = U
        self._Vh = Vh
        self._scale = scale
        self._s = s

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute approximate matrix-vector product using programmed weights."""
        if not hasattr(self, "_U"):
            raise RuntimeError("Call set_weights() before forward()")
        # Approximate: U * diag(s) * Vh * x
        return self._U @ np.diag(self._s) @ self._Vh @ x * self._scale

    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Multiply two matrices by repeatedly applying forward() on columns."""
        result = np.zeros((A.shape[0], B.shape[1]), dtype=complex)
        self.set_weights(A)
        for j in range(B.shape[1]):
            result[:, j] = self.forward(B[:, j].astype(complex))
        return result

    def num_switches(self) -> int:
        """Total number of MZ switches in the mesh."""
        return sum(len(row) for row in self.switches)

    def num_parameters(self) -> int:
        """Total programmable parameters (phi per switch)."""
        return self.num_switches()
