"""MachZehnderSwitch: 2×2 optical switch with phase-controlled splitting ratio."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class MZParams:
    phi: float = 0.0       # differential phase shift (radians)
    theta: float = np.pi / 4  # coupler angle (π/4 = 50/50 splitter)
    loss: float = 0.0      # insertion loss (amplitude factor, 0 = lossless)


class MachZehnderSwitch:
    """2×2 Mach-Zehnder interferometer switch.

    Implements the transfer matrix:
        U = [[cos(θ)e^(iφ),  i·sin(θ)],
             [i·sin(θ)e^(iφ), cos(θ)]]
    """

    def __init__(self, params: MZParams):
        self.params = params

    def transfer_matrix(self) -> np.ndarray:
        """Return the 2×2 complex transfer matrix."""
        theta = self.params.theta
        phi = self.params.phi
        t = np.cos(theta)
        k = np.sin(theta)
        mat = np.array([
            [t * np.exp(1j * phi), 1j * k],
            [1j * k * np.exp(1j * phi), t],
        ], dtype=complex)
        # Apply insertion loss
        loss_factor = 1.0 - self.params.loss
        return mat * loss_factor

    def apply(self, fields: np.ndarray) -> np.ndarray:
        """Apply switch to input field vector [E_in1, E_in2]."""
        return self.transfer_matrix() @ fields

    def splitting_ratio(self) -> tuple[float, float]:
        """Return (cross, bar) power splitting ratios for unit input on port 1."""
        fields_in = np.array([1.0 + 0j, 0.0 + 0j])
        fields_out = self.apply(fields_in)
        bar = abs(fields_out[0]) ** 2
        cross = abs(fields_out[1]) ** 2
        return cross, bar

    def set_cross(self) -> None:
        """Configure switch for full cross state (all power to cross port)."""
        self.params.phi = np.pi / 2
        self.params.theta = np.pi / 4

    def set_bar(self) -> None:
        """Configure switch for full bar state (all power to bar port)."""
        self.params.phi = 0.0
        self.params.theta = 0.0

    def is_unitary(self) -> bool:
        """Check if transfer matrix is approximately unitary (lossless)."""
        U = self.transfer_matrix()
        product = U @ U.conj().T
        return np.allclose(product, np.eye(2), atol=1e-6)
