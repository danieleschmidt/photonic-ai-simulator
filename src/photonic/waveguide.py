"""WaveguideSimulator: propagation matrix method for optical waveguides."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class WaveguideParams:
    length: float          # meters
    refractive_index: float = 1.5
    loss_db_per_m: float = 0.0
    wavelength: float = 1550e-9  # meters (1550 nm default)


class WaveguideSimulator:
    """Simulate optical field propagation through a waveguide using the transfer matrix method."""

    def __init__(self, params: WaveguideParams):
        self.params = params

    @property
    def propagation_constant(self) -> float:
        """Phase propagation constant β = 2π * n / λ."""
        return 2 * np.pi * self.params.refractive_index / self.params.wavelength

    @property
    def loss_per_meter(self) -> float:
        """Field loss coefficient α from dB/m spec."""
        return self.params.loss_db_per_m / (20 * np.log10(np.e))

    def transfer_matrix(self) -> np.ndarray:
        """2×2 complex transfer matrix for the waveguide segment."""
        beta = self.propagation_constant
        alpha = self.loss_per_meter
        L = self.params.length
        phase = np.exp(1j * beta * L - alpha * L)
        return np.array([[phase, 0], [0, np.conj(phase)]], dtype=complex)

    def propagate(self, field_in: complex) -> complex:
        """Propagate a complex field amplitude through the waveguide."""
        beta = self.propagation_constant
        alpha = self.loss_per_meter
        L = self.params.length
        return field_in * np.exp(1j * beta * L - alpha * L)

    def phase_shift(self) -> float:
        """Total phase accumulated over the waveguide length (radians)."""
        return self.propagation_constant * self.params.length

    def power_transmission(self) -> float:
        """Power transmission ratio (0–1) accounting for loss."""
        alpha = self.loss_per_meter
        L = self.params.length
        return np.exp(-2 * alpha * L)

    def cascade(self, other: "WaveguideSimulator") -> "WaveguideSimulator":
        """Return a new WaveguideSimulator representing two waveguides in series."""
        p = self.params
        total_length = p.length + other.params.length
        avg_n = (p.refractive_index * p.length + other.params.refractive_index * other.params.length) / total_length
        total_loss = p.loss_db_per_m * p.length + other.params.loss_db_per_m * other.params.length
        avg_loss = total_loss / total_length
        return WaveguideSimulator(WaveguideParams(
            length=total_length,
            refractive_index=avg_n,
            loss_db_per_m=avg_loss,
            wavelength=p.wavelength,
        ))
