"""Waveguide simulator using transfer matrix method."""
import numpy as np
from typing import Tuple


class WaveguideSimulator:
    """Simulates optical field propagation through a waveguide segment."""

    def __init__(self, length_um: float, n_eff: float = 2.4, loss_db_cm: float = 0.5):
        self.length_um = length_um
        self.n_eff = n_eff
        self.loss_db_cm = loss_db_cm
        self.wavelength_um = 1.55

    @property
    def _alpha(self) -> float:
        """Power attenuation coefficient (1/um)."""
        return self.loss_db_cm / (10 * np.log10(np.e) * 1e4)

    def propagation_constant(self) -> float:
        return 2 * np.pi * self.n_eff / self.wavelength_um

    def transfer_matrix(self) -> np.ndarray:
        beta = self.propagation_constant()
        phase = beta * self.length_um
        att = np.exp(-self._alpha * self.length_um / 2)
        return att * np.array([
            [np.exp(1j * phase), 0],
            [0, np.exp(-1j * phase)]
        ])

    def propagate(self, E_in: complex) -> complex:
        """Propagate input field through waveguide."""
        T = self.transfer_matrix()
        return complex(T[0, 0] * E_in)

    def insertion_loss_db(self) -> float:
        E_out = self.propagate(1.0 + 0j)
        return -10 * np.log10(abs(E_out) ** 2)


class MachZehnderSwitch:
    """Mach-Zehnder interferometer optical switch."""

    def __init__(self, arm_length_um: float = 100.0, n_eff: float = 2.4):
        self.arm_length_um = arm_length_um
        self.n_eff = n_eff
        self.wavelength_um = 1.55
        self._delta_n: float = 0.0

    def set_phase(self, delta_phi: float) -> None:
        """Set differential phase shift between arms."""
        self._delta_phi = delta_phi

    def set_voltage(self, voltage: float, dn_dV: float = 1e-4) -> None:
        """Electro-optic phase modulation via voltage."""
        self._delta_n = dn_dV * voltage
        beta_eo = 2 * np.pi * self._delta_n / self.wavelength_um
        self._delta_phi = beta_eo * self.arm_length_um

    def transmission(self) -> Tuple[float, float]:
        """Return (bar_port, cross_port) power transmission."""
        phi = getattr(self, "_delta_phi", 0.0)
        t_bar = np.cos(phi / 2) ** 2
        t_cross = np.sin(phi / 2) ** 2
        return float(t_bar), float(t_cross)

    def apply(self, E_in: complex) -> Tuple[complex, complex]:
        """Apply MZI to input field. Returns (E_bar, E_cross)."""
        phi = getattr(self, "_delta_phi", 0.0)
        E_bar = E_in * np.cos(phi / 2)
        E_cross = E_in * 1j * np.sin(phi / 2)
        return complex(E_bar), complex(E_cross)
