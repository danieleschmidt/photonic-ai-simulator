"""Tests for WaveguideSimulator."""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from photonic.waveguide import WaveguideSimulator, WaveguideParams


def make_wg(length=1e-3, n=1.5, loss=0.0, wl=1550e-9):
    return WaveguideSimulator(WaveguideParams(length=length, refractive_index=n, loss_db_per_m=loss, wavelength=wl))


def test_propagation_constant():
    wg = make_wg(n=1.5, wl=1550e-9)
    beta = wg.propagation_constant
    expected = 2 * np.pi * 1.5 / 1550e-9
    assert abs(beta - expected) < 1.0


def test_power_transmission_lossless():
    wg = make_wg(loss=0.0)
    assert abs(wg.power_transmission() - 1.0) < 1e-9


def test_power_transmission_lossy():
    wg = make_wg(loss=1.0)
    assert wg.power_transmission() < 1.0


def test_propagate_preserves_amplitude_lossless():
    wg = make_wg(loss=0.0)
    out = wg.propagate(1.0 + 0j)
    assert abs(abs(out) - 1.0) < 1e-9


def test_propagate_applies_phase():
    wg = make_wg(length=0.0)
    out = wg.propagate(1.0 + 0j)
    assert abs(out - 1.0) < 1e-9


def test_transfer_matrix_shape():
    wg = make_wg()
    M = wg.transfer_matrix()
    assert M.shape == (2, 2)


def test_transfer_matrix_dtype():
    wg = make_wg()
    M = wg.transfer_matrix()
    assert np.iscomplexobj(M)


def test_phase_shift_positive():
    wg = make_wg(length=1e-3)
    assert wg.phase_shift() > 0


def test_cascade_length():
    wg1 = make_wg(length=1e-3)
    wg2 = make_wg(length=2e-3)
    combined = wg1.cascade(wg2)
    assert abs(combined.params.length - 3e-3) < 1e-15
