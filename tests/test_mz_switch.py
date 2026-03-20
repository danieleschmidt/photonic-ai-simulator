"""Tests for MachZehnderSwitch."""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from photonic.mz_switch import MachZehnderSwitch, MZParams


def make_mz(phi=0.0, theta=np.pi/4, loss=0.0):
    return MachZehnderSwitch(MZParams(phi=phi, theta=theta, loss=loss))


def test_transfer_matrix_shape():
    mz = make_mz()
    assert mz.transfer_matrix().shape == (2, 2)


def test_lossless_unitary():
    mz = make_mz()
    assert mz.is_unitary()


def test_lossy_not_unitary():
    mz = make_mz(loss=0.2)
    assert not mz.is_unitary()


def test_splitting_50_50():
    mz = make_mz(theta=np.pi/4)
    cross, bar = mz.splitting_ratio()
    assert abs(cross - 0.5) < 1e-9
    assert abs(bar - 0.5) < 1e-9


def test_bar_state():
    mz = make_mz()
    mz.set_bar()
    cross, bar = mz.splitting_ratio()
    assert bar > 0.99


def test_apply_returns_vector():
    mz = make_mz()
    out = mz.apply(np.array([1.0+0j, 0.0+0j]))
    assert out.shape == (2,)


def test_power_conservation_lossless():
    mz = make_mz()
    fields_in = np.array([0.6+0.8j, 0.3-0.4j])
    fields_out = mz.apply(fields_in)
    power_in = np.sum(np.abs(fields_in)**2)
    power_out = np.sum(np.abs(fields_out)**2)
    assert abs(power_in - power_out) < 1e-9
