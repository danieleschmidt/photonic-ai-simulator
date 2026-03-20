"""Tests for PhotonicAccelerator."""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from photonic.accelerator import PhotonicAccelerator


def test_init():
    acc = PhotonicAccelerator(4)
    assert acc.size == 4


def test_num_switches():
    acc = PhotonicAccelerator(4)
    # triangular mesh: 3+2+1 = 6 switches
    assert acc.num_switches() == 6


def test_matrix_multiply_shape():
    acc = PhotonicAccelerator(4)
    A = np.eye(4)
    B = np.random.randn(4, 4)
    C = acc.matrix_multiply(A, B)
    assert C.shape == (4, 4)


def test_forward_shape():
    acc = PhotonicAccelerator(4)
    A = np.eye(4)
    acc.set_weights(A)
    x = np.random.randn(4)
    out = acc.forward(x)
    assert out.shape == (4,)


def test_forward_without_set_weights_raises():
    acc = PhotonicAccelerator(4)
    with pytest.raises(RuntimeError):
        acc.forward(np.ones(4))


def test_matrix_multiply_identity():
    acc = PhotonicAccelerator(4)
    A = np.eye(4)
    B = np.random.randn(4, 4)
    C = acc.matrix_multiply(A, B).real
    assert np.allclose(C, B, atol=0.1)
