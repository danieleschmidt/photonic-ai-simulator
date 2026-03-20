"""Tests for photonic-ai-simulator."""
import numpy as np
import pytest
from photonic_ai.waveguide import WaveguideSimulator, MachZehnderSwitch
from photonic_ai.accelerator import PhotonicAccelerator, AccelerationBenchmark


class TestWaveguideSimulator:
    def test_propagation_constant(self):
        wg = WaveguideSimulator(length_um=100.0, n_eff=2.4)
        beta = wg.propagation_constant()
        assert beta > 0

    def test_propagate_preserves_phase(self):
        wg = WaveguideSimulator(length_um=10.0, loss_db_cm=0.0)
        E_out = wg.propagate(1.0 + 0j)
        # No loss: power conserved
        assert abs(abs(E_out)**2 - 1.0) < 1e-6

    def test_insertion_loss_positive(self):
        wg = WaveguideSimulator(length_um=1000.0, n_eff=2.4, loss_db_cm=1.0)
        loss = wg.insertion_loss_db()
        assert loss > 0

    def test_transfer_matrix_shape(self):
        wg = WaveguideSimulator(length_um=50.0)
        T = wg.transfer_matrix()
        assert T.shape == (2, 2)


class TestMachZehnderSwitch:
    def test_bar_state_zero_phase(self):
        mzi = MachZehnderSwitch()
        mzi.set_phase(0.0)
        t_bar, t_cross = mzi.transmission()
        assert abs(t_bar - 1.0) < 1e-10
        assert abs(t_cross) < 1e-10

    def test_cross_state_pi_phase(self):
        mzi = MachZehnderSwitch()
        mzi.set_phase(np.pi)
        t_bar, t_cross = mzi.transmission()
        assert abs(t_bar) < 1e-10
        assert abs(t_cross - 1.0) < 1e-10

    def test_power_conservation(self):
        mzi = MachZehnderSwitch()
        for phi in [0, np.pi/4, np.pi/2, np.pi]:
            mzi.set_phase(phi)
            t_bar, t_cross = mzi.transmission()
            assert abs(t_bar + t_cross - 1.0) < 1e-10

    def test_apply_fields(self):
        mzi = MachZehnderSwitch()
        mzi.set_phase(np.pi / 2)
        E_bar, E_cross = mzi.apply(1.0 + 0j)
        total_power = abs(E_bar)**2 + abs(E_cross)**2
        assert abs(total_power - 1.0) < 1e-10

    def test_voltage_modulation(self):
        mzi = MachZehnderSwitch()
        mzi.set_voltage(0.0)
        t_bar0, _ = mzi.transmission()
        mzi.set_voltage(10.0)
        t_bar10, _ = mzi.transmission()
        assert t_bar0 != t_bar10


class TestPhotonicAccelerator:
    def test_matvec_shape(self):
        acc = PhotonicAccelerator(8)
        x = np.random.randn(8)
        y = acc.matvec(x)
        assert y.shape == (8,)

    def test_matmul_correctness(self):
        acc = PhotonicAccelerator(4)
        A = np.eye(4)
        acc.load_matrix(A)
        B = np.random.randn(4, 4)
        C = acc.matmul(B)
        np.testing.assert_allclose(C, B, atol=1e-10)

    def test_batch_matvec(self):
        acc = PhotonicAccelerator(8)
        A = np.random.randn(8, 8)
        acc.load_matrix(A)
        X = np.random.randn(5, 8)
        Y = acc.batch_matvec(X)
        assert Y.shape == (5, 8)

    def test_demo_64x64(self):
        bench = AccelerationBenchmark(size=64)
        msg = bench.demo_64x64()
        assert "64x64" in msg
        assert "error=" in msg


class TestAccelerationBenchmark:
    def test_run_keys(self):
        bench = AccelerationBenchmark(size=8, repeats=5)
        results = bench.run()
        assert "size" in results
        assert "photonic_s" in results
        assert "electronic_s" in results
        assert results["correct"]

    def test_correctness(self):
        bench = AccelerationBenchmark(size=16, repeats=3)
        results = bench.run()
        assert results["correct"]
