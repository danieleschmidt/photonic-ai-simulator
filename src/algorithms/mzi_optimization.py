"""
Mach-Zehnder Interferometer (MZI) Optimization Algorithms

Implements foundational algorithms for optimizing MZI networks in photonic neural networks,
including phase shifter configuration, thermal drift compensation, and coupling optimization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
import logging

logger = logging.getLogger(__name__)


@dataclass
class MZIConfiguration:
    """Configuration for MZI network optimization."""
    num_mzis: int
    target_transmission: np.ndarray
    power_budget_mw: float = 500.0
    thermal_stability_requirement: float = 0.01  # 1% stability
    fabrication_tolerance: float = 0.1  # 10% tolerance
    wavelength_range_nm: Tuple[float, float] = (1530.0, 1570.0)


class MZIOptimizer:
    """
    Optimizes MZI network configurations for photonic neural networks.
    
    Implements gradient-based and evolutionary algorithms for finding optimal
    phase shifter settings that achieve target transmission characteristics.
    """
    
    def __init__(self, config: MZIConfiguration):
        """Initialize MZI optimizer."""
        self.config = config
        self.optimization_history = []
        
    def optimize_transmission_matrix(self, 
                                   target_matrix: np.ndarray,
                                   algorithm: str = "gradient_descent") -> Dict[str, Any]:
        """
        Optimize MZI network to achieve target transmission matrix.
        
        Args:
            target_matrix: Target transmission matrix (complex)
            algorithm: Optimization algorithm ("gradient_descent", "evolutionary")
            
        Returns:
            Optimization results with phase settings
        """
        logger.info(f"Starting MZI optimization with {algorithm} algorithm")
        
        # Initialize phase settings
        num_phases = self._calculate_phase_count(target_matrix.shape)
        initial_phases = np.random.uniform(0, 2*np.pi, num_phases)
        
        if algorithm == "gradient_descent":
            result = self._gradient_based_optimization(target_matrix, initial_phases)
        elif algorithm == "evolutionary":
            result = self._evolutionary_optimization(target_matrix, initial_phases)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        # Validate result
        validation = self._validate_solution(result["phases"], target_matrix)
        result.update(validation)
        
        return result
    
    def optimize_power_efficiency(self, 
                                 transmission_matrix: np.ndarray,
                                 power_constraint: float) -> Dict[str, Any]:
        """
        Optimize for minimum power consumption while maintaining transmission.
        
        Args:
            transmission_matrix: Desired transmission matrix
            power_constraint: Maximum power consumption (mW)
            
        Returns:
            Power-optimized phase settings
        """
        def power_objective(phases):
            """Objective function minimizing power consumption."""
            T_actual = self._compute_transmission_matrix(phases, transmission_matrix.shape)
            fidelity = np.abs(np.trace(T_actual @ transmission_matrix.conj().T))**2
            fidelity /= (np.trace(T_actual @ T_actual.conj().T) * 
                        np.trace(transmission_matrix @ transmission_matrix.conj().T))
            
            # Power consumption model (15mW per phase shifter)
            power_consumption = np.sum(np.abs(phases)**2) * 15.0
            
            # Penalty for exceeding power budget
            power_penalty = max(0, power_consumption - power_constraint) * 1000
            
            return -(fidelity - 0.01 * power_consumption) + power_penalty
        
        num_phases = self._calculate_phase_count(transmission_matrix.shape)
        initial_phases = np.random.uniform(0, 2*np.pi, num_phases)
        
        result = minimize(
            power_objective,
            initial_phases,
            method='L-BFGS-B',
            bounds=[(0, 2*np.pi)] * num_phases,
            options={'maxiter': 1000}
        )
        
        optimized_T = self._compute_transmission_matrix(result.x, transmission_matrix.shape)
        power_consumption = np.sum(np.abs(result.x)**2) * 15.0
        
        return {
            "phases": result.x,
            "transmission_matrix": optimized_T,
            "power_consumption_mw": power_consumption,
            "fidelity": self._compute_fidelity(optimized_T, transmission_matrix),
            "optimization_success": result.success
        }
    
    def compensate_thermal_drift(self, 
                                original_phases: np.ndarray,
                                temperature_change: float) -> np.ndarray:
        """
        Compensate for thermal drift in MZI phase shifters.
        
        Args:
            original_phases: Original phase settings
            temperature_change: Temperature change in Kelvin
            
        Returns:
            Compensated phase settings
        """
        # Thermal coefficient for silicon photonics: 1.8e-4 /K
        thermal_coefficient = 1.8e-4
        
        # Phase drift model
        phase_drift = thermal_coefficient * temperature_change * np.pi * 2
        
        # Compensate by adjusting phases
        compensated_phases = original_phases - phase_drift
        
        # Wrap phases to [0, 2π]
        compensated_phases = np.mod(compensated_phases, 2*np.pi)
        
        logger.info(f"Applied thermal compensation for {temperature_change:.2f}K change")
        
        return compensated_phases
    
    def _gradient_based_optimization(self, 
                                   target_matrix: np.ndarray,
                                   initial_phases: np.ndarray) -> Dict[str, Any]:
        """Gradient-based optimization using finite differences."""
        def objective(phases):
            T_actual = self._compute_transmission_matrix(phases, target_matrix.shape)
            return -self._compute_fidelity(T_actual, target_matrix)
        
        result = minimize(
            objective,
            initial_phases,
            method='BFGS',
            options={'gtol': 1e-6, 'maxiter': 500}
        )
        
        optimized_T = self._compute_transmission_matrix(result.x, target_matrix.shape)
        
        return {
            "phases": result.x,
            "transmission_matrix": optimized_T,
            "fidelity": -result.fun,
            "iterations": result.nit,
            "optimization_success": result.success
        }
    
    def _evolutionary_optimization(self, 
                                 target_matrix: np.ndarray,
                                 initial_phases: np.ndarray) -> Dict[str, Any]:
        """Evolutionary optimization for global search."""
        def objective(phases):
            T_actual = self._compute_transmission_matrix(phases, target_matrix.shape)
            return -self._compute_fidelity(T_actual, target_matrix)
        
        bounds = [(0, 2*np.pi) for _ in range(len(initial_phases))]
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=200,
            popsize=15,
            atol=1e-6
        )
        
        optimized_T = self._compute_transmission_matrix(result.x, target_matrix.shape)
        
        return {
            "phases": result.x,
            "transmission_matrix": optimized_T,
            "fidelity": -result.fun,
            "iterations": result.nit,
            "optimization_success": result.success
        }
    
    def _compute_transmission_matrix(self, 
                                   phases: np.ndarray,
                                   matrix_shape: Tuple[int, int]) -> np.ndarray:
        """Compute transmission matrix from phase settings."""
        N = matrix_shape[0]
        
        # Build transmission matrix using triangular decomposition
        T = np.eye(N, dtype=complex)
        phase_idx = 0
        
        for i in range(N):
            for j in range(i+1, N):
                if phase_idx < len(phases):
                    # MZI coupling matrix
                    theta = phases[phase_idx]
                    phi = phases[phase_idx + 1] if phase_idx + 1 < len(phases) else 0
                    
                    mzi_matrix = np.array([
                        [np.cos(theta) * np.exp(1j * phi), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta) * np.exp(1j * phi)]
                    ])
                    
                    # Apply to appropriate submatrix
                    temp_T = T.copy()
                    temp_T[i:i+2, j:j+2] = mzi_matrix @ temp_T[i:i+2, j:j+2]
                    T = temp_T
                    
                    phase_idx += 2
        
        return T
    
    def _compute_fidelity(self, 
                         actual_matrix: np.ndarray,
                         target_matrix: np.ndarray) -> float:
        """Compute fidelity between actual and target transmission matrices."""
        # Normalized overlap fidelity
        numerator = np.abs(np.trace(actual_matrix @ target_matrix.conj().T))**2
        denominator = (np.trace(actual_matrix @ actual_matrix.conj().T) * 
                      np.trace(target_matrix @ target_matrix.conj().T))
        
        return numerator / (denominator + 1e-10)
    
    def _calculate_phase_count(self, matrix_shape: Tuple[int, int]) -> int:
        """Calculate number of phase parameters needed."""
        N = matrix_shape[0]
        return N * (N - 1)  # 2 phases per MZI, N(N-1)/2 MZIs
    
    def _validate_solution(self, 
                          phases: np.ndarray,
                          target_matrix: np.ndarray) -> Dict[str, Any]:
        """Validate optimized solution."""
        T_actual = self._compute_transmission_matrix(phases, target_matrix.shape)
        fidelity = self._compute_fidelity(T_actual, target_matrix)
        
        # Power consumption
        power_consumption = np.sum(np.abs(phases)**2) * 15.0
        
        # Thermal stability (phase sensitivity)
        phase_sensitivity = np.std(phases)
        
        return {
            "validation_fidelity": fidelity,
            "power_consumption_mw": power_consumption,
            "phase_sensitivity": phase_sensitivity,
            "meets_power_budget": power_consumption <= self.config.power_budget_mw,
            "thermal_stability": phase_sensitivity <= self.config.thermal_stability_requirement
        }


def create_identity_target(size: int) -> np.ndarray:
    """Create identity transmission matrix target."""
    return np.eye(size, dtype=complex)


def create_hadamard_target(size: int) -> np.ndarray:
    """Create Hadamard-like transformation target."""
    if size == 2:
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    else:
        # Generalized Hadamard for larger sizes
        H = np.ones((size, size), dtype=complex)
        for i in range(size):
            for j in range(size):
                H[i, j] *= (-1)**(bin(i & j).count('1'))
        return H / np.sqrt(size)


def create_fourier_target(size: int) -> np.ndarray:
    """Create discrete Fourier transform matrix target."""
    n = np.arange(size)
    k = n.reshape((size, 1))
    M = np.exp(-2j * np.pi * k * n / size)
    return M / np.sqrt(size)