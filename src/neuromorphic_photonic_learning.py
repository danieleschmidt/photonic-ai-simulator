"""
Neuromorphic Photonic Learning: Novel Brain-Inspired Optical Computing.

This module implements revolutionary neuromorphic principles in photonic neural networks,
enabling spike-based temporal processing and synaptic plasticity in optical domain.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

try:
    from .core import PhotonicProcessor, WavelengthConfig, ThermalConfig
    from .models import PhotonicNeuralNetwork, LayerConfig
    from .optimization import OptimizationConfig
    from .research_innovations import QuantumEnhancedPhotonicProcessor
except ImportError:
    from core import PhotonicProcessor, WavelengthConfig, ThermalConfig
    from models import PhotonicNeuralNetwork, LayerConfig
    from optimization import OptimizationConfig
    from research_innovations import QuantumEnhancedPhotonicProcessor

logger = logging.getLogger(__name__)


class SpikeEncoding(Enum):
    """Spike encoding methods for neuromorphic processing."""
    RATE_BASED = "rate_based"
    TEMPORAL = "temporal"
    POPULATION = "population"
    PHASE = "phase"


@dataclass
class NeuromorphicConfig:
    """Configuration for neuromorphic photonic processing."""
    spike_encoding: SpikeEncoding = SpikeEncoding.TEMPORAL
    membrane_time_constant: float = 1e-9  # ns (optical response time)
    refractory_period: float = 0.5e-9     # ns (minimum spike interval)
    threshold_adaptation: bool = True      # Enable adaptive thresholds
    synaptic_plasticity: bool = True      # Enable STDP-like learning
    temporal_window: float = 10e-9        # ns (integration window)
    max_spike_rate: float = 1e12          # Hz (maximum optical spike rate)
    plasticity_learning_rate: float = 0.001


@dataclass
class SynapticState:
    """State representation for neuromorphic synapses."""
    weight: float
    last_spike_time: float = -np.inf
    trace: float = 0.0  # Synaptic trace for plasticity
    efficacy: float = 1.0  # Dynamic synaptic efficacy


class NeuromorphicPhotonicNeuron:
    """
    Spike-based photonic neuron with temporal dynamics.
    
    Implements leaky integrate-and-fire dynamics using optical intensity
    modulation and phase-based spike timing.
    """
    
    def __init__(self, neuron_id: int, config: NeuromorphicConfig):
        self.neuron_id = neuron_id
        self.config = config
        
        # Neuron state
        self.membrane_potential = 0.0
        self.threshold = 1.0
        self.last_spike_time = -np.inf
        self.spike_count = 0
        
        # Adaptation parameters
        self.threshold_adaptation_rate = 0.01
        self.base_threshold = 1.0
        
        # Input synapses
        self.synapses: Dict[int, SynapticState] = {}
        
        # Spike history for temporal coding
        self.spike_history: List[float] = []
        
    def add_synapse(self, source_id: int, weight: float = 1.0):
        """Add synaptic connection from source neuron."""
        self.synapses[source_id] = SynapticState(weight=weight)
        
    def integrate(self, inputs: Dict[int, float], current_time: float) -> bool:
        """
        Integrate synaptic inputs and determine spike generation.
        
        Args:
            inputs: Dict mapping source neuron IDs to spike intensities
            current_time: Current simulation time (ns)
            
        Returns:
            bool: True if neuron spikes
        """
        # Check refractory period
        if current_time - self.last_spike_time < self.config.refractory_period:
            return False
            
        # Membrane potential decay
        time_since_spike = current_time - self.last_spike_time
        decay_factor = np.exp(-time_since_spike / self.config.membrane_time_constant)
        self.membrane_potential *= decay_factor
        
        # Integrate synaptic inputs
        total_input = 0.0
        for source_id, intensity in inputs.items():
            if source_id in self.synapses:
                synapse = self.synapses[source_id]
                # Apply synaptic weight and efficacy
                weighted_input = intensity * synapse.weight * synapse.efficacy
                total_input += weighted_input
                
                # Update synaptic trace for plasticity
                if self.config.synaptic_plasticity:
                    synapse.trace = synapse.trace * 0.95 + 0.05 * intensity
                    
        self.membrane_potential += total_input
        
        # Check for spike generation
        if self.membrane_potential >= self.threshold:
            self._generate_spike(current_time)
            return True
            
        return False
    
    def _generate_spike(self, spike_time: float):
        """Generate spike and update neuron state."""
        self.last_spike_time = spike_time
        self.spike_count += 1
        self.spike_history.append(spike_time)
        
        # Reset membrane potential
        self.membrane_potential = 0.0
        
        # Threshold adaptation
        if self.config.threshold_adaptation:
            self.threshold += self.threshold_adaptation_rate
            
        # Limit spike history size
        if len(self.spike_history) > 1000:
            self.spike_history = self.spike_history[-500:]
    
    def apply_stdp(self, pre_spike_times: List[float], learning_rate: float = None):
        """
        Apply Spike-Timing Dependent Plasticity.
        
        Args:
            pre_spike_times: List of presynaptic spike times
            learning_rate: Learning rate for plasticity (optional)
        """
        if not self.config.synaptic_plasticity:
            return
            
        lr = learning_rate or self.config.plasticity_learning_rate
        
        # STDP time windows
        tau_plus = 20e-9   # ns (potentiation window)
        tau_minus = 20e-9  # ns (depression window)
        
        for pre_time in pre_spike_times:
            for post_time in self.spike_history:
                dt = post_time - pre_time
                
                # Potentiation (pre before post)
                if 0 < dt <= tau_plus:
                    weight_change = lr * np.exp(-dt / tau_plus)
                    for synapse in self.synapses.values():
                        synapse.weight = np.clip(synapse.weight + weight_change, 0.0, 2.0)
                
                # Depression (post before pre)
                elif -tau_minus <= dt < 0:
                    weight_change = -lr * np.exp(dt / tau_minus)
                    for synapse in self.synapses.values():
                        synapse.weight = np.clip(synapse.weight + weight_change, 0.0, 2.0)


class NeuromorphicPhotonicNetwork:
    """
    Neuromorphic photonic neural network with spike-based processing.
    
    Implements temporal coding and synaptic plasticity in the optical domain
    for brain-inspired computing with sub-nanosecond dynamics.
    """
    
    def __init__(self, 
                 layer_sizes: List[int],
                 config: NeuromorphicConfig = None,
                 wavelength_config: WavelengthConfig = None):
        self.config = config or NeuromorphicConfig()
        self.wavelength_config = wavelength_config or WavelengthConfig()
        self.layer_sizes = layer_sizes
        
        # Create neuromorphic neurons
        self.neurons: Dict[int, NeuromorphicPhotonicNeuron] = {}
        neuron_id = 0
        
        for layer_idx, layer_size in enumerate(layer_sizes):
            for neuron_idx in range(layer_size):
                self.neurons[neuron_id] = NeuromorphicPhotonicNeuron(
                    neuron_id, self.config
                )
                neuron_id += 1
        
        # Connect layers
        self._connect_layers()
        
        # Performance metrics
        self.inference_times = []
        self.spike_rates = []
        
    def _connect_layers(self):
        """Connect neurons between adjacent layers."""
        neuron_id = 0
        
        for layer_idx in range(len(self.layer_sizes) - 1):
            current_layer_size = self.layer_sizes[layer_idx]
            next_layer_size = self.layer_sizes[layer_idx + 1]
            
            # Current layer neurons
            current_layer_start = neuron_id
            neuron_id += current_layer_size
            
            # Next layer neurons
            next_layer_start = neuron_id
            
            # Connect each neuron in current layer to all neurons in next layer
            for i in range(current_layer_size):
                for j in range(next_layer_size):
                    source_id = current_layer_start + i
                    target_id = next_layer_start + j
                    
                    # Initialize with small random weights
                    weight = np.random.normal(0.5, 0.1)
                    self.neurons[target_id].add_synapse(source_id, weight)
    
    def encode_input(self, data: np.ndarray, encoding_method: SpikeEncoding = None) -> Dict[int, List[float]]:
        """
        Encode input data into spike trains.
        
        Args:
            data: Input data array
            encoding_method: Spike encoding method
            
        Returns:
            Dict mapping neuron IDs to spike time lists
        """
        method = encoding_method or self.config.spike_encoding
        spike_trains = {}
        
        input_size = self.layer_sizes[0]
        
        if method == SpikeEncoding.RATE_BASED:
            # Rate-based encoding: spike frequency proportional to input intensity
            for i in range(min(len(data), input_size)):
                intensity = np.clip(data[i], 0, 1)
                spike_rate = intensity * self.config.max_spike_rate
                
                # Generate Poisson spike train
                num_spikes = np.random.poisson(spike_rate * self.config.temporal_window)
                spike_times = np.sort(np.random.uniform(
                    0, self.config.temporal_window, num_spikes
                ))
                spike_trains[i] = spike_times.tolist()
                
        elif method == SpikeEncoding.TEMPORAL:
            # Temporal encoding: spike timing encodes information
            for i in range(min(len(data), input_size)):
                intensity = np.clip(data[i], 0.01, 1)  # Avoid zero
                # Earlier spikes for higher intensities
                spike_time = (1 - intensity) * self.config.temporal_window
                spike_trains[i] = [spike_time]
                
        elif method == SpikeEncoding.PHASE:
            # Phase encoding: use optical phase to encode information
            for i in range(min(len(data), input_size)):
                phase = data[i] * 2 * np.pi
                # Convert phase to spike timing
                spike_time = (phase / (2 * np.pi)) * self.config.temporal_window
                spike_trains[i] = [spike_time]
                
        return spike_trains
    
    def forward(self, spike_trains: Dict[int, List[float]], 
                simulation_time: float = None) -> Tuple[Dict[int, List[float]], Dict[str, float]]:
        """
        Forward pass through neuromorphic network.
        
        Args:
            spike_trains: Input spike trains
            simulation_time: Total simulation time (ns)
            
        Returns:
            Tuple of (output spike trains, performance metrics)
        """
        start_time = time.time()
        sim_time = simulation_time or self.config.temporal_window
        
        # Time discretization
        dt = 0.01e-9  # 10 ps resolution
        time_steps = int(sim_time / dt)
        
        # Track spike activity
        all_spike_times = {neuron_id: [] for neuron_id in self.neurons.keys()}
        
        # Initialize input spikes
        for neuron_id, spikes in spike_trains.items():
            all_spike_times[neuron_id] = spikes.copy()
        
        # Simulate network dynamics
        for step in range(time_steps):
            current_time = step * dt
            
            # Process each neuron
            for neuron_id, neuron in self.neurons.items():
                # Collect inputs from connected neurons
                inputs = {}
                
                for source_id in neuron.synapses.keys():
                    # Check for recent spikes from source
                    source_spikes = all_spike_times.get(source_id, [])
                    intensity = 0.0
                    
                    for spike_time in source_spikes:
                        # Exponential decay of spike influence
                        time_diff = current_time - spike_time
                        if 0 <= time_diff <= self.config.temporal_window:
                            intensity += np.exp(-time_diff / (self.config.temporal_window * 0.1))
                    
                    if intensity > 0:
                        inputs[source_id] = intensity
                
                # Integrate and potentially spike
                if neuron.integrate(inputs, current_time):
                    all_spike_times[neuron_id].append(current_time)
        
        # Extract output layer spikes
        output_start = sum(self.layer_sizes[:-1])
        output_spikes = {
            neuron_id - output_start: spikes 
            for neuron_id, spikes in all_spike_times.items()
            if neuron_id >= output_start
        }
        
        # Calculate performance metrics
        inference_time = time.time() - start_time
        total_spikes = sum(len(spikes) for spikes in all_spike_times.values())
        spike_rate = total_spikes / (sim_time * 1e-9)  # Hz
        
        metrics = {
            'inference_time_ms': inference_time * 1000,
            'simulation_time_ns': sim_time * 1e9,
            'total_spikes': total_spikes,
            'average_spike_rate_hz': spike_rate,
            'temporal_precision_ps': dt * 1e12,
            'synaptic_operations': sum(len(n.synapses) for n in self.neurons.values())
        }
        
        self.inference_times.append(inference_time)
        self.spike_rates.append(spike_rate)
        
        return output_spikes, metrics
    
    def train_stdp(self, 
                   input_data: np.ndarray, 
                   target_data: np.ndarray = None,
                   num_epochs: int = 100) -> Dict[str, List[float]]:
        """
        Train network using Spike-Timing Dependent Plasticity.
        
        Args:
            input_data: Training input data
            target_data: Training target data (optional for unsupervised)
            num_epochs: Number of training epochs
            
        Returns:
            Training history with metrics
        """
        history = {
            'spike_rates': [],
            'synaptic_changes': [],
            'temporal_coherence': []
        }
        
        for epoch in range(num_epochs):
            epoch_metrics = {
                'spike_rate': 0.0,
                'weight_changes': 0.0,
                'coherence': 0.0
            }
            
            # Process each training sample
            for sample in input_data:
                # Encode input
                spike_trains = self.encode_input(sample)
                
                # Forward pass
                output_spikes, metrics = self.forward(spike_trains)
                
                # Apply STDP to all neurons
                for neuron_id, neuron in self.neurons.items():
                    # Collect presynaptic spike times
                    pre_spikes = []
                    for source_id in neuron.synapses.keys():
                        if source_id in spike_trains:
                            pre_spikes.extend(spike_trains[source_id])
                    
                    # Apply STDP
                    if pre_spikes:
                        neuron.apply_stdp(pre_spikes)
                
                # Update metrics
                epoch_metrics['spike_rate'] += metrics['average_spike_rate_hz']
            
            # Average metrics over samples
            num_samples = len(input_data)
            epoch_metrics['spike_rate'] /= num_samples
            
            # Calculate synaptic weight changes
            total_weight_change = 0.0
            for neuron in self.neurons.values():
                for synapse in neuron.synapses.values():
                    total_weight_change += abs(synapse.weight - 0.5)  # Relative to initial
            epoch_metrics['weight_changes'] = total_weight_change
            
            # Calculate temporal coherence (spike synchrony measure)
            all_spikes = []
            for neuron in self.neurons.values():
                all_spikes.extend(neuron.spike_history[-10:])  # Recent spikes
            
            if len(all_spikes) > 1:
                spike_array = np.array(all_spikes)
                coherence = 1.0 / (1.0 + np.std(spike_array))  # Higher coherence = lower std
                epoch_metrics['coherence'] = coherence
            
            # Store history
            history['spike_rates'].append(epoch_metrics['spike_rate'])
            history['synaptic_changes'].append(epoch_metrics['weight_changes'])
            history['temporal_coherence'].append(epoch_metrics['coherence'])
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Spike Rate={epoch_metrics['spike_rate']:.2e} Hz, "
                           f"Weight Changes={epoch_metrics['weight_changes']:.3f}, "
                           f"Coherence={epoch_metrics['coherence']:.3f}")
        
        return history
    
    def get_network_state(self) -> Dict[str, Any]:
        """Get comprehensive network state for analysis."""
        state = {
            'num_neurons': len(self.neurons),
            'total_synapses': sum(len(n.synapses) for n in self.neurons.values()),
            'average_threshold': np.mean([n.threshold for n in self.neurons.values()]),
            'total_spikes': sum(n.spike_count for n in self.neurons.values()),
            'average_spike_rate': np.mean(self.spike_rates) if self.spike_rates else 0.0,
            'inference_time_ms': np.mean(self.inference_times) * 1000 if self.inference_times else 0.0,
            'synaptic_weights': {
                'mean': np.mean([s.weight for n in self.neurons.values() for s in n.synapses.values()]),
                'std': np.std([s.weight for n in self.neurons.values() for s in n.synapses.values()]),
                'min': np.min([s.weight for n in self.neurons.values() for s in n.synapses.values()]),
                'max': np.max([s.weight for n in self.neurons.values() for s in n.synapses.values()])
            }
        }
        
        return state


def create_neuromorphic_benchmark(task: str = "temporal_pattern") -> NeuromorphicPhotonicNetwork:
    """
    Create neuromorphic network optimized for specific benchmarking tasks.
    
    Args:
        task: Benchmark task ("temporal_pattern", "spike_classification", "adaptive_learning")
        
    Returns:
        Configured neuromorphic photonic network
    """
    if task == "temporal_pattern":
        # Optimized for temporal pattern recognition
        config = NeuromorphicConfig(
            spike_encoding=SpikeEncoding.TEMPORAL,
            membrane_time_constant=0.5e-9,
            synaptic_plasticity=True,
            temporal_window=20e-9
        )
        return NeuromorphicPhotonicNetwork([100, 50, 10], config)
        
    elif task == "spike_classification":
        # Optimized for spike rate-based classification
        config = NeuromorphicConfig(
            spike_encoding=SpikeEncoding.RATE_BASED,
            membrane_time_constant=1e-9,
            synaptic_plasticity=True,
            max_spike_rate=5e11
        )
        return NeuromorphicPhotonicNetwork([784, 100, 10], config)
        
    elif task == "adaptive_learning":
        # Optimized for online adaptive learning
        config = NeuromorphicConfig(
            spike_encoding=SpikeEncoding.PHASE,
            threshold_adaptation=True,
            plasticity_learning_rate=0.005,
            temporal_window=50e-9
        )
        return NeuromorphicPhotonicNetwork([50, 100, 50, 10], config)
        
    else:
        raise ValueError(f"Unknown benchmark task: {task}")


# Export key components
__all__ = [
    'NeuromorphicConfig',
    'SpikeEncoding',
    'NeuromorphicPhotonicNeuron',
    'NeuromorphicPhotonicNetwork',
    'create_neuromorphic_benchmark'
]