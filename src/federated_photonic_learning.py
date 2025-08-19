"""
Federated Photonic Learning Framework.

Novel distributed learning system for photonic neural networks that enables
collaborative training across multiple photonic processors while preserving
data privacy and optimizing for optical communication constraints.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
import hashlib
import time
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import pickle
import json

try:
    from .core import PhotonicProcessor, WavelengthConfig, ThermalConfig, FabricationConfig
    from .models import PhotonicNeuralNetwork, LayerConfig
    from .training import ForwardOnlyTrainer, TrainingConfig
    from .optimization import OptimizationConfig
    from .research_innovations import QuantumEnhancedPhotonicProcessor
except ImportError:
    from core import PhotonicProcessor, WavelengthConfig, ThermalConfig, FabricationConfig
    from models import PhotonicNeuralNetwork, LayerConfig
    from training import ForwardOnlyTrainer, TrainingConfig
    from optimization import OptimizationConfig
    from research_innovations import QuantumEnhancedPhotonicProcessor

logger = logging.getLogger(__name__)


@dataclass
class FederatedConfig:
    """Configuration for federated photonic learning."""
    # Federated learning parameters
    num_clients: int = 10
    rounds: int = 100
    client_fraction: float = 0.5
    local_epochs: int = 3
    
    # Optical communication parameters
    wavelength_channels: int = 8
    optical_bandwidth_ghz: float = 100.0
    transmission_loss_db: float = 0.2
    coherence_time_ns: float = 1000.0
    
    # Privacy and security
    differential_privacy: bool = True
    privacy_epsilon: float = 1.0
    secure_aggregation: bool = True
    homomorphic_encryption: bool = False
    
    # Performance optimization
    compression_ratio: float = 0.1
    quantization_bits: int = 8
    sparse_updates: bool = True
    adaptive_learning_rate: bool = True


@dataclass
class OpticalCommunicationMetrics:
    """Metrics for optical communication between photonic processors."""
    transmission_latency_ns: float
    channel_utilization: float
    signal_to_noise_ratio_db: float
    bit_error_rate: float
    power_consumption_mw: float
    thermal_crosstalk: float


class PhotonicClient:
    """
    Photonic neural network client for federated learning.
    
    Represents a single photonic processor participating in federated training
    with local data and model updates.
    """
    
    def __init__(self, 
                 client_id: str,
                 model: PhotonicNeuralNetwork,
                 local_data: Tuple[np.ndarray, np.ndarray],
                 federated_config: FederatedConfig):
        """Initialize photonic client."""
        self.client_id = client_id
        self.model = model
        self.local_data = local_data
        self.config = federated_config
        
        # Client state
        self.round_number = 0
        self.is_selected = False
        self.local_updates = {}
        
        # Communication system
        self.optical_transmitter = OpticalTransmitter(federated_config)
        self.optical_receiver = OpticalReceiver(federated_config)
        
        # Privacy mechanisms
        if federated_config.differential_privacy:
            self.privacy_mechanism = DifferentialPrivacyMechanism(federated_config.privacy_epsilon)
        else:
            self.privacy_mechanism = None
        
        # Performance tracking
        self.training_history = []
        self.communication_metrics = []
        
        logger.info(f"Photonic client {client_id} initialized")
    
    def local_training(self, global_model_weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Perform local training on client data.
        
        Args:
            global_model_weights: Global model weights from server
            
        Returns:
            Local model updates and training metrics
        """
        # Update local model with global weights
        self._update_local_model(global_model_weights)
        
        # Perform local training
        X_local, y_local = self.local_data
        
        # Configure trainer for photonic hardware
        training_config = TrainingConfig(
            learning_rate=0.001,
            batch_size=min(32, len(X_local) // 4),
            epochs=self.config.local_epochs,
            forward_only=True,  # Use forward-only training for photonic hardware
            thermal_compensation=True,
            fabrication_aware=True
        )
        
        trainer = ForwardOnlyTrainer(self.model, training_config)
        
        # Train locally
        start_time = time.time()
        training_history = trainer.train(X_local, y_local)
        training_time = time.time() - start_time
        
        # Extract model updates
        local_updates = self._extract_model_updates(global_model_weights)
        
        # Apply privacy mechanisms
        if self.privacy_mechanism:
            local_updates = self.privacy_mechanism.apply_privacy(local_updates)
        
        # Compress updates for optical transmission
        compressed_updates = self._compress_updates(local_updates)
        
        # Training metrics
        training_metrics = {
            "training_time_s": training_time,
            "final_loss": training_history["train_loss"][-1],
            "final_accuracy": training_history["train_accuracy"][-1],
            "data_size": len(X_local),
            "update_size_mb": self._compute_update_size(compressed_updates),
            "compression_ratio": len(compressed_updates) / len(local_updates) if local_updates else 1.0
        }
        
        self.training_history.append(training_metrics)
        
        return {
            "client_id": self.client_id,
            "updates": compressed_updates,
            "metrics": training_metrics,
            "round": self.round_number
        }
    
    def _update_local_model(self, global_weights: Dict[str, np.ndarray]):
        """Update local model with global weights."""
        for layer_idx, layer in enumerate(self.model.layers):
            layer_key = f"layer_{layer_idx}_weights"
            if layer_key in global_weights:
                layer.weights = global_weights[layer_key].copy()
    
    def _extract_model_updates(self, global_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract local model updates (difference from global model)."""
        updates = {}
        
        for layer_idx, layer in enumerate(self.model.layers):
            layer_key = f"layer_{layer_idx}_weights"
            if layer_key in global_weights:
                # Compute weight difference
                update = layer.weights - global_weights[layer_key]
                updates[layer_key] = update
        
        return updates
    
    def _compress_updates(self, updates: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compress model updates for efficient optical transmission."""
        compressed_updates = {}
        
        for key, update in updates.items():
            if self.config.sparse_updates:
                # Apply sparsification
                threshold = np.percentile(np.abs(update), (1 - self.config.compression_ratio) * 100)
                sparse_update = np.where(np.abs(update) > threshold, update, 0)
                compressed_updates[key] = sparse_update
            else:
                # Apply quantization
                quantized_update = self._quantize_weights(update, self.config.quantization_bits)
                compressed_updates[key] = quantized_update
        
        return compressed_updates
    
    def _quantize_weights(self, weights: np.ndarray, bits: int) -> np.ndarray:
        """Quantize weights to specified bit precision."""
        # Separate amplitude and phase for complex weights
        amplitude = np.abs(weights)
        phase = np.angle(weights)
        
        # Quantize amplitude
        max_amp = np.max(amplitude)
        if max_amp > 0:
            quantized_amp = np.round(amplitude / max_amp * (2**bits - 1)) / (2**bits - 1) * max_amp
        else:
            quantized_amp = amplitude
        
        # Quantize phase
        quantized_phase = np.round(phase / (2 * np.pi) * (2**bits - 1)) / (2**bits - 1) * (2 * np.pi)
        
        return quantized_amp * np.exp(1j * quantized_phase)
    
    def _compute_update_size(self, updates: Dict[str, np.ndarray]) -> float:
        """Compute size of updates in MB."""
        total_size = 0
        for update in updates.values():
            total_size += update.nbytes
        return total_size / (1024 * 1024)
    
    def transmit_updates(self, server_address: str) -> OpticalCommunicationMetrics:
        """Transmit local updates to server via optical communication."""
        start_time = time.perf_counter_ns()
        
        # Prepare updates for optical transmission
        serialized_updates = self._serialize_updates(self.local_updates)
        
        # Transmit via optical channels
        transmission_metrics = self.optical_transmitter.transmit(
            data=serialized_updates,
            destination=server_address,
            wavelength_channels=self.config.wavelength_channels
        )
        
        transmission_latency = time.perf_counter_ns() - start_time
        transmission_metrics.transmission_latency_ns = transmission_latency
        
        self.communication_metrics.append(transmission_metrics)
        
        return transmission_metrics
    
    def _serialize_updates(self, updates: Dict[str, np.ndarray]) -> bytes:
        """Serialize updates for optical transmission."""
        # Convert complex arrays to format suitable for optical encoding
        serializable_updates = {}
        for key, update in updates.items():
            if np.iscomplexobj(update):
                serializable_updates[key] = {
                    "real": update.real,
                    "imag": update.imag,
                    "shape": update.shape
                }
            else:
                serializable_updates[key] = update
        
        return pickle.dumps(serializable_updates)


class PhotonicFederatedServer:
    """
    Federated learning server for photonic neural networks.
    
    Coordinates training across multiple photonic clients using optical
    communication and specialized aggregation algorithms.
    """
    
    def __init__(self, 
                 global_model: PhotonicNeuralNetwork,
                 federated_config: FederatedConfig):
        """Initialize photonic federated server."""
        self.global_model = global_model
        self.config = federated_config
        
        # Server state
        self.round_number = 0
        self.clients = {}
        self.aggregation_history = []
        
        # Optical communication system
        self.optical_communication_hub = OpticalCommunicationHub(federated_config)
        
        # Aggregation algorithms
        self.aggregator = PhotonicModelAggregator(federated_config)
        
        # Performance monitoring
        self.round_metrics = []
        
        logger.info("Photonic federated server initialized")
    
    def register_client(self, client: PhotonicClient):
        """Register a new photonic client."""
        self.clients[client.client_id] = client
        logger.info(f"Client {client.client_id} registered")
    
    def run_federated_learning(self) -> Dict[str, Any]:
        """
        Run federated learning process across all rounds.
        
        Returns:
            Comprehensive training results and metrics
        """
        logger.info(f"Starting federated learning for {self.config.rounds} rounds")
        
        for round_num in range(self.config.rounds):
            self.round_number = round_num
            
            round_start_time = time.time()
            
            # Client selection
            selected_clients = self._select_clients()
            logger.info(f"Round {round_num}: Selected {len(selected_clients)} clients")
            
            # Distribute global model
            global_weights = self._get_global_weights()
            
            # Collect local updates
            client_updates = self._collect_client_updates(selected_clients, global_weights)
            
            # Aggregate updates
            aggregated_weights = self.aggregator.aggregate_updates(client_updates, global_weights)
            
            # Update global model
            self._update_global_model(aggregated_weights)
            
            # Evaluate global model
            evaluation_metrics = self._evaluate_global_model()
            
            # Record round metrics
            round_time = time.time() - round_start_time
            round_metrics = {
                "round": round_num,
                "selected_clients": len(selected_clients),
                "aggregation_time_s": round_time,
                "global_accuracy": evaluation_metrics.get("accuracy", 0.0),
                "global_loss": evaluation_metrics.get("loss", float('inf')),
                "communication_overhead_mb": sum(
                    update["metrics"]["update_size_mb"] for update in client_updates
                ),
                "average_client_accuracy": np.mean([
                    update["metrics"]["final_accuracy"] for update in client_updates
                ])
            }
            
            self.round_metrics.append(round_metrics)
            
            # Adaptive learning rate
            if self.config.adaptive_learning_rate:
                self._adapt_learning_rate(round_metrics)
            
            # Log progress
            if round_num % 10 == 0:
                logger.info(f"Round {round_num}: "
                          f"Global accuracy={evaluation_metrics.get('accuracy', 0.0):.4f}, "
                          f"Avg client accuracy={round_metrics['average_client_accuracy']:.4f}")
        
        # Compile final results
        final_results = {
            "final_global_accuracy": self.round_metrics[-1]["global_accuracy"],
            "final_global_loss": self.round_metrics[-1]["global_loss"],
            "total_rounds": self.config.rounds,
            "total_communication_mb": sum(m["communication_overhead_mb"] for m in self.round_metrics),
            "average_round_time_s": np.mean([m["aggregation_time_s"] for m in self.round_metrics]),
            "convergence_round": self._find_convergence_round(),
            "round_metrics": self.round_metrics
        }
        
        logger.info("Federated learning completed successfully")
        
        return final_results
    
    def _select_clients(self) -> List[str]:
        """Select clients for current round."""
        num_selected = int(self.config.client_fraction * len(self.clients))
        selected_client_ids = np.random.choice(
            list(self.clients.keys()), 
            size=min(num_selected, len(self.clients)), 
            replace=False
        )
        
        # Update client selection status
        for client_id in self.clients:
            self.clients[client_id].is_selected = client_id in selected_client_ids
            self.clients[client_id].round_number = self.round_number
        
        return list(selected_client_ids)
    
    def _get_global_weights(self) -> Dict[str, np.ndarray]:
        """Extract global model weights."""
        weights = {}
        for layer_idx, layer in enumerate(self.global_model.layers):
            weights[f"layer_{layer_idx}_weights"] = layer.weights.copy()
        return weights
    
    def _collect_client_updates(self, 
                              selected_clients: List[str], 
                              global_weights: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Collect updates from selected clients."""
        client_updates = []
        
        # Parallel client training
        with ThreadPoolExecutor(max_workers=min(len(selected_clients), 8)) as executor:
            futures = {}
            
            for client_id in selected_clients:
                client = self.clients[client_id]
                future = executor.submit(client.local_training, global_weights)
                futures[future] = client_id
            
            # Collect results
            for future in futures:
                try:
                    update_result = future.result(timeout=300)  # 5 minute timeout
                    client_updates.append(update_result)
                except Exception as e:
                    client_id = futures[future]
                    logger.error(f"Client {client_id} training failed: {e}")
        
        return client_updates
    
    def _update_global_model(self, aggregated_weights: Dict[str, np.ndarray]):
        """Update global model with aggregated weights."""
        for layer_idx, layer in enumerate(self.global_model.layers):
            layer_key = f"layer_{layer_idx}_weights"
            if layer_key in aggregated_weights:
                layer.weights = aggregated_weights[layer_key]
    
    def _evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate global model performance."""
        # Use validation data from a subset of clients
        validation_data = []
        for client in list(self.clients.values())[:3]:  # Use first 3 clients for validation
            X_val, y_val = client.local_data
            validation_data.append((X_val[:50], y_val[:50]))  # Small validation set
        
        if not validation_data:
            return {"accuracy": 0.0, "loss": float('inf')}
        
        # Combine validation data
        X_combined = np.concatenate([X for X, y in validation_data])
        y_combined = np.concatenate([y for X, y in validation_data])
        
        # Evaluate
        try:
            predictions, _ = self.global_model.forward(X_combined)
            
            # Compute accuracy
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_combined, axis=1) if y_combined.ndim > 1 else y_combined
            accuracy = np.mean(pred_classes == true_classes)
            
            # Compute loss
            loss = np.mean((predictions - y_combined) ** 2)
            
            return {"accuracy": accuracy, "loss": loss}
            
        except Exception as e:
            logger.error(f"Global model evaluation failed: {e}")
            return {"accuracy": 0.0, "loss": float('inf')}
    
    def _adapt_learning_rate(self, round_metrics: Dict[str, Any]):
        """Adapt learning rate based on convergence progress."""
        if len(self.round_metrics) >= 5:
            # Check improvement in last 5 rounds
            recent_accuracies = [m["global_accuracy"] for m in self.round_metrics[-5:]]
            improvement = recent_accuracies[-1] - recent_accuracies[0]
            
            # Adjust learning rate for all clients
            if improvement < 0.01:  # Slow improvement
                for client in self.clients.values():
                    # Reduce learning rate
                    current_lr = getattr(client.model, 'learning_rate', 0.001)
                    new_lr = max(0.0001, current_lr * 0.9)
                    setattr(client.model, 'learning_rate', new_lr)
            elif improvement > 0.05:  # Fast improvement
                for client in self.clients.values():
                    # Increase learning rate slightly
                    current_lr = getattr(client.model, 'learning_rate', 0.001)
                    new_lr = min(0.01, current_lr * 1.1)
                    setattr(client.model, 'learning_rate', new_lr)
    
    def _find_convergence_round(self) -> int:
        """Find the round where the model converged."""
        if len(self.round_metrics) < 10:
            return len(self.round_metrics)
        
        # Look for plateau in accuracy
        accuracies = [m["global_accuracy"] for m in self.round_metrics]
        
        for i in range(10, len(accuracies)):
            # Check if accuracy hasn't improved significantly in last 10 rounds
            recent_improvement = max(accuracies[i-10:i]) - min(accuracies[i-10:i])
            if recent_improvement < 0.01:
                return i - 10
        
        return len(self.round_metrics)


class PhotonicModelAggregator:
    """
    Specialized model aggregation for photonic neural networks.
    
    Implements photonic-aware aggregation considering optical interference
    effects and hardware constraints.
    """
    
    def __init__(self, federated_config: FederatedConfig):
        """Initialize photonic model aggregator."""
        self.config = federated_config
        
    def aggregate_updates(self, 
                         client_updates: List[Dict[str, Any]], 
                         global_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Aggregate client updates using photonic-aware algorithm.
        
        Args:
            client_updates: List of client update dictionaries
            global_weights: Current global model weights
            
        Returns:
            Aggregated model weights
        """
        if not client_updates:
            return global_weights
        
        # Extract updates and weights
        client_weight_updates = [update["updates"] for update in client_updates]
        client_data_sizes = [update["metrics"]["data_size"] for update in client_updates]
        
        # Choose aggregation method
        if self.config.secure_aggregation:
            aggregated_updates = self._secure_aggregate(client_weight_updates, client_data_sizes)
        else:
            aggregated_updates = self._fedavg_aggregate(client_weight_updates, client_data_sizes)
        
        # Apply photonic-specific constraints
        aggregated_updates = self._apply_photonic_constraints(aggregated_updates)
        
        # Compute new global weights
        new_global_weights = {}
        for key in global_weights:
            if key in aggregated_updates:
                new_global_weights[key] = global_weights[key] + aggregated_updates[key]
            else:
                new_global_weights[key] = global_weights[key]
        
        return new_global_weights
    
    def _fedavg_aggregate(self, 
                         client_updates: List[Dict[str, np.ndarray]], 
                         data_sizes: List[int]) -> Dict[str, np.ndarray]:
        """Standard FedAvg aggregation with data size weighting."""
        if not client_updates:
            return {}
        
        # Compute weights based on data sizes
        total_data = sum(data_sizes)
        weights = [size / total_data for size in data_sizes]
        
        # Aggregate each layer
        aggregated = {}
        for key in client_updates[0].keys():
            # Weighted average of client updates
            weighted_updates = [
                weight * client_updates[i][key] 
                for i, weight in enumerate(weights)
                if key in client_updates[i]
            ]
            
            if weighted_updates:
                aggregated[key] = np.sum(weighted_updates, axis=0)
        
        return aggregated
    
    def _secure_aggregate(self, 
                         client_updates: List[Dict[str, np.ndarray]], 
                         data_sizes: List[int]) -> Dict[str, np.ndarray]:
        """Secure aggregation with added noise for privacy."""
        # Start with standard aggregation
        aggregated = self._fedavg_aggregate(client_updates, data_sizes)
        
        # Add calibrated noise for privacy
        noise_scale = self.config.privacy_epsilon * 0.1
        
        for key, update in aggregated.items():
            # Add Gaussian noise calibrated to privacy budget
            noise = np.random.normal(0, noise_scale, update.shape).astype(update.dtype)
            aggregated[key] = update + noise
        
        return aggregated
    
    def _apply_photonic_constraints(self, updates: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply photonic hardware constraints to aggregated updates."""
        constrained_updates = {}
        
        for key, update in updates.items():
            # Constrain complex weights to realistic ranges
            if np.iscomplexobj(update):
                amplitude = np.abs(update)
                phase = np.angle(update)
                
                # Limit amplitude to prevent saturation
                max_amplitude = 2.0
                constrained_amplitude = np.clip(amplitude, 0, max_amplitude)
                
                # Ensure phase stability (avoid rapid phase changes)
                phase_smoothed = self._smooth_phase(phase)
                
                constrained_updates[key] = constrained_amplitude * np.exp(1j * phase_smoothed)
            else:
                # Real-valued constraints
                constrained_updates[key] = np.clip(update, -5.0, 5.0)
        
        return constrained_updates
    
    def _smooth_phase(self, phase: np.ndarray, smoothing_factor: float = 0.1) -> np.ndarray:
        """Apply phase smoothing to reduce rapid fluctuations."""
        if phase.size <= 1:
            return phase
        
        # Simple low-pass filtering
        smoothed = phase.copy()
        for i in range(1, phase.size):
            smoothed.flat[i] = (1 - smoothing_factor) * smoothed.flat[i-1] + smoothing_factor * phase.flat[i]
        
        return smoothed


class OpticalTransmitter:
    """Optical data transmission system for federated learning."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        
    def transmit(self, 
                data: bytes, 
                destination: str, 
                wavelength_channels: int) -> OpticalCommunicationMetrics:
        """Simulate optical data transmission."""
        
        # Simulate transmission metrics
        data_size_mb = len(data) / (1024 * 1024)
        
        # Transmission latency based on data size and bandwidth
        transmission_time_s = data_size_mb * 8 / self.config.optical_bandwidth_ghz  # Convert to seconds
        transmission_latency_ns = transmission_time_s * 1e9
        
        # Channel utilization
        channel_utilization = min(1.0, data_size_mb / (wavelength_channels * 10))  # Assume 10MB per channel capacity
        
        # Signal quality metrics (simplified simulation)
        snr_db = 30.0 - self.config.transmission_loss_db * np.log10(data_size_mb + 1)
        ber = 1e-9 * (10 ** (self.config.transmission_loss_db / 10))
        
        # Power consumption (proportional to data size and channels)
        power_mw = wavelength_channels * 50 + data_size_mb * 10
        
        # Thermal crosstalk (increases with channel utilization)
        thermal_crosstalk = channel_utilization * 0.1
        
        return OpticalCommunicationMetrics(
            transmission_latency_ns=transmission_latency_ns,
            channel_utilization=channel_utilization,
            signal_to_noise_ratio_db=snr_db,
            bit_error_rate=ber,
            power_consumption_mw=power_mw,
            thermal_crosstalk=thermal_crosstalk
        )


class OpticalReceiver:
    """Optical data reception system for federated learning."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        
    def receive(self, transmission_metrics: OpticalCommunicationMetrics) -> bytes:
        """Simulate optical data reception."""
        # In a real implementation, this would demodulate optical signals
        # For simulation, we return placeholder data
        return b"simulated_received_data"


class OpticalCommunicationHub:
    """Central hub for managing optical communications between clients and server."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.active_connections = {}
        self.communication_log = []
        
    def route_communication(self, 
                          source: str, 
                          destination: str, 
                          data: bytes) -> OpticalCommunicationMetrics:
        """Route optical communication between federated participants."""
        
        # Create optical transmitter for this communication
        transmitter = OpticalTransmitter(self.config)
        
        # Simulate transmission
        metrics = transmitter.transmit(
            data=data,
            destination=destination,
            wavelength_channels=self.config.wavelength_channels
        )
        
        # Log communication
        self.communication_log.append({
            "timestamp": time.time(),
            "source": source,
            "destination": destination,
            "data_size_bytes": len(data),
            "metrics": asdict(metrics)
        })
        
        return metrics


class DifferentialPrivacyMechanism:
    """Differential privacy mechanism for federated photonic learning."""
    
    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        
    def apply_privacy(self, updates: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply differential privacy to model updates."""
        private_updates = {}
        
        for key, update in updates.items():
            # Compute sensitivity (max change in update)
            sensitivity = np.max(np.abs(update))
            
            # Add Laplace noise calibrated to privacy budget
            noise_scale = sensitivity / self.epsilon
            
            if np.iscomplexobj(update):
                # Add noise to both real and imaginary parts
                real_noise = np.random.laplace(0, noise_scale, update.shape)
                imag_noise = np.random.laplace(0, noise_scale, update.shape)
                noise = real_noise + 1j * imag_noise
            else:
                noise = np.random.laplace(0, noise_scale, update.shape)
            
            private_updates[key] = update + noise
        
        return private_updates


def create_federated_photonic_system(
    num_clients: int,
    model_config: Dict[str, Any],
    data_distribution: List[Tuple[np.ndarray, np.ndarray]],
    federated_config: FederatedConfig = None
) -> Tuple[PhotonicFederatedServer, List[PhotonicClient]]:
    """
    Create complete federated photonic learning system.
    
    Args:
        num_clients: Number of photonic clients
        model_config: Configuration for photonic neural network
        data_distribution: List of (X, y) data for each client
        federated_config: Federated learning configuration
        
    Returns:
        Tuple of (server, list of clients)
    """
    if federated_config is None:
        federated_config = FederatedConfig(num_clients=num_clients)
    
    # Create global model
    wavelength_config = WavelengthConfig(num_channels=federated_config.wavelength_channels)
    thermal_config = ThermalConfig()
    fabrication_config = FabricationConfig()
    
    layer_configs = model_config.get("layer_configs", [
        LayerConfig(input_dim=784, output_dim=128, activation="relu"),
        LayerConfig(input_dim=128, output_dim=10, activation="sigmoid")
    ])
    
    global_model = PhotonicNeuralNetwork(
        layer_configs, wavelength_config, thermal_config, fabrication_config
    )
    
    # Create server
    server = PhotonicFederatedServer(global_model, federated_config)
    
    # Create clients
    clients = []
    for i in range(num_clients):
        client_id = f"photonic_client_{i:03d}"
        
        # Create local model (copy of global model)
        local_model = PhotonicNeuralNetwork(
            layer_configs, wavelength_config, thermal_config, fabrication_config
        )
        
        # Assign local data
        if i < len(data_distribution):
            local_data = data_distribution[i]
        else:
            # Generate synthetic data if not enough provided
            X_synthetic = np.random.randn(100, layer_configs[0].input_dim)
            y_synthetic = np.eye(layer_configs[-1].output_dim)[np.random.randint(0, layer_configs[-1].output_dim, 100)]
            local_data = (X_synthetic, y_synthetic)
        
        client = PhotonicClient(client_id, local_model, local_data, federated_config)
        clients.append(client)
        server.register_client(client)
    
    logger.info(f"Federated photonic system created with {num_clients} clients")
    
    return server, clients


def run_federated_experiment(
    server: PhotonicFederatedServer,
    clients: List[PhotonicClient],
    experiment_name: str = "federated_photonic_experiment"
) -> Dict[str, Any]:
    """
    Run complete federated photonic learning experiment.
    
    Args:
        server: Federated server
        clients: List of photonic clients
        experiment_name: Name for the experiment
        
    Returns:
        Comprehensive experiment results
    """
    logger.info(f"Starting federated experiment: {experiment_name}")
    
    experiment_start_time = time.time()
    
    # Run federated learning
    federated_results = server.run_federated_learning()
    
    experiment_duration = time.time() - experiment_start_time
    
    # Collect communication metrics
    total_communication_overhead = sum(
        len(client.communication_metrics) for client in clients
    )
    
    average_optical_latency = np.mean([
        metric.transmission_latency_ns 
        for client in clients 
        for metric in client.communication_metrics
    ]) if total_communication_overhead > 0 else 0.0
    
    # Compile comprehensive results
    experiment_results = {
        "experiment_name": experiment_name,
        "experiment_duration_s": experiment_duration,
        "federated_results": federated_results,
        "communication_metrics": {
            "total_transmissions": total_communication_overhead,
            "average_optical_latency_ns": average_optical_latency,
            "total_data_transmitted_mb": sum(
                sum(client.training_history)
                for client in clients
                if hasattr(client, 'training_history')
            )
        },
        "client_performance": [
            {
                "client_id": client.client_id,
                "final_accuracy": client.training_history[-1]["final_accuracy"] if client.training_history else 0.0,
                "total_training_time_s": sum(h["training_time_s"] for h in client.training_history),
                "communication_events": len(client.communication_metrics)
            }
            for client in clients
        ],
        "system_efficiency": {
            "convergence_efficiency": federated_results["convergence_round"] / federated_results["total_rounds"],
            "communication_efficiency": federated_results["final_global_accuracy"] / federated_results["total_communication_mb"],
            "time_efficiency": federated_results["final_global_accuracy"] / experiment_duration
        }
    }
    
    logger.info(f"Federated experiment completed successfully")
    logger.info(f"Final global accuracy: {federated_results['final_global_accuracy']:.4f}")
    logger.info(f"Total experiment time: {experiment_duration:.2f}s")
    
    return experiment_results