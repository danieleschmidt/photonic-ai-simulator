"""
Tests for photonic neural network training algorithms.
"""

import pytest
import numpy as np
from src.training import (
    TrainingConfig, ForwardOnlyTrainer, HardwareAwareOptimizer,
    create_training_pipeline
)
from src.models import create_benchmark_network


class TestTrainingConfig:
    """Test training configuration."""
    
    def test_default_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.forward_only == True
        assert config.thermal_compensation == True
    
    def test_custom_config(self):
        """Test custom training configuration."""
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=64,
            epochs=50,
            forward_only=False
        )
        assert config.learning_rate == 0.01
        assert config.batch_size == 64
        assert config.epochs == 50
        assert config.forward_only == False


class TestForwardOnlyTrainer:
    """Test forward-only training implementation."""
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        return create_benchmark_network("vowel_classification")
    
    @pytest.fixture
    def config(self):
        """Create test training configuration."""
        return TrainingConfig(
            learning_rate=0.01,
            batch_size=4,
            epochs=2,
            forward_only=True,
            num_perturbations=1  # Reduced for faster testing
        )
    
    @pytest.fixture
    def trainer(self, model, config):
        """Create test trainer."""
        return ForwardOnlyTrainer(model, config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X = np.random.randn(20, 10)  # 20 samples, 10 features
        y = np.eye(6)[np.random.randint(0, 6, 20)]  # One-hot encoded labels
        return X, y
    
    def test_trainer_initialization(self, trainer, config):
        """Test trainer initialization."""
        assert trainer.config == config
        assert trainer.config.forward_only == True
        assert len(trainer.training_metrics) > 0
    
    def test_train_step(self, trainer, sample_data):
        """Test single training step."""
        X, y = sample_data
        batch_x = X[:4]
        batch_y = y[:4]
        
        metrics = trainer.train_step(batch_x, batch_y)
        
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "latency_ns" in metrics
        assert "power_mw" in metrics
        assert metrics["loss"] >= 0
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["latency_ns"] > 0
    
    def test_validate_step(self, trainer, sample_data):
        """Test validation step."""
        X, y = sample_data
        batch_x = X[:4]
        batch_y = y[:4]
        
        metrics = trainer.validate_step(batch_x, batch_y)
        
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"] >= 0
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_loss_computation(self, trainer):
        """Test loss computation."""
        outputs = np.array([[0.8, 0.2], [0.3, 0.7]])
        targets = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        loss = trainer._compute_loss(outputs, targets)
        assert loss >= 0
        assert np.isfinite(loss)
    
    def test_accuracy_computation(self, trainer):
        """Test accuracy computation."""
        # Perfect predictions
        outputs = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1]])
        targets = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        
        accuracy = trainer._compute_accuracy(outputs, targets)
        assert accuracy == 1.0
        
        # Wrong predictions
        outputs = np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
        targets = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        
        accuracy = trainer._compute_accuracy(outputs, targets)
        assert accuracy == 0.0
    
    def test_full_training(self, trainer, sample_data):
        """Test complete training loop."""
        X, y = sample_data
        
        # Run training for just 1 epoch for speed
        trainer.config.epochs = 1
        history = trainer.train(X, y)
        
        assert "epoch" in history
        assert "train_loss" in history
        assert "train_accuracy" in history
        assert "val_loss" in history
        assert "val_accuracy" in history
        assert len(history["epoch"]) == 1
        assert len(history["train_loss"]) == 1


class TestHardwareAwareOptimizer:
    """Test hardware-aware optimization."""
    
    @pytest.fixture
    def optimizer(self):
        """Create test optimizer."""
        return HardwareAwareOptimizer(
            learning_rate=0.01,
            power_budget_mw=100.0,
            noise_tolerance=0.1
        )
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        return create_benchmark_network("vowel_classification")
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.learning_rate == 0.01
        assert optimizer.power_budget == 100.0
        assert optimizer.noise_tolerance == 0.1
        assert optimizer.momentum == 0.9
    
    def test_noise_robustness(self, optimizer):
        """Test noise robustness constraints."""
        weights = np.array([1.0 + 1j, 10.0 + 5j, 0.1 + 0.1j])
        robust_weights = optimizer._apply_noise_robustness(weights)
        
        # Should clip extreme values
        assert np.all(np.abs(robust_weights) <= 1.0 / optimizer.noise_tolerance)
        assert np.all(np.isfinite(robust_weights))
    
    def test_power_constraints(self, optimizer):
        """Test power constraint enforcement."""
        high_power_weights = np.array([5.0 + 5j, 3.0 + 4j, 2.0 + 2j])
        
        constrained_weights = optimizer._enforce_power_constraints(
            high_power_weights, layer_idx=0, current_total_power=50.0
        )
        
        # Power should be reduced
        original_power = np.sum(np.abs(high_power_weights) ** 2) * 15.0
        new_power = np.sum(np.abs(constrained_weights) ** 2) * 15.0
        
        if original_power + 50.0 > optimizer.power_budget:
            assert new_power <= original_power
    
    def test_finite_difference_gradients(self, optimizer, model):
        """Test finite difference gradient computation."""
        X = np.random.randn(2, 10)
        y = np.eye(6)[np.random.randint(0, 6, 2)]
        
        def loss_function(inputs, targets):
            outputs, _ = model.forward(inputs, measure_latency=False)
            return np.mean((outputs - targets) ** 2)
        
        layer = model.layers[0]
        gradients = optimizer._compute_finite_difference_gradients(
            layer, loss_function, X, y, epsilon=1e-3
        )
        
        assert gradients.shape == layer.weights.shape
        assert np.all(np.isfinite(gradients))


class TestTrainingPipeline:
    """Test training pipeline creation."""
    
    def test_forward_only_pipeline(self):
        """Test forward-only training pipeline creation."""
        model = create_benchmark_network("vowel_classification")
        
        trainer = create_training_pipeline(
            model, 
            training_type="forward_only",
            learning_rate=0.01,
            batch_size=16
        )
        
        assert isinstance(trainer, ForwardOnlyTrainer)
        assert trainer.config.forward_only == True
        assert trainer.config.learning_rate == 0.01
        assert trainer.config.batch_size == 16
    
    def test_invalid_training_type(self):
        """Test invalid training type handling."""
        model = create_benchmark_network("vowel_classification")
        
        with pytest.raises(ValueError, match="Unsupported training type"):
            create_training_pipeline(model, training_type="invalid")
    
    def test_pipeline_with_custom_config(self):
        """Test pipeline with custom configuration."""
        model = create_benchmark_network("mnist")
        
        trainer = create_training_pipeline(
            model,
            training_type="forward_only", 
            learning_rate=0.005,
            epochs=50,
            batch_size=64,
            power_constraint_mw=300.0
        )
        
        assert trainer.config.learning_rate == 0.005
        assert trainer.config.epochs == 50
        assert trainer.config.batch_size == 64
        assert trainer.config.power_constraint_mw == 300.0