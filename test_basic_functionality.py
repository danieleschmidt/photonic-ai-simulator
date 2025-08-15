#!/usr/bin/env python3
"""
Basic functionality test to verify core imports and operation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic module imports."""
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
        
        # Test core photonic AI simulator imports
        from core import PhotonicProcessor, WavelengthConfig
        print("✓ Core photonic components imported")
        
        from models import PhotonicNeuralNetwork, MZILayer
        print("✓ Model components imported")
        
        from training import ForwardOnlyTrainer, HardwareAwareOptimizer
        print("✓ Training components imported")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic photonic simulation functionality."""
    try:
        import numpy as np
        from core import PhotonicProcessor, WavelengthConfig, ThermalConfig, FabricationConfig
        
        # Create basic configuration
        wl_config = WavelengthConfig(
            num_channels=4,
            center_wavelength=1550.0,
            wavelength_spacing=0.8
        )
        print("✓ Wavelength configuration created")
        
        # Create thermal and fabrication configs
        thermal_config = ThermalConfig()
        fab_config = FabricationConfig()
        print("✓ Thermal and fabrication configurations created")
        
        # Create processor
        processor = PhotonicProcessor(wl_config, thermal_config, fab_config, enable_noise=False)
        print("✓ Photonic processor created")
        
        # Test basic MZI operation
        test_input = np.array([1.0, 0.5])  # Simple optical signal
        phase_shift = 0.5  # radians
        output1, output2 = processor.mach_zehnder_operation(test_input, phase_shift)
        print(f"✓ Basic MZI operation completed, outputs: {output1}, {output2}")
        
        # Test wavelength multiplexed operation
        batch_size, input_dim, num_wavelengths = 2, 3, 4
        test_inputs = np.random.randn(batch_size, input_dim, num_wavelengths) + 1j * np.random.randn(batch_size, input_dim, num_wavelengths)
        weights = np.random.randn(input_dim, 2, num_wavelengths) + 1j * np.random.randn(input_dim, 2, num_wavelengths)
        result = processor.wavelength_multiplexed_operation(test_inputs, weights)
        print(f"✓ Wavelength multiplexed operation completed, output shape: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running Basic Functionality Tests\n")
    
    # Test imports
    print("1. Testing Imports...")
    import_success = test_basic_imports()
    
    if import_success:
        print("\n2. Testing Basic Functionality...")
        func_success = test_basic_functionality()
        
        if func_success:
            print("\n✅ All basic tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Functionality tests failed")
            sys.exit(1)
    else:
        print("\n❌ Import tests failed")
        sys.exit(1)