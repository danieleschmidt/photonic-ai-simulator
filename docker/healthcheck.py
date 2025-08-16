#!/usr/bin/env python3
"""
Health check script for production deployment.
Validates system components and API endpoints.
"""

import sys
import time
import json
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, '/app')

def check_python_environment():
    """Check Python environment and core imports."""
    try:
        import numpy as np
        import sys
        
        # Check Python version
        if sys.version_info < (3, 8):
            return False, f"Python version {sys.version_info} < 3.8"
        
        # Check NumPy
        np.array([1, 2, 3])
        
        return True, "Python environment OK"
    except Exception as e:
        return False, f"Python environment error: {e}"


def check_core_modules():
    """Check core application modules."""
    try:
        # Test core imports
        from src.core import PhotonicProcessor, WavelengthConfig, ThermalConfig, FabricationConfig
        from src.models import create_benchmark_network, PhotonicNeuralNetwork
        from src.training import create_training_pipeline
        
        # Test basic functionality
        config = WavelengthConfig(num_channels=4)
        thermal_config = ThermalConfig()
        fab_config = FabricationConfig()
        
        processor = PhotonicProcessor(config, thermal_config, fab_config)
        
        return True, "Core modules OK"
    except Exception as e:
        return False, f"Core modules error: {e}"


def check_model_creation():
    """Check model creation and basic inference."""
    try:
        from src.models import create_benchmark_network
        import numpy as np
        
        # Create a small test model
        model = create_benchmark_network('vowel_classification')
        
        # Test inference
        test_input = np.random.randn(2, 10)
        predictions, metrics = model.forward(test_input)
        
        # Validate output
        if predictions.shape != (2, 6):
            return False, f"Invalid output shape: {predictions.shape}"
        
        if metrics['total_latency_ns'] <= 0:
            return False, f"Invalid latency: {metrics['total_latency_ns']}"
        
        return True, "Model creation and inference OK"
    except Exception as e:
        return False, f"Model creation error: {e}"


def check_gpu_availability():
    """Check GPU availability if enabled."""
    try:
        import os
        
        if os.getenv('PHOTONIC_ENABLE_GPU', 'false').lower() != 'true':
            return True, "GPU check skipped (disabled)"
        
        try:
            import cupy as cp
            device_count = cp.cuda.runtime.getDeviceCount()
            
            if device_count == 0:
                return False, "No CUDA devices found"
            
            # Test basic GPU operation
            x = cp.array([1, 2, 3])
            y = cp.sum(x)
            result = cp.asnumpy(y)
            
            if result != 6:
                return False, f"GPU computation error: {result} != 6"
            
            return True, f"GPU OK ({device_count} devices)"
        except ImportError:
            return False, "CuPy not available"
        except Exception as e:
            return False, f"GPU error: {e}"
            
    except Exception as e:
        return False, f"GPU check error: {e}"


def check_file_system():
    """Check file system permissions and required directories."""
    try:
        import os
        
        required_dirs = [
            '/app/data',
            '/app/logs', 
            '/app/models'
        ]
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                return False, f"Missing directory: {directory}"
            
            if not os.access(directory, os.R_OK | os.W_OK):
                return False, f"No read/write access to: {directory}"
        
        # Test file creation
        test_file = '/app/data/.healthcheck'
        try:
            with open(test_file, 'w') as f:
                f.write('healthcheck')
            
            with open(test_file, 'r') as f:
                content = f.read()
            
            if content != 'healthcheck':
                return False, "File write/read test failed"
            
            os.remove(test_file)
            
        except Exception as e:
            return False, f"File system test error: {e}"
        
        return True, "File system OK"
    except Exception as e:
        return False, f"File system check error: {e}"


def check_configuration():
    """Check configuration files."""
    try:
        config_file = '/app/data/config.json'
        
        if not Path(config_file).exists():
            return False, f"Configuration file not found: {config_file}"
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        required_sections = ['system', 'security', 'performance']
        for section in required_sections:
            if section not in config:
                return False, f"Missing configuration section: {section}"
        
        return True, "Configuration OK"
    except Exception as e:
        return False, f"Configuration check error: {e}"


def check_api_endpoint():
    """Check API endpoint availability (if running)."""
    try:
        import urllib.request
        import socket
        
        # Check if API port is open
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        
        if result != 0:
            return True, "API endpoint check skipped (not running)"
        
        # Try to connect to health endpoint
        try:
            with urllib.request.urlopen('http://localhost:8000/health', timeout=5) as response:
                if response.status == 200:
                    return True, "API endpoint OK"
                else:
                    return False, f"API returned status {response.status}"
        except Exception as e:
            return False, f"API endpoint error: {e}"
            
    except Exception as e:
        return False, f"API check error: {e}"


def run_health_checks():
    """Run all health checks and return results."""
    checks = [
        ("Python Environment", check_python_environment),
        ("Core Modules", check_core_modules),
        ("Model Creation", check_model_creation),
        ("GPU Availability", check_gpu_availability),
        ("File System", check_file_system),
        ("Configuration", check_configuration),
        ("API Endpoint", check_api_endpoint)
    ]
    
    results = []
    all_passed = True
    
    print(f"Health Check Report - {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 60)
    
    for check_name, check_function in checks:
        try:
            start_time = time.time()
            passed, message = check_function()
            duration = time.time() - start_time
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{check_name:<20} {status:<8} {message} ({duration:.3f}s)")
            
            results.append({
                "name": check_name,
                "passed": passed,
                "message": message,
                "duration": duration
            })
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            print(f"{check_name:<20} ❌ ERROR  {error_msg}")
            
            results.append({
                "name": check_name,
                "passed": False,
                "message": error_msg,
                "duration": 0
            })
            
            all_passed = False
    
    print("=" * 60)
    
    # Calculate summary
    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    success_rate = passed_count / total_count * 100
    
    print(f"Overall Status: {'✅ HEALTHY' if all_passed else '❌ UNHEALTHY'}")
    print(f"Checks Passed: {passed_count}/{total_count} ({success_rate:.1f}%)")
    
    # Save detailed results
    health_report = {
        "timestamp": time.time(),
        "iso_timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "overall_healthy": all_passed,
        "checks_passed": passed_count,
        "total_checks": total_count,
        "success_rate": success_rate,
        "checks": results
    }
    
    try:
        with open('/app/logs/health_check.json', 'w') as f:
            json.dump(health_report, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save health report: {e}")
    
    return all_passed


def main():
    """Main health check entry point."""
    try:
        healthy = run_health_checks()
        sys.exit(0 if healthy else 1)
    except Exception as e:
        print(f"Health check failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()