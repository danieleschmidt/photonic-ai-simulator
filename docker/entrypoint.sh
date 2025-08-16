#!/bin/bash
# Production entrypoint script for Photonic AI Simulator
# Handles initialization, health checks, and graceful shutdown

set -euo pipefail

# Configuration
export PHOTONIC_HOME="/app"
export PHOTONIC_DATA_DIR="/app/data"
export PHOTONIC_LOGS_DIR="/app/logs"
export PHOTONIC_MODELS_DIR="/app/models"
export PYTHONPATH="/app:$PYTHONPATH"

# Logging configuration
setup_logging() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Setting up logging..."
    
    # Create log directories
    mkdir -p "$PHOTONIC_LOGS_DIR"
    
    # Configure log rotation
    cat > /tmp/logrotate.conf << EOF
$PHOTONIC_LOGS_DIR/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 644 photonic photonic
}
EOF
}

# System health checks
health_check() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Running system health checks..."
    
    # Check Python environment
    python3 -c "import sys; print(f'Python version: {sys.version}')"
    
    # Check GPU availability
    if [ "${PHOTONIC_ENABLE_GPU:-false}" = "true" ]; then
        python3 -c "
try:
    import cupy as cp
    print(f'CUDA devices: {cp.cuda.runtime.getDeviceCount()}')
except ImportError:
    print('CUDA not available')
"
    fi
    
    # Check core modules
    python3 -c "
from src.core import PhotonicProcessor
from src.models import create_benchmark_network
print('Core modules imported successfully')
"
    
    # Check data directories
    for dir in "$PHOTONIC_DATA_DIR" "$PHOTONIC_LOGS_DIR" "$PHOTONIC_MODELS_DIR"; do
        if [ ! -d "$dir" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Creating directory: $dir"
            mkdir -p "$dir"
        fi
    done
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Health checks completed successfully"
}

# Initialize system
initialize_system() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Initializing Photonic AI Simulator..."
    
    # Set file permissions
    find /app -type f -name "*.py" -exec chmod 644 {} \;
    find /app -type f -name "*.sh" -exec chmod 755 {} \;
    
    # Initialize configuration
    if [ ! -f "$PHOTONIC_DATA_DIR/config.json" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Creating default configuration..."
        python3 -c "
import json
import os

config = {
    'system': {
        'log_level': os.getenv('PHOTONIC_LOG_LEVEL', 'INFO'),
        'enable_gpu': os.getenv('PHOTONIC_ENABLE_GPU', 'false').lower() == 'true',
        'enable_monitoring': os.getenv('PHOTONIC_ENABLE_MONITORING', 'true').lower() == 'true',
        'worker_count': int(os.getenv('PHOTONIC_WORKER_COUNT', '4')),
        'thread_count': int(os.getenv('PHOTONIC_THREAD_COUNT', '8'))
    },
    'security': {
        'require_auth': os.getenv('PHOTONIC_REQUIRE_AUTH', 'true').lower() == 'true',
        'session_timeout': int(os.getenv('PHOTONIC_SESSION_TIMEOUT', '3600')),
        'max_concurrent_requests': int(os.getenv('PHOTONIC_MAX_CONCURRENT_REQUESTS', '100')),
        'rate_limit': os.getenv('PHOTONIC_RATE_LIMIT', '1000/hour')
    },
    'performance': {
        'batch_size': int(os.getenv('PHOTONIC_BATCH_SIZE', '32')),
        'cache_size_mb': int(os.getenv('PHOTONIC_CACHE_SIZE_MB', '512')),
        'optimization_level': 'high'
    }
}

with open('$PHOTONIC_DATA_DIR/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print('Configuration initialized')
"
    fi
    
    # Initialize models directory
    if [ ! -f "$PHOTONIC_MODELS_DIR/.initialized" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Initializing models..."
        python3 -c "
from src.models import create_benchmark_network
import os

models_dir = '$PHOTONIC_MODELS_DIR'
tasks = ['mnist', 'cifar10', 'vowel_classification']

for task in tasks:
    model_path = os.path.join(models_dir, f'{task}_model.npy')
    if not os.path.exists(model_path):
        print(f'Creating {task} model...')
        model = create_benchmark_network(task)
        model.save_model(model_path)

# Mark as initialized
with open(os.path.join(models_dir, '.initialized'), 'w') as f:
    f.write('initialized')

print('Models initialized')
"
    fi
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] System initialization completed"
}

# Start monitoring services
start_monitoring() {
    if [ "${PHOTONIC_ENABLE_MONITORING:-true}" = "true" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Starting monitoring services..."
        
        # Start metrics collection in background
        python3 -c "
from src.utils.monitoring import start_metrics_collection
start_metrics_collection()
" &
        
        echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Monitoring services started"
    fi
}

# Graceful shutdown handler
shutdown_handler() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Received shutdown signal, gracefully shutting down..."
    
    # Kill background processes
    jobs -p | xargs -r kill
    
    # Save any pending data
    python3 -c "
import os
import json
from datetime import datetime

shutdown_info = {
    'shutdown_time': datetime.utcnow().isoformat(),
    'reason': 'graceful_shutdown'
}

with open('$PHOTONIC_LOGS_DIR/shutdown.log', 'a') as f:
    f.write(json.dumps(shutdown_info) + '\n')

print('Shutdown information saved')
"
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Graceful shutdown completed"
    exit 0
}

# Register signal handlers
trap shutdown_handler SIGTERM SIGINT

# Main execution
main() {
    local mode="${1:-production}"
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Starting Photonic AI Simulator in $mode mode..."
    
    # Setup
    setup_logging
    health_check
    initialize_system
    start_monitoring
    
    case "$mode" in
        "production")
            echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Starting production server..."
            exec python3 -c "
from src.deployment import run_production_server
run_production_server()
"
            ;;
        "api")
            echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Starting API server..."
            exec uvicorn src.api:app \
                --host 0.0.0.0 \
                --port 8000 \
                --workers ${PHOTONIC_WORKER_COUNT:-4} \
                --log-level ${PHOTONIC_LOG_LEVEL:-info} \
                --access-log \
                --loop uvloop
            ;;
        "worker")
            echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Starting background worker..."
            exec celery -A src.tasks worker \
                --loglevel=${PHOTONIC_LOG_LEVEL:-info} \
                --concurrency=${PHOTONIC_WORKER_COUNT:-4}
            ;;
        "scheduler")
            echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Starting task scheduler..."
            exec celery -A src.tasks beat \
                --loglevel=${PHOTONIC_LOG_LEVEL:-info}
            ;;
        "benchmark")
            echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Running benchmarks..."
            python3 -c "
from src.cli import benchmark_command
import sys
sys.argv = ['benchmark', '--task', 'all', '--samples', '1000']
benchmark_command()
"
            ;;
        "test")
            echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Running tests..."
            python3 -m pytest tests/ -v --tb=short
            ;;
        "shell")
            echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Starting interactive shell..."
            exec python3 -c "
from src.models import create_benchmark_network
from src.training import create_training_pipeline
from src.optimization import create_optimized_network
import numpy as np

print('Photonic AI Simulator - Interactive Shell')
print('Available functions:')
print('  - create_benchmark_network(task)')
print('  - create_training_pipeline(model, type)')
print('  - create_optimized_network(task, level)')
print('')

# Start interactive session
import IPython
IPython.embed()
"
            ;;
        *)
            echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Unknown mode: $mode"
            echo "Available modes: production, api, worker, scheduler, benchmark, test, shell"
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"