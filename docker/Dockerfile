# Multi-stage production Docker image for Photonic AI Simulator
# Optimized for performance with GPU support and security hardening

# Build stage - Ubuntu 22.04 with CUDA support
FROM nvidia/cuda:12.3-devel-ubuntu22.04 AS builder

# Set working directory
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    libfftw3-dev \
    libhdf5-dev \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /build/venv
ENV PATH="/build/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    gunicorn \
    uvicorn[standard] \
    prometheus-client \
    statsd \
    structlog \
    sentry-sdk \
    redis \
    celery

# Copy source code
COPY src/ ./src/
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install -e .

# Run tests to validate build
COPY tests/ ./tests/
RUN python -m pytest tests/test_core.py -v

# Production stage - Minimal runtime image
FROM nvidia/cuda:12.3-runtime-ubuntu22.04 AS production

# Create non-root user for security
RUN groupadd -r photonic && useradd -r -g photonic -d /app -s /bin/bash -c "Photonic AI User" photonic

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    libblas3 \
    liblapack3 \
    libfftw3-3 \
    libhdf5-103 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set up application directory
WORKDIR /app
RUN chown photonic:photonic /app

# Copy virtual environment from builder
COPY --from=builder --chown=photonic:photonic /build/venv /app/venv

# Copy application code
COPY --from=builder --chown=photonic:photonic /build/src /app/src
COPY --from=builder --chown=photonic:photonic /build/setup.py /app/

# Copy configuration and deployment files
COPY --chown=photonic:photonic docker/entrypoint.sh /app/
COPY --chown=photonic:photonic docker/healthcheck.py /app/
COPY --chown=photonic:photonic docker/config/ /app/config/

# Make scripts executable
RUN chmod +x /app/entrypoint.sh

# Switch to non-root user
USER photonic

# Set environment variables
ENV PATH="/app/venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV PHOTONIC_CONFIG_PATH="/app/config"
ENV PHOTONIC_LOG_LEVEL="INFO"
ENV PHOTONIC_ENABLE_GPU="true"
ENV PHOTONIC_ENABLE_MONITORING="true"

# Security hardening
ENV PHOTONIC_REQUIRE_AUTH="true"
ENV PHOTONIC_SESSION_TIMEOUT="3600"
ENV PHOTONIC_MAX_CONCURRENT_REQUESTS="100"
ENV PHOTONIC_RATE_LIMIT="1000/hour"

# Performance tuning
ENV PHOTONIC_WORKER_COUNT="4"
ENV PHOTONIC_THREAD_COUNT="8"
ENV PHOTONIC_BATCH_SIZE="32"
ENV PHOTONIC_CACHE_SIZE_MB="512"

# Expose ports
EXPOSE 8000 8001 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /app/healthcheck.py

# Volume for persistent data
VOLUME ["/app/data", "/app/logs", "/app/models"]

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["production"]