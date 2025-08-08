# Multi-stage build for optimized production image
FROM python:3.9-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code and install package
COPY . /app
WORKDIR /app
RUN pip install -e .

# Production stage
FROM python:3.9-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --from=builder /app /app
WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash photonic
RUN chown -R photonic:photonic /app
USER photonic

# Expose port for Jupyter notebook (optional)
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import photonic_ai_simulator; print('Health check passed')"

# Default command
CMD ["python", "-c", "import photonic_ai_simulator; print('Photonic AI Simulator ready!')"]

# GPU-enabled variant
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu-production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies with GPU support
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install cupy-cuda11x>=10.0.0

# Copy application code
COPY . /app
WORKDIR /app
RUN pip install -e .[gpu]

# Create non-root user
RUN useradd --create-home --shell /bin/bash photonic
RUN chown -R photonic:photonic /app
USER photonic

# Health check with GPU validation
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD python -c "import photonic_ai_simulator; from src.optimization import GPU_AVAILABLE; print(f'GPU available: {GPU_AVAILABLE}')"

# Default command
CMD ["python", "-c", "import photonic_ai_simulator; from src.optimization import GPU_AVAILABLE; print(f'Photonic AI Simulator ready! GPU: {GPU_AVAILABLE}')"]