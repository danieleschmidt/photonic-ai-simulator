# photonic-ai-simulator

A Python simulation framework for photonic computing accelerators — exploring 100x ML acceleration via optical matrix multiplication.

## Architecture

- **WaveguideSimulator** — propagation matrix method for optical waveguides
- **MachZehnderSwitch** — 2×2 optical switch with phase-controlled splitting
- **PhotonicAccelerator** — chains MZ switches to form an optical matrix-multiply unit
- **AccelerationBenchmark** — compare photonic MAC ops/J vs electronic baseline

## Requirements

- Python 3.9+
- numpy

## Installation

```bash
pip install numpy
```

## Quick Demo

```bash
python demo.py
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```
