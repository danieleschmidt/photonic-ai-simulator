# photonic-ai-simulator

Simulation of photonic AI accelerator components: waveguides, Mach-Zehnder switches, and optical matrix multiply units.

## Components

- **WaveguideSimulator** — Transfer matrix propagation with loss and phase
- **MachZehnderSwitch** — Electro-optic switch with bar/cross port outputs
- **PhotonicAccelerator** — Optical matrix-vector multiply unit
- **AccelerationBenchmark** — Throughput comparison vs electronic baseline, 64×64 demo

## Usage

```python
from photonic_ai.accelerator import AccelerationBenchmark
bench = AccelerationBenchmark(size=64)
print(bench.demo_64x64())
```

## Install & Test

```bash
pip install -r requirements.txt
pytest tests/ -v
```
