# lapsim

A modular Python implementation of the BFR lapsim architecture.

## Structure

- `src/lapsim/suspension.py` — suspension parameters
- `src/lapsim/aero.py` — aero parameters
- `src/lapsim/engine.py` — engine parameters
- `src/lapsim/car.py` — car model and force helpers
- `src/lapsim/ggv.py` — GGV envelope representation
- `src/lapsim/track.py` — track geometry loader
- `src/lapsim/simulate.py` — orchestration layer
- `src/lapsim/transient_sim.py` — optional time-domain solver

## Quick start

```bash
cd python_lapsim
python -m pip install -e .
```
