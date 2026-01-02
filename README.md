## Dielectric Breakdown: Geometry Effect (Needle–Plane vs Parallel Plates)

This project provides a research-grade numerical simulation and visualization suite to study how electrode geometry influences electric-field intensification and predicted breakdown inception in gases (default: air at 1 atm).

Key capabilities:
- Parameterized geometries: parallel plates and axisymmetric needle–plane.
- Sparse finite-volume Laplace solver on stretched (non-uniform) tensor grids.
- Optional axisymmetric formulation for needle–plane.
- Field-line tracing with 4th-order RK and adaptive step.
- Raether–Meek (Townsend) ionization integral along field lines for inception prediction.
- High-quality static visualizations (equipotentials, |E|, streamlines) and interactive notebook.
- CLI for parameter sweeps and report-quality figure export.

### Quickstart
1. Create a Python environment (3.9+ recommended) and install requirements:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Run an example (needle–plane):
```
python -m dielectric_breakdown.cli --geometry needle_plane --gap 5e-3 --needle-tip-radius 50e-6 --voltage 20e3 --nx 400 --ny 300 --axisymmetric
```
3. Open the demo notebook:
```
jupyter lab notebooks/GeometryEffect.ipynb
```

Outputs are written to `outputs/` by default.

### References
- Young, Freedman, Ford. University Physics (Electric Potential).
- Ordin, S. Finite difference for electrostatic potential and field simulation. Int. J. Phys. Math. 2025;7(2):109–117.
- Raether–Meek avalanche/streamer inception criterion.




