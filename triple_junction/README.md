### Triple-junction (metal–dielectric–air) study

This folder contains a high-fidelity 2D electrostatic simulator and analysis workflow for fields near a metal–dielectric–air triple junction. It relies on the core package in `dielectric_breakdown/` and extends it with:

- Heterogeneous permittivity support (ε map) in the solver
- A dedicated `TripleJunctionGeometry`
- Advanced visualizations and diagnostics tailored to the junction
- Parameter sweep utilities

#### Single run

```bash
python -m triple_junction.run --width 0.02 --height 0.02 --td 5e-4 --epsr 8 --voltage 15000 --pad-width 3e-3 --pad-height 6e-4 --nx 520 --ny 380
```

Outputs are written to `triple_junction/outputs/<case>/` and include:
- `overview.png`: V, |E|, and streamlines with material/electrode overlays
- `zoom_left.png`: zoomed panels around the triple junction
- `diagnostics.png`: log–log scaling of |E| vs distance from the junction
- `potential.csv`, `Emag.csv`, and `metrics.json`

#### Parameter sweep

```bash
python -m triple_junction.sweep
```

Writes a CSV summary to `triple_junction/outputs/sweeps/sweep_results.csv` with peak field in air near the junction and continuity metrics.



