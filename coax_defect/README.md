### Coaxial/HV cable with dielectric defect (2D cross-section)

This module simulates a coaxial or HV cable cross-section with an insulating annulus between an inner (HV) and outer (ground) conductor. It supports localized defects such as air bubbles or high-permittivity inclusions (e.g., water droplets). The solver uses the core `dielectric_breakdown/` package with a spatial permittivity map and Dirichlet electrodes painted on circular interfaces.

Key features:
- Annular geometry with user-specified `Rin`, `Rout`, `V0`, and base `εr`
- Circular defect region with configurable radius, position, and `εr`
- High-quality visualizations: overview panels, zoom-in near the defect, and radial diagnostics comparing |E|(r) to the ideal coax formula
- CSV exports (`potential.csv`, `Emag.csv`) and JSON metrics

#### Single run

```bash
python -m coax_defect.run --Rin 2e-3 --Rout 10e-3 --V0 15000 --epsr 2.3 \
  --defect-type bubble --defect-radius 0.5e-3 --nx 720 --ny 720
```

Outputs are written to `coax_defect/outputs/<case>/` and include:
- `overview.png`: V, |E|, streamlines, and overlays of electrodes/materials
- `zoom_defect.png`: zoomed panels around the defect
- `radial_profiles.png`: |E|(r) along a ray vs ideal coax baseline
- `potential.csv`, `Emag.csv`, and `metrics.json`

#### Notes
- The computation uses a non-uniform stretched grid with focus near the circular electrodes and cable center for accuracy.
- The outer rectangle boundaries are set to zero-flux (Neumann) so fields are governed by the imposed circular electrodes and ε-map.
- The defect region is clamped to the dielectric annulus so it cannot overwrite the metal electrodes.



