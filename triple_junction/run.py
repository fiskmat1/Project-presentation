from __future__ import annotations

import argparse
import os
import json
from typing import Tuple, List
import numpy as np

from dielectric_breakdown.geometry import TripleJunctionGeometry, GeometryProblem
from dielectric_breakdown.solver import solve_potential
from dielectric_breakdown.field import compute_field, find_peak_field
from .viz import plot_overview_with_materials, plot_zoom_near_junction, plot_singularity_diagnostics


def _parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Triple-junction (metal–dielectric–air) simulator")
	p.add_argument("--width", type=float, default=0.02, help="Domain width (m)")
	p.add_argument("--height", type=float, default=0.02, help="Domain height (m)")
	p.add_argument("--td", type=float, default=0.5e-3, help="Dielectric thickness (m)")
	p.add_argument("--epsr", type=float, default=4.0, help="Dielectric relative permittivity")
	p.add_argument("--voltage", type=float, default=10e3, help="HV pad potential (V)")
	p.add_argument("--pad-width", type=float, default=3e-3, help="Metal pad width (m)")
	p.add_argument("--pad-height", type=float, default=0.6e-3, help="Metal pad thickness shown in 2D (m)")
	p.add_argument("--pad-center-x", type=float, default=None, help="Pad center x (m); default center of domain")
	p.add_argument("--nx", type=int, default=520, help="Grid nodes in x")
	p.add_argument("--ny", type=int, default=380, help="Grid nodes in y")
	p.add_argument("--method", choices=["cg", "direct"], default="cg", help="Linear solver")
	p.add_argument("--outdir", type=str, default="triple_junction/outputs", help="Output directory")
	p.add_argument("--name", type=str, default=None, help="Optional case name for folder")
	p.add_argument("--no-plot", action="store_true", help="Disable figure generation")
	return p.parse_args()


def _case_folder(args: argparse.Namespace) -> str:
	case = args.name or f"epsr{args.epsr:.2f}_td{args.td*1e3:.2f}mm_V{args.voltage/1e3:.0f}kV"
	return os.path.join(args.outdir, case)


def _junction_window(problem: GeometryProblem, span_x: float, span_y: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
	g = problem.grid
	td = float(problem.meta.get("dielectric_thickness", (g.y[-1] - g.y[0]) * 0.25))
	x0 = float(problem.meta.get("junction_x_left", 0.5 * (g.x[0] + g.x[-1])))
	xmin, xmax = x0 - span_x, x0 + span_x
	ymin, ymax = td, min(g.y[-1], td + span_y)
	i0 = int(np.clip(np.searchsorted(g.x, xmin, side="left"), 0, g.nx - 2))
	i1 = int(np.clip(np.searchsorted(g.x, xmax, side="right"), 1, g.nx))
	j0 = int(np.clip(np.searchsorted(g.y, ymin, side="left"), 0, g.ny - 2))
	j1 = int(np.clip(np.searchsorted(g.y, ymax, side="right"), 1, g.ny))
	return (i0, i1), (j0, j1)


def _compute_metrics(problem: GeometryProblem, V: np.ndarray) -> dict:
	g = problem.grid
	Ex, Ey, Emag = compute_field(g, V)
	Emax, (j_pk, i_pk) = find_peak_field(Emag, mask_exclude=problem.electrode_mask)
	x_pk, y_pk = g.x[i_pk], g.y[j_pk]
	# Peak in air near junction within a small window
	(i0, i1), (j0, j1) = _junction_window(problem, span_x=0.003, span_y=0.006)
	eps = getattr(problem, "epsilon", None)
	air_mask = None
	if eps is not None:
		air_mask = (eps < 1.5)
	window = Emag[j0:j1, i0:i1].copy()
	if air_mask is not None:
		window[~air_mask[j0:j1, i0:i1]] = -np.inf
	jj, ii = np.unravel_index(np.nanargmax(window), window.shape)
	E_air_peak = float(window[jj, ii])
	x_air_peak, y_air_peak = g.x[i0 + ii], g.y[j0 + jj]
	# Continuity of normal displacement across interface just left of junction
	td = float(problem.meta.get("dielectric_thickness", g.y[g.ny // 2]))
	x0 = float(problem.meta.get("junction_x_left", g.x[g.nx // 2]))
	i_mid = int(np.clip(np.searchsorted(g.x, x0, side="left"), 1, g.nx - 2))
	j_air = int(np.clip(np.searchsorted(g.y, td + 1e-6, side="right") - 1, 1, g.ny - 2))
	j_die = int(np.clip(np.searchsorted(g.y, td - 1e-6, side="left"), 1, g.ny - 2))
	Ey_air = float(Ey[j_air, i_mid])
	Ey_die = float(Ey[j_die, i_mid])
	epsr = float(problem.meta.get("eps_r", 1.0))
	Dn_air = 1.0 * Ey_air
	Dn_die = epsr * Ey_die
	return {
		"Emax_global": float(Emax),
		"Emax_global_x": float(x_pk),
		"Emax_global_y": float(y_pk),
		"Emax_air_near_junction": float(E_air_peak),
		"Emax_air_near_junction_x": float(x_air_peak),
		"Emax_air_near_junction_y": float(y_air_peak),
		"Dn_air_at_interface": float(Dn_air),
		"Dn_dielectric_at_interface": float(Dn_die),
		"Dn_ratio_die_over_air": float(Dn_die / (Dn_air + 1e-25)),
	}


def main() -> None:
	args = _parse_args()
	os.makedirs(args.outdir, exist_ok=True)
	out_case = _case_folder(args)
	os.makedirs(out_case, exist_ok=True)

	print("Building triple-junction geometry...")
	geom = TripleJunctionGeometry(
		width=args.width,
		height=args.height,
		dielectric_thickness=args.td,
		eps_r=args.epsr,
		voltage=args.voltage,
		pad_width=args.pad_width,
		pad_height=args.pad_height,
		pad_center_x=args.pad_center_x,
		nx=args.nx,
		ny=args.ny,
	)
	problem = geom.build()
	print("Solving potential...")
	V = solve_potential(problem, method=args.method)
	Ex, Ey, Emag = compute_field(problem.grid, V)
	metrics = _compute_metrics(problem, V)
	print(f"Peak |E| globally = {metrics['Emax_global']:.3e} V/m")
	print(f"Peak |E| in air near junction = {metrics['Emax_air_near_junction']:.3e} V/m")

	# Export numeric
	np.savetxt(os.path.join(out_case, "potential.csv"), V, delimiter=",")
	np.savetxt(os.path.join(out_case, "Emag.csv"), Emag, delimiter=",")
	with open(os.path.join(out_case, "metrics.json"), "w") as f:
		json.dump(metrics, f, indent=2)

	# Figures
	if not args.no_plot:
		title = f"Triple junction | V={args.voltage:.0f} V, td={args.td*1e3:.2f} mm, epsr={args.epsr:.1f}"
		fig = plot_overview_with_materials(problem, V, title=title, output_path=os.path.join(out_case, "overview.png"))
		figz = plot_zoom_near_junction(problem, V, which="left", output_path=os.path.join(out_case, "zoom_left.png"), title="Zoom at left junction")
		figdiag = plot_singularity_diagnostics(problem, V, output_path=os.path.join(out_case, "diagnostics.png"), title="Field scaling near junction")
	print(f"Outputs in: {out_case}")


if __name__ == "__main__":
	main()



