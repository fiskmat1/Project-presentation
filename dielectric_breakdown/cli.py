from __future__ import annotations

import argparse
import os
from typing import List, Tuple
import numpy as np

from .geometry import ParallelPlatesGeometry, NeedlePlaneGeometry
try:
	# Optional: available only when triple_junction folder is present
	from .geometry import TripleJunctionGeometry  # type: ignore
	_HAS_TJ = True
except Exception:
	_HAS_TJ = False
from .solver import solve_potential
from .field import compute_field, find_peak_field
from .breakdown import raether_meek_integral


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Dielectric Breakdown Geometry Simulator")
	geoms = ["plates", "needle_plane"] + (["triple_junction"] if _HAS_TJ else [])
	parser.add_argument("--geometry", choices=geoms, default="needle_plane")
	parser.add_argument("--width", type=float, default=0.02, help="Domain width (m)")
	parser.add_argument("--height", type=float, default=0.02, help="Domain height (m) for needle-plane")
	parser.add_argument("--gap", type=float, default=5e-3, help="Gap (m) for plates or tip-plane distance")
	parser.add_argument("--voltage", type=float, default=20e3, help="Applied voltage (V)")
	parser.add_argument("--needle-tip-radius", type=float, default=50e-6, help="Needle tip radius (m)")
	parser.add_argument("--nx", type=int, default=360, help="Grid nodes in x")
	parser.add_argument("--ny", type=int, default=280, help="Grid nodes in y")
	parser.add_argument("--axisymmetric", action="store_true", help="Treat x as radius for needle-plane")
	# Triple junction specific
	parser.add_argument("--td", type=float, default=0.5e-3, help="[TJ] Dielectric thickness (m)")
	parser.add_argument("--epsr", type=float, default=4.0, help="[TJ] Dielectric relative permittivity")
	parser.add_argument("--pad-width", type=float, default=3e-3, help="[TJ] Metal pad width (m)")
	parser.add_argument("--pad-height", type=float, default=0.6e-3, help="[TJ] Metal pad height (m)")
	parser.add_argument("--pad-center-x", type=float, default=None, help="[TJ] Pad center x (m)")
	parser.add_argument("--outdir", type=str, default="outputs")
	parser.add_argument("--method", choices=["cg", "direct"], default="cg")
	parser.add_argument("--export", action="store_true", help="Save figures and CSV metrics")
	parser.add_argument("--no-plot", action="store_true", help="Disable plotting (avoid matplotlib dependency)")
	parser.add_argument("--zoom-tip", action="store_true", help="Also export zoomed panels around needle tip")
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	os.makedirs(args.outdir, exist_ok=True)

	if args.geometry == "plates":
		geom = ParallelPlatesGeometry(width=args.width, gap=args.gap, voltage=args.voltage, nx=args.nx, ny=args.ny)
		problem = geom.build()
	elif args.geometry == "needle_plane":
		geom = NeedlePlaneGeometry(
			width=args.width,
			height=args.height,
			gap=args.gap,
			needle_tip_radius=args.needle_tip_radius,
			voltage=args.voltage,
			nx=args.nx,
			ny=args.ny,
			axisymmetric=args.axisymmetric,
		)
		problem = geom.build()
	else:
		if not _HAS_TJ:
			raise RuntimeError("Triple junction geometry not available in this build.")
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

	print("Assembling and solving...")
	V = solve_potential(problem, method=args.method)
	Ex, Ey, Emag = compute_field(problem.grid, V)
	Emax, (j_pk, i_pk) = find_peak_field(Emag, mask_exclude=problem.electrode_mask)
	x_pk, y_pk = problem.grid.x[i_pk], problem.grid.y[j_pk]
	print(f"Peak |E| = {Emax:.3e} V/m at (x={x_pk:.4e}, y={y_pk:.4e})")

	# Seeds near high-voltage electrode for breakdown integral
	seeds: List[Tuple[float, float]] = []
	if args.geometry == "plates":
		y_seed = problem.grid.y[-1] - 0.25 * (problem.grid.y[-1] - problem.grid.y[-2])
		for xi in np.linspace(problem.grid.x[0], problem.grid.x[-1], 21):
			seeds.append((xi, y_seed))
	else:
		# near the tip along a small arc
		r_tip = args.needle_tip_radius
		z_tip = args.gap
		for r in np.linspace(0.0, 3.0 * r_tip, 13):
			seeds.append((r, z_tip + 2.5 * r_tip))

	res = raether_meek_integral(
		problem.grid,
		V,
		seeds=seeds,
		p=101325.0,
		k_threshold=20.0,
		max_length=max(args.gap, 0.01),
		step_init=min(args.gap, 1e-3) / 20.0,
		stop_region=problem.electrode_mask,
	)
	print(f"Max Raether–Meek integral I = {res['best_integral']:.2f} (threshold ~ {res['k_threshold']:.1f})")
	if res["crossed_threshold"]:
		print("Predicted inception: THRESHOLD CROSSED")
	else:
		print("Predicted inception: NO inception at these parameters")

	# Optional plotting (lazy import to avoid hard dependency)
	if not args.no_plot:
		if args.geometry == "triple_junction":
			# Use advanced TJ visualization if available
			try:
				from triple_junction.viz import plot_overview_with_materials as _plot_overview, plot_zoom_near_junction as _plot_zoom
				title = f"triple_junction | V={args.voltage:.0f} V, td={args.td:.3e} m, epsr={args.epsr:.1f}"
				fig = _plot_overview(
					problem,
					V,
					title=title,
					output_path=os.path.join(args.outdir, "panels.png") if args.export else None,
				)
				if args.export:
					fig.savefig(os.path.join(args.outdir, "panels.png"), dpi=300)
				figz = _plot_zoom(
					problem,
					V,
					which="left",
					output_path=os.path.join(args.outdir, "panels_zoom.png") if args.export else None,
					title="Zoom near junction",
				)
				if args.export:
					figz.savefig(os.path.join(args.outdir, "panels_zoom.png"), dpi=300)
			except Exception:
				from .visualize import plot_static_panels
				title = f"{args.geometry} | V={args.voltage:.0f} V"
				fig = plot_static_panels(
					problem,
					V,
					output_path=os.path.join(args.outdir, "panels.png") if args.export else None,
					title=title,
				)
				if args.export:
					fig.savefig(os.path.join(args.outdir, "panels.png"), dpi=300)
		else:
			from .visualize import plot_static_panels
			title = f"{args.geometry} | V={args.voltage:.0f} V, gap={args.gap:.3e} m"
			fig = plot_static_panels(
				problem,
				V,
				output_path=os.path.join(args.outdir, "panels.png") if args.export else None,
				title=title,
			)
			if args.export:
				fig.savefig(os.path.join(args.outdir, "panels.png"), dpi=300)
			# For needle-plane, optionally add zoomed tip panels
			if args.geometry == "needle_plane" and (args.zoom_tip or args.export):
				from .visualize import plot_panels_zoom_tip
				figz = plot_panels_zoom_tip(
					problem,
					V,
					radius_factor=25.0,
					height_factor=30.0,
					output_path=os.path.join(args.outdir, "panels_zoom.png") if args.export else None,
					title="Zoom near needle tip",
				)
				if args.export:
					figz.savefig(os.path.join(args.outdir, "panels_zoom.png"), dpi=300)
	# Always export numeric data if requested
	if args.export:
		np.savetxt(os.path.join(args.outdir, "potential.csv"), V, delimiter=",")
		np.savetxt(os.path.join(args.outdir, "Emag.csv"), np.array(Emag), delimiter=",")
		with open(os.path.join(args.outdir, "metrics.txt"), "w") as f:
			f.write(f"PeakE, {Emax:.6e}\n")
			f.write(f"PeakX, {x_pk:.6e}\n")
			f.write(f"PeakY, {y_pk:.6e}\n")
			f.write(f"RaetherMeekMax, {res['best_integral']:.6f}\n")
	print("Done.")


if __name__ == "__main__":
	main()


