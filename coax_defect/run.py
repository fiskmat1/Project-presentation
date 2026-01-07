import os
import json
import numpy as np
import argparse

"""
This module can be run in two ways:

1) As a package module (recommended):
   python3 -m coax_defect.run ...

2) As a plain script (works even without __init__.py):
   python3 coax_defect/run.py ...
"""

try:
	# Package-style imports
	from .geometry import (
		build_problem,
		solve_potential,
		compute_field,
		find_peak_field,
	)
except ImportError:  # pragma: no cover
	# Script-style fallback (no package context)
	import sys

	sys.path.insert(0, os.path.dirname(__file__))
	from geometry import (  # type: ignore
		build_problem,
		solve_potential,
		compute_field,
		find_peak_field,
	)


def case_folder(Rin, Rout, V0, epsr, defect_type, defect_radius, defect_epsr, outdir, name):
	def km(v):
		# Store radii in millimetres in folder names
		return round(v * 1e3, 3)

	Rin_mm = "%.3fmm" % km(Rin)
	Rout_mm = "%.3fmm" % km(Rout)
	VkV = "%dkV" % int(round(V0 / 1000.0))

	base = "coax_epsr%.2f_Rin%s_Rout%s_V%s" % (epsr, Rin_mm, Rout_mm, VkV)

	if defect_type != "none":
		rd = "%.3fmm" % km(defect_radius)
		if defect_type == "bubble":
			sfx = "_bubble_r%s" % rd
		else:
			if defect_epsr is None:
				de = 80.0
			else:
				de = float(defect_epsr)
			sfx = "_incl_eps%.1f_r%s" % (de, rd)
		base = base + sfx

	if name:
		base = base + "_" + str(name)

	return os.path.join(outdir, base)


def compute_metrics(problem, V, Em):
	meta = problem.get("meta", {})
	Emax, pos = find_peak_field(Em, None)
	j_max, i_max = pos
	x = problem["x"]
	y = problem["y"]

	metrics = {}
	metrics["Emax_global"] = float(Emax)
	metrics["x_at_Emax"] = float(x[i_max])
	metrics["y_at_Emax"] = float(y[j_max])

	Rin = float(meta.get("Rin", 0.0))
	Rout = float(meta.get("Rout", 1.0))
	V0 = float(meta.get("voltage", 0.0))

	if Rin > 0.0 and Rout > Rin:
		E_ideal_max = V0 / (Rin * np.log(Rout / Rin))
	else:
		E_ideal_max = 0.0

	metrics["E_ideal_Rin"] = float(E_ideal_max)

	if E_ideal_max != 0.0:
		metrics["enhancement_vs_ideal"] = float(Emax / (E_ideal_max + 1.0e-25))
	else:
		metrics["enhancement_vs_ideal"] = 0.0

	rd = float(meta.get("defect_radius", 0.0))

	if rd > 0.0:
		metrics["defect_radius_m"] = float(rd)
		metrics["defect_center_x"] = float(meta.get("defect_cx", 0.0))
		metrics["defect_center_y"] = float(meta.get("defect_cy", 0.0))

	return metrics


def run_case(
	Rin=2.0e-3,
	Rout=10.0e-3,
	V0=15000.0,
	epsr=2.3,
	defect_type="bubble",
	defect_radius=0.5e-3,
	defect_center_x=None,
	defect_center_y=None,
	defect_epsr=None,
	nx=720,
	ny=720,
	outdir="coax_defect/outputs",
	name=None,
	make_plots=True,
	max_iter=20000,
	tol=1e-6,
	omega=1.6,
):

	if not os.path.exists(outdir):
		os.makedirs(outdir)

	out_case = case_folder(
		Rin,
		Rout,
		V0,
		epsr,
		defect_type,
		defect_radius,
		defect_epsr,
		outdir,
		name,
	)

	if not os.path.exists(out_case):
		os.makedirs(out_case)

	problem = build_problem(
		voltage=V0,
		inner_radius=Rin,
		outer_radius=Rout,
		eps_r=epsr,
		defect_type=defect_type,
		defect_radius=defect_radius,
		defect_center_x=defect_center_x,
		defect_center_y=defect_center_y,
		defect_epsr=defect_epsr,
		nx=nx,
		ny=ny,
	)

	V = solve_potential(problem, max_iter=max_iter, tol=tol, omega=omega)
	Ex, Ey, Emag = compute_field(problem, V)

	np.savetxt(os.path.join(out_case, "potential.csv"), V, delimiter=",")
	np.savetxt(os.path.join(out_case, "Emag.csv"), Emag, delimiter=",")

	metrics = compute_metrics(problem, V, Emag)

	with open(os.path.join(out_case, "metrics.json"), "w") as f:
		json.dump(metrics, f, indent=2)

	if make_plots:
		# Import plotting tools lazily so that pure-metrics runs avoid Matplotlib overhead
		try:
			from .viz import plot_overview, plot_zoom_near_defect, plot_radial_diagnostics
		except ImportError:  # pragma: no cover
			from viz import plot_overview, plot_zoom_near_defect, plot_radial_diagnostics  # type: ignore

		title = (
			"Coaxial cable V0=%.0f kV, Rin=%.2f mm, Rout=%.2f mm, epsr=%.2f, defect=%s"
			% (
				V0 / 1000.0,
				Rin * 1000.0,
				Rout * 1000.0,
				epsr,
				defect_type,
			)
		)

		plot_overview(
			problem,
			V,
			title=title,
			output_path=os.path.join(out_case, "overview.png"),
		)

		plot_zoom_near_defect(
			problem,
			V,
			output_path=os.path.join(out_case, "zoom_defect.png"),
			title="Zoom near defect",
		)

		plot_radial_diagnostics(
			problem,
			V,
			phi_deg=0.0,
			output_path=os.path.join(out_case, "radial_profiles.png"),
			title="Radial |E|(r) vs ideal",
		)

	return out_case


def _parse_args(argv=None):
	p = argparse.ArgumentParser(
		description="Solve electrostatic field for a coaxial cable cross-section with an optional dielectric defect.",
	)
	p.add_argument("--Rin", type=float, default=2.0e-3, help="Inner conductor radius (m)")
	p.add_argument("--Rout", type=float, default=10.0e-3, help="Outer conductor radius (m)")
	p.add_argument("--V0", type=float, default=15000.0, help="Applied voltage on inner conductor (V)")
	p.add_argument("--epsr", type=float, default=2.3, help="Relative permittivity of XLPE insulation")

	p.add_argument(
		"--defect-type",
		type=str,
		default="bubble",
		choices=["none", "bubble", "inclusion"],
		help="Defect type",
	)
	p.add_argument("--defect-radius", type=float, default=0.5e-3, help="Defect radius (m)")
	p.add_argument("--defect-epsr", type=float, default=None, help="Defect relative permittivity (for inclusion)")
	p.add_argument("--defect-cx", type=float, default=None, help="Defect center x (m). Default: mid-radius on +x axis")
	p.add_argument("--defect-cy", type=float, default=None, help="Defect center y (m). Default: mid-radius on +x axis")

	p.add_argument("--nx", type=int, default=720, help="Grid points in x")
	p.add_argument("--ny", type=int, default=720, help="Grid points in y")
	p.add_argument("--outdir", type=str, default="coax_defect/outputs", help="Output directory")
	p.add_argument("--name", type=str, default=None, help="Optional suffix for output folder name")
	p.add_argument("--no-plots", action="store_true", help="Do not generate PNG figures")

	p.add_argument("--max-iter", type=int, default=20000, help="Max solver iterations")
	p.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance (max delta)")
	p.add_argument("--omega", type=float, default=1.6, help="SOR relaxation factor")

	return p.parse_args(argv)


if __name__ == "__main__":
	args = _parse_args()
	# For baseline runs, ignore defect-specific settings.
	if args.defect_type == "none":
		defect_radius = 0.0
		defect_epsr = None
		defect_cx = None
		defect_cy = None
	else:
		defect_radius = float(args.defect_radius)
		defect_epsr = args.defect_epsr
		defect_cx = args.defect_cx
		defect_cy = args.defect_cy

	path = run_case(
		Rin=float(args.Rin),
		Rout=float(args.Rout),
		V0=float(args.V0),
		epsr=float(args.epsr),
		defect_type=str(args.defect_type),
		defect_radius=defect_radius,
		defect_center_x=defect_cx,
		defect_center_y=defect_cy,
		defect_epsr=defect_epsr,
		nx=int(args.nx),
		ny=int(args.ny),
		outdir=str(args.outdir),
		name=args.name,
		make_plots=(not args.no_plots),
		max_iter=int(args.max_iter),
		tol=float(args.tol),
		omega=float(args.omega),
	)
	print("Outputs in:", path)
