from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm

from dielectric_breakdown.geometry import GeometryProblem
from dielectric_breakdown.field import compute_field


def _extent(problem: GeometryProblem) -> Tuple[float, float, float, float]:
	g = problem.grid
	return (g.x[0], g.x[-1], g.y[0], g.y[-1])


def _material_mask(problem: GeometryProblem, threshold: float = 1.05) -> np.ndarray | None:
	eps = getattr(problem, "epsilon", None)
	if eps is None:
		return None
	return (eps > threshold).astype(float)


def plot_overview_with_materials(
	problem: GeometryProblem,
	V: np.ndarray,
	title: Optional[str] = None,
	output_path: Optional[str] = None,
) -> plt.Figure:
	"""
	Overview panels tailored for triple junction:
	- V contours with metal and dielectric overlays
	- |E| log heatmap
	- Streamlines
	"""
	g = problem.grid
	X, Y = g.X, g.Y
	Ex, Ey, Emag = compute_field(g, V)
	mmask = _material_mask(problem)
	extent = _extent(problem)

	fig, axs = plt.subplots(1, 3, figsize=(17, 5.0), constrained_layout=True)

	# Potential
	ax = axs[0]
	c = ax.contourf(X, Y, V, levels=36, cmap="viridis")
	ax.contour(X, Y, V, colors="k", linewidths=0.25, levels=16)
	ax.set_title("Potential V (V)")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")
	fig.colorbar(c, ax=ax, shrink=0.9)
	# Overlays
	ax.contourf(X, Y, problem.electrode_mask, levels=[0.5, 1.5], colors=["#ff0066"], alpha=0.65)
	if mmask is not None:
		ax.contourf(X, Y, mmask, levels=[0.5, 1.5], colors=["#6699ff"], alpha=0.25)

	# |E|
	ax = axs[1]
	Em_plot = np.clip(Emag, 1e-2, np.nanmax(Emag))
	im = ax.imshow(
		Em_plot,
		origin="lower",
		extent=extent,
		aspect="auto",
		norm=LogNorm(vmin=max(np.nanmin(Em_plot), 1e-2), vmax=np.nanmax(Em_plot)),
		cmap="inferno",
	)
	ax.set_title("|E| (V/m)")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")
	fig.colorbar(im, ax=ax, shrink=0.9)
	ax.contour(X, Y, problem.electrode_mask, levels=[0.5], colors="white", linewidths=1.0)
	if mmask is not None:
		ax.contour(X, Y, mmask, levels=[0.5], colors="#66b2ff", linewidths=1.0)

	# Streamlines
	ax = axs[2]
	ax.contourf(X, Y, V, levels=28, cmap="viridis", alpha=0.75)
	# resample field onto uniform tensor for streamplot
	xu = np.linspace(g.x[0], g.x[-1], min(140, g.nx))
	yu = np.linspace(g.y[0], g.y[-1], min(140, g.ny))
	def _bilinear(F: np.ndarray, xs: np.ndarray, ys: np.ndarray, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
		Fi = np.zeros((yq.size, xq.size), dtype=float)
		ix1 = np.clip(np.searchsorted(xs, xq, side="right") - 1, 0, xs.size - 2)
		tx = (xq - xs[ix1]) / np.maximum(xs[ix1 + 1] - xs[ix1], 1e-16)
		iy1 = np.clip(np.searchsorted(ys, yq, side="right") - 1, 0, ys.size - 2)
		ty = (yq - ys[iy1]) / np.maximum(ys[iy1 + 1] - ys[iy1], 1e-16)
		for jj, (j0, wy) in enumerate(zip(iy1, ty)):
			f0 = (1 - wy) * F[j0, :] + wy * F[j0 + 1, :]
			left = f0[ix1]
			right = f0[ix1 + 1]
			Fi[jj, :] = (1 - tx) * left + tx * right
		return Fi
	Exu = _bilinear(Ex, g.x, g.y, xu, yu)
	Eyu = _bilinear(Ey, g.x, g.y, xu, yu)
	ax.streamplot(xu, yu, Exu, Eyu, density=1.25, color="k", linewidth=0.9)
	ax.set_title("Streamlines")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")
	ax.contour(X, Y, problem.electrode_mask, levels=[0.5], colors="white", linewidths=1.0)
	if mmask is not None:
		ax.contour(X, Y, mmask, levels=[0.5], colors="#66b2ff", linewidths=1.0)

	if title:
		fig.suptitle(title, fontsize=13)
	if output_path:
		fig.savefig(output_path, dpi=240)
	return fig


def plot_zoom_near_junction(
	problem: GeometryProblem,
	V: np.ndarray,
	which: str = "left",
	x_span: float | None = None,
	y_span: float | None = None,
	output_path: Optional[str] = None,
	title: Optional[str] = None,
) -> plt.Figure:
	g = problem.grid
	meta = getattr(problem, "meta", {})
	td = float(meta.get("dielectric_thickness", (g.y[-1] - g.y[0]) * 0.25))
	x0 = float(meta.get("junction_x_left" if which == "left" else "junction_x_right", 0.5 * (g.x[0] + g.x[-1])))
	if x_span is None:
		# focus tightly around the junction
		x_span = max(10.0 * np.min(np.diff(g.x)), 0.02 * (g.x[-1] - g.x[0]))
	if y_span is None:
		y_span = max(15.0 * np.min(np.diff(g.y)), 0.03 * (g.y[-1] - g.y[0]))
	xmin, xmax = x0 - x_span, x0 + x_span
	ymin, ymax = max(g.y[0], td - y_span), min(g.y[-1], td + y_span)
	# Crop indices
	i0 = int(np.clip(np.searchsorted(g.x, xmin, side="left"), 0, g.nx - 2))
	i1 = int(np.clip(np.searchsorted(g.x, xmax, side="right"), 1, g.nx))
	j0 = int(np.clip(np.searchsorted(g.y, ymin, side="left"), 0, g.ny - 2))
	j1 = int(np.clip(np.searchsorted(g.y, ymax, side="right"), 1, g.ny))
	Xs, Ys = np.meshgrid(g.x[i0:i1], g.y[j0:j1], indexing="xy")
	Vs = V[j0:j1, i0:i1]
	Ex, Ey, Em = compute_field(g, V)
	Exs, Eys, Ems = Ex[j0:j1, i0:i1], Ey[j0:j1, i0:i1], Em[j0:j1, i0:i1]
	eps = getattr(problem, "epsilon", None)
	mmask = None if eps is None else (eps[j0:j1, i0:i1] > 1.05).astype(float)

	fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
	# Potential
	ax = axs[0]
	c = ax.pcolormesh(Xs, Ys, Vs, shading="auto", cmap="viridis", norm=PowerNorm(gamma=0.65))
	ax.contour(Xs, Ys, Vs, colors="k", linewidths=0.35, levels=14)
	ax.set_title("V near triple junction")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")
	fig.colorbar(c, ax=ax, shrink=0.9)
	ax.contour(Xs, Ys, problem.electrode_mask[j0:j1, i0:i1], levels=[0.5], colors="#ff0066", linewidths=1.0)
	if mmask is not None:
		ax.contour(Xs, Ys, mmask, levels=[0.5], colors="#66b2ff", linewidths=1.0)

	# |E|
	ax = axs[1]
	Em_plot = np.clip(Ems, 1e-1, np.nanmax(Ems))
	im = ax.pcolormesh(Xs, Ys, Em_plot, shading="auto", cmap="inferno", norm=LogNorm(vmin=1e1, vmax=np.nanmax(Em_plot)))
	ax.set_title("|E| near junction (V/m)")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")
	fig.colorbar(im, ax=ax, shrink=0.9)
	ax.contour(Xs, Ys, problem.electrode_mask[j0:j1, i0:i1], levels=[0.5], colors="white", linewidths=1.0)

	# Streamlines
	ax = axs[2]
	ax.pcolormesh(Xs, Ys, Vs, shading="auto", cmap="viridis", alpha=0.65)
	xu = np.linspace(Xs[0, 0], Xs[0, -1], min(160, Xs.shape[1] * 2))
	yu = np.linspace(Ys[0, 0], Ys[-1, 0], min(160, Ys.shape[0] * 2))
	def _bilinear(F: np.ndarray, xs: np.ndarray, ys: np.ndarray, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
		Fi = np.zeros((yq.size, xq.size), dtype=float)
		ix1 = np.clip(np.searchsorted(xs, xq, side="right") - 1, 0, xs.size - 2)
		tx = (xq - xs[ix1]) / np.maximum(xs[ix1 + 1] - xs[ix1], 1e-16)
		iy1 = np.clip(np.searchsorted(ys, yq, side="right") - 1, 0, ys.size - 2)
		ty = (yq - ys[iy1]) / np.maximum(ys[iy1 + 1] - ys[iy1], 1e-16)
		for jj, (j0i, wy) in enumerate(zip(iy1, ty)):
			f0 = (1 - wy) * F[j0i, :] + wy * F[j0i + 1, :]
			left = f0[ix1]
			right = f0[ix1 + 1]
			Fi[jj, :] = (1 - tx) * left + tx * right
		return Fi
	def _nearest_bool(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
		ii = np.clip(np.searchsorted(xs, xq, side="right") - 1, 0, xs.size - 1)
		jj = np.clip(np.searchsorted(ys, yq, side="right") - 1, 0, ys.size - 1)
		M = np.zeros((yq.size, xq.size), dtype=bool)
		for r, j0i in enumerate(jj):
			M[r, :] = mask[j0i, ii]
		return M
	Exu = _bilinear(Exs, g.x[i0:i1], g.y[j0:j1], xu, yu)
	Eyu = _bilinear(Eys, g.x[i0:i1], g.y[j0:j1], xu, yu)
	# Prevent lines from starting/propagating inside the metal by masking with NaNs
	mask_e = _nearest_bool(problem.electrode_mask[j0:j1, i0:i1], g.x[i0:i1], g.y[j0:j1], xu, yu)
	Exu[mask_e] = np.nan
	Eyu[mask_e] = np.nan
	# Concentrate seeds near the triple junction so lines visibly emanate from it
	x0_seed = x0
	y0_seed = td
	r_base = 2.5 * max(np.min(np.diff(g.x)), np.min(np.diff(g.y)))
	angles = np.deg2rad(np.linspace(20.0, 160.0, 18))  # mostly in air above the interface
	seeds = np.column_stack([x0_seed + r_base * np.cos(angles), y0_seed + r_base * np.sin(angles)])
	# Keep seeds within window and outside metal
	valid = (seeds[:, 0] >= xu[0]) & (seeds[:, 0] <= xu[-1]) & (seeds[:, 1] >= yu[0]) & (seeds[:, 1] <= yu[-1])
	ii_seed = np.clip(np.searchsorted(xu, seeds[:, 0], side="right") - 1, 0, xu.size - 1)
	jj_seed = np.clip(np.searchsorted(yu, seeds[:, 1], side="right") - 1, 0, yu.size - 1)
	valid &= ~mask_e[jj_seed, ii_seed]
	seeds = seeds[valid]
	ax.streamplot(
		xu,
		yu,
		Exu,
		Eyu,
		density=1.0,
		color="k",
		linewidth=0.9,
		start_points=seeds,
		minlength=0.02,
		maxlength=5.0,
		integration_direction="both",
	)
	ax.set_title("Local streamlines (seeded at junction)")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")

	if title:
		fig.suptitle(title, fontsize=12)
	if output_path:
		fig.savefig(output_path, dpi=260)
	return fig


def plot_singularity_diagnostics(
	problem: GeometryProblem,
	V: np.ndarray,
	radii: np.ndarray | None = None,
	output_path: Optional[str] = None,
	title: Optional[str] = None,
) -> plt.Figure:
	"""
	Quantify field amplification scaling near the triple junction by sampling |E|
	versus radial distance along rays above (air) and below (dielectric).
	We fit slope d log|E| / d log r which should asymptote to (λ-1), where V~r^λ.
	"""
	g = problem.grid
	meta = getattr(problem, "meta", {})
	td = float(meta.get("dielectric_thickness", (g.y[-1] - g.y[0]) * 0.25))
	x0 = float(meta.get("junction_x_left", 0.5 * (g.x[0] + g.x[-1])))
	y0 = td
	Ex, Ey, Em = compute_field(g, V)
	# helper: evaluate Em at arbitrary points via bilinear interpolation
	def interp(M: np.ndarray, x: float, y: float) -> float:
		xs, ys = g.x, g.y
		i1 = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, xs.size - 2)
		j1 = np.clip(np.searchsorted(ys, y, side="right") - 1, 0, ys.size - 2)
		x0i, x1i = xs[i1], xs[i1 + 1]
		y0i, y1i = ys[j1], ys[j1 + 1]
		tx = 0.0 if x1i == x0i else (x - x0i) / (x1i - x0i)
		ty = 0.0 if y1i == y0i else (y - y0i) / (y1i - y0i)
		f00 = M[j1, i1]
		f10 = M[j1, i1 + 1]
		f01 = M[j1 + 1, i1]
		f11 = M[j1 + 1, i1 + 1]
		return float((1 - ty) * ((1 - tx) * f00 + tx * f10) + ty * ((1 - tx) * f01 + tx * f11))
	# radii
	if radii is None:
		# pick radii spanning roughly one decade above the minimum grid spacing near the interface
		dr = max(np.min(np.diff(g.x)), np.min(np.diff(g.y)))
		radii = np.geomspace(dr * 0.7, 30.0 * dr, 24)
	# Rays: air (90°), dielectric (-90°), shallow air (60°), shallow dielectric (-60°)
	angles = np.deg2rad([90.0, -90.0, 60.0, -60.0])
	labels = ["air (up)", "dielectric (down)", "air 60°", "dielectric -60°"]
	color = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]
	fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.6), constrained_layout=True)
	for ang, lab, col in zip(angles, labels, color):
		vals = []
		for r in radii:
			xp = x0 + r * np.cos(ang)
			yp = y0 + r * np.sin(ang)
			# Skip out-of-domain points
			if not (g.x[0] <= xp <= g.x[-1] and g.y[0] <= yp <= g.y[-1]):
				vals.append(np.nan)
				continue
			vals.append(interp(Em, xp, yp))
		vals = np.array(vals, dtype=float)
		ax.loglog(radii, vals, "-o", ms=3, lw=1.2, label=lab, color=col, alpha=0.9)
	# Local slope using finite differences on log-log values
	def slope(y: np.ndarray, x: np.ndarray) -> np.ndarray:
		ly = np.log(y)
		lx = np.log(x)
		s = np.gradient(ly, lx)
		return s
	# Show slope for the first curve as reference on twin axis
	ref_vals = np.array([interp(Em, x0, td + r) for r in radii])
	ax2 = ax.twinx()
	ax2.plot(radii, slope(ref_vals, radii), color="#888888", lw=1.0, alpha=0.8)
	ax2.set_ylabel("d log|E| / d log r (air, 90°)")
	ax.set_xlabel("r from junction (m)")
	ax.set_ylabel("|E| (V/m)")
	ax.grid(True, which="both", linestyle="--", alpha=0.3)
	ax.legend()
	if title:
		ax.set_title(title)
	if output_path:
		fig.savefig(output_path, dpi=260)
	return fig


