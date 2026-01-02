from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
try:
	import plotly.graph_objects as go  # optional
except Exception:
	go = None

from .geometry import GeometryProblem
from .field import compute_field


def _extent_from_grid(problem: GeometryProblem) -> Tuple[float, float, float, float]:
	g = problem.grid
	return (g.x[0], g.x[-1], g.y[0], g.y[-1])


def plot_static_panels(
	problem: GeometryProblem,
	V: np.ndarray,
	stream_density: float = 1.3,
	output_path: Optional[str] = None,
	title: Optional[str] = None,
) -> plt.Figure:
	g = problem.grid
	Ex, Ey, Emag = compute_field(g, V)
	X, Y = g.X, g.Y
	extent = _extent_from_grid(problem)

	fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

	# Potential contours
	ax = axs[0]
	c = ax.contourf(X, Y, V, levels=30, cmap="viridis")
	ax.contour(X, Y, V, colors="k", linewidths=0.2, levels=15)
	ax.set_title("Potential V (V)")
	ax.set_xlabel("x (m)" if not g.axisymmetric else "r (m)")
	ax.set_ylabel("y (m)" if not g.axisymmetric else "z (m)")
	cb = fig.colorbar(c, ax=ax, shrink=0.9)
	# Electrode overlay
	ax.contourf(X, Y, problem.electrode_mask, levels=[0.5, 1.5], colors=["#ff0066"], alpha=0.6)

	# E magnitude
	ax = axs[1]
	# Use logarithmic colormap for dynamic range
	Em_plot = np.clip(Emag, 1e-1, np.nanmax(Emag))
	im = ax.imshow(
		Em_plot,
		origin="lower",
		extent=extent,
		aspect="auto",
		norm=LogNorm(vmin=max(np.nanmin(Em_plot), 1e-1), vmax=np.nanmax(Em_plot)),
		cmap="inferno",
	)
	ax.set_title("|E| (V/m)")
	ax.set_xlabel("x (m)" if not g.axisymmetric else "r (m)")
	ax.set_ylabel("y (m)" if not g.axisymmetric else "z (m)")
	fig.colorbar(im, ax=ax, shrink=0.9)
	ax.contour(X, Y, problem.electrode_mask, levels=[0.5], colors="white", linewidths=1.0)

	# Streamlines over potential background
	ax = axs[2]
	ax.contourf(X, Y, V, levels=20, cmap="viridis", alpha=0.7)
	# Streamplot requires uniform grids; resample Ex,Ey onto uniform tensors
	def bilinear_grid(F: np.ndarray, xs: np.ndarray, ys: np.ndarray, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
		Fi = np.zeros((yq.size, xq.size), dtype=float)
		# Precompute indices and weights along x
		ix1 = np.clip(np.searchsorted(xs, xq, side="right") - 1, 0, xs.size - 2)
		tx = (xq - xs[ix1]) / np.maximum(xs[ix1 + 1] - xs[ix1], 1e-16)
		# Along y
		iy1 = np.clip(np.searchsorted(ys, yq, side="right") - 1, 0, ys.size - 2)
		ty = (yq - ys[iy1]) / np.maximum(ys[iy1 + 1] - ys[iy1], 1e-16)
		for jj, (j0, wy) in enumerate(zip(iy1, ty)):
			f0 = (1 - wy) * F[j0, :] + wy * F[j0 + 1, :]
			# Now interpolate along x for this row using vectorized blend
			left = f0[ix1]
			right = f0[ix1 + 1]
			Fi[jj, :] = (1 - tx) * left + tx * right
		return Fi
	xu = np.linspace(g.x[0], g.x[-1], min(100, g.nx))
	yu = np.linspace(g.y[0], g.y[-1], min(100, g.ny))
	Exu = bilinear_grid(Ex, g.x, g.y, xu, yu)
	Eyu = bilinear_grid(Ey, g.x, g.y, xu, yu)
	ax.streamplot(xu, yu, Exu, Eyu, density=stream_density, color="k", linewidth=0.8)
	ax.set_title("Streamlines of E")
	ax.set_xlabel("x (m)" if not g.axisymmetric else "r (m)")
	ax.set_ylabel("y (m)" if not g.axisymmetric else "z (m)")
	ax.contour(X, Y, problem.electrode_mask, levels=[0.5], colors="white", linewidths=1.0)

	if title:
		fig.suptitle(title, fontsize=13)
	if output_path:
		fig.savefig(output_path, dpi=220)
	return fig


def plot_interactive_plotly(
	problem: GeometryProblem,
	V: np.ndarray,
	add_streams: bool = True,
	max_quiver: int = 24,
) -> "go.Figure":
	if go is None:
		raise RuntimeError("plotly is not installed. Install plotly to use interactive visualization.")
	g = problem.grid
	Ex, Ey, Emag = compute_field(g, V)
	# Potential heatmap
	fig = go.Figure()
	fig.add_trace(
		go.Heatmap(
			x=g.x,
			y=g.y,
			z=V,
			colorscale="Viridis",
			colorbar=dict(title="V (V)"),
			showscale=True,
		)
	)
	# Equipotential lines
	fig.add_trace(go.Contour(x=g.x, y=g.y, z=V, contours=dict(coloring="lines", showlabels=False), line=dict(color="black", width=1)))
	# Quiver (downsampled)
	if max_quiver > 0:
		ii = np.linspace(0, g.nx - 1, min(max_quiver, g.nx)).astype(int)
		jj = np.linspace(0, g.ny - 1, min(max_quiver, g.ny)).astype(int)
		Xq, Yq = np.meshgrid(g.x[ii], g.y[jj], indexing="xy")
		Exq, Eyq = Ex[np.ix_(jj, ii)], Ey[np.ix_(jj, ii)]
		scale = np.nanmax(np.hypot(Exq, Eyq)) + 1e-9
		ux, uy = Exq / scale, Eyq / scale
		fig.add_trace(
			go.Cone(
				x=Xq.ravel(),
				y=Yq.ravel(),
				z=np.zeros_like(Xq.ravel()),
				u=ux.ravel(),
				v=uy.ravel(),
				w=np.zeros_like(ux.ravel()),
				sizemode="absolute",
				sizeref=0.5,
				anchor="tip",
				colorscale="Greys",
				showscale=False,
				opacity=0.7,
			)
		)
	fig.update_layout(
		title="Potential and Field",
		xaxis_title="x (m)" if not g.axisymmetric else "r (m)",
		yaxis_title="y (m)" if not g.axisymmetric else "z (m)",
	)
	return fig


def plot_panels_zoom_tip(
	problem: GeometryProblem,
	V: np.ndarray,
	radius_factor: float = 25.0,
	height_factor: float = 30.0,
	output_path: Optional[str] = None,
	title: Optional[str] = None,
) -> plt.Figure:
	"""
	Zoomed panels near the needle tip for axisymmetric needle–plane geometry.
	Window: r in [0, radius_factor * r_tip], z in [gap - height_factor*r_tip, gap + height_factor*r_tip]
	"""
	g = problem.grid
	meta = getattr(problem, "meta", {})
	r_tip = float(meta.get("needle_tip_radius", 5e-5))
	gap = float(meta.get("gap", 1e-3))
	rmax = radius_factor * r_tip
	zmin = max(g.y[0], gap - height_factor * r_tip)
	zmax = min(g.y[-1], gap + height_factor * r_tip)

	# Determine slice indices
	i_max = int(np.clip(np.searchsorted(g.x, rmax, side="right"), 2, g.nx))
	j0 = int(np.clip(np.searchsorted(g.y, zmin, side="left"), 0, g.ny - 2))
	j1 = int(np.clip(np.searchsorted(g.y, zmax, side="right"), 2, g.ny))

	Vs = V[j0:j1, :i_max]
	Xs, Ys = np.meshgrid(g.x[:i_max], g.y[j0:j1], indexing="xy")
	from .field import compute_field as _cf
	Ex, Ey, Emag = _cf(g, V)  # compute on full grid once
	Exs, Eys, Es = Ex[j0:j1, :i_max], Ey[j0:j1, :i_max], Emag[j0:j1, :i_max]

	fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
	# Potential with power-law norm to enhance low values near plane
	ax = axs[0]
	c = ax.pcolormesh(Xs, Ys, Vs, shading="auto", cmap="viridis", norm=PowerNorm(gamma=0.6))
	ax.contour(Xs, Ys, Vs, colors="k", linewidths=0.3, levels=12)
	ax.set_title("V near tip")
	ax.set_xlabel("r (m)")
	ax.set_ylabel("z (m)")
	fig.colorbar(c, ax=ax, shrink=0.9)

	# |E| log plot
	ax = axs[1]
	Em_plot = np.clip(Es, 1e-1, np.nanmax(Es))
	im = ax.pcolormesh(Xs, Ys, Em_plot, shading="auto", cmap="inferno", norm=LogNorm(vmin=1e2, vmax=np.nanmax(Em_plot)))
	ax.set_title("|E| near tip (V/m)")
	ax.set_xlabel("r (m)")
	ax.set_ylabel("z (m)")
	fig.colorbar(im, ax=ax, shrink=0.9)

	# Local streamlines (resample onto uniform fine mesh in window)
	ax = axs[2]
	ax.pcolormesh(Xs, Ys, Vs, shading="auto", cmap="viridis", alpha=0.6)
	xu = np.linspace(g.x[0], rmax, 80)
	yu = np.linspace(g.y[j0], g.y[j1 - 1], 120)
	def bilinear_grid(F: np.ndarray, xs: np.ndarray, ys: np.ndarray, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
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
	Exu = bilinear_grid(Ex, g.x, g.y, xu, yu)
	Eyu = bilinear_grid(Ey, g.x, g.y, xu, yu)
	ax.streamplot(xu, yu, Exu, Eyu, density=1.2, color="k", linewidth=0.8)
	ax.set_title("Streamlines near tip")
	ax.set_xlabel("r (m)")
	ax.set_ylabel("z (m)")

	if title:
		fig.suptitle(title, fontsize=12)
	if output_path:
		fig.savefig(output_path, dpi=240)
	return fig


