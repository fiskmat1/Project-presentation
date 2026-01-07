import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .geometry import compute_field


def _extent(problem):
	x = problem["x"]
	y = problem["y"]
	return (x[0], x[-1], y[0], y[-1])


def _extract_meta(problem):
	return problem.get("meta", {}) or {}


def _material_labels(problem, eps_thr=1.05):
	eps = problem.get("epsilon")
	if eps is None:
		return None
	labels = np.zeros_like(eps, dtype=int)
	labels[eps >= eps_thr] = 1
	meta = _extract_meta(problem)
	rd = float(meta.get("defect_radius", 0.0))
	if rd > 0.0:
		cx = float(meta.get("cx", 0.0))
		cy = float(meta.get("cy", 0.0))
		dcx = float(meta.get("defect_cx", cx))
		dcy = float(meta.get("defect_cy", cy))
		X = problem["X"]
		Y = problem["Y"]
		Rd = np.sqrt((X - dcx) ** 2 + (Y - dcy) ** 2)
		labels[np.abs(Rd) <= rd] = 2
	return labels


def _draw_circles(ax, meta, **kwargs):
	cx = float(meta.get("cx", 0.0))
	cy = float(meta.get("cy", 0.0))
	Rin = float(meta.get("Rin", 0.0))
	Rout = float(meta.get("Rout", 0.0))
	th = np.linspace(0.0, 2.0 * np.pi, 361)
	ax.plot(cx + Rin * np.cos(th), cy + Rin * np.sin(th), **kwargs)
	ax.plot(cx + Rout * np.cos(th), cy + Rout * np.sin(th), **kwargs)


def plot_overview(problem, V, title=None, output_path=None):
	X = problem["X"]
	Y = problem["Y"]
	Ex, Ey, Em = compute_field(problem, V)
	extent = _extent(problem)
	meta = _extract_meta(problem)
	labels = _material_labels(problem)
	x = problem["x"]
	y = problem["y"]
	nx = int(problem["nx"])
	ny = int(problem["ny"])

	fig, axs = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)

	ax = axs[0]
	c = ax.contourf(X, Y, V, levels=34, cmap="viridis")
	ax.contour(X, Y, V, levels=16, colors="k", linewidths=0.25)
	ax.set_title("Potential V (V)")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")
	fig.colorbar(c, ax=ax, shrink=0.9)
	if labels is not None:
		ax.contourf(X, Y, (labels == 1).astype(float), levels=[0.5, 1.5], colors=["#5da5ff"], alpha=0.20)
		ax.contourf(X, Y, (labels == 2).astype(float), levels=[0.5, 1.5], colors=["#ffcc00"], alpha=0.40)
	ax.contour(X, Y, problem["electrode_mask"], levels=[0.5], colors="#ff0066", linewidths=1.0)
	_draw_circles(ax, meta, color="#222222", lw=0.8, alpha=0.7)

	ax = axs[1]
	Em_plot = np.clip(Em, 1e-2, np.nanmax(Em))
	im = ax.imshow(
		Em_plot,
		origin="lower",
		extent=extent,
		aspect="equal",
		norm=LogNorm(vmin=max(np.nanmin(Em_plot), 1e-2), vmax=np.nanmax(Em_plot)),
		cmap="inferno",
	)
	ax.set_title("|E| (V/m, log)")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")
	fig.colorbar(im, ax=ax, shrink=0.9)
	ax.contour(X, Y, problem["electrode_mask"], levels=[0.5], colors="white", linewidths=1.0)
	if labels is not None:
		ax.contour(X, Y, (labels == 2).astype(float), levels=[0.5], colors="#ffcc00", linewidths=1.0)
	_draw_circles(ax, meta, color="white", lw=0.6, alpha=0.7)

	ax = axs[2]
	ax.contourf(X, Y, V, levels=28, cmap="viridis", alpha=0.70)
	xu = np.linspace(x[0], x[-1], min(160, nx))
	yu = np.linspace(y[0], y[-1], min(160, ny))

	def _bilinear(F, xs, ys, xq, yq):
		Fi = np.zeros((yq.size, xq.size), dtype=float)
		ix1 = np.clip(np.searchsorted(xs, xq, side="right") - 1, 0, xs.size - 2)
		tx = (xq - xs[ix1]) / np.maximum(xs[ix1 + 1] - xs[ix1], 1.0e-16)
		iy1 = np.clip(np.searchsorted(ys, yq, side="right") - 1, 0, ys.size - 2)
		ty = (yq - ys[iy1]) / np.maximum(ys[iy1 + 1] - ys[iy1], 1.0e-16)
		for jj, (j0, wy) in enumerate(zip(iy1, ty)):
			f0 = (1.0 - wy) * F[j0, :] + wy * F[j0 + 1, :]
			left = f0[ix1]
			right = f0[ix1 + 1]
			Fi[jj, :] = (1.0 - tx) * left + tx * right
		return Fi

	Exu = _bilinear(Ex, x, y, xu, yu)
	Eyu = _bilinear(Ey, x, y, xu, yu)

	def _nearest_bool(mask, xs, ys, xq, yq):
		ii = np.clip(np.searchsorted(xs, xq, side="right") - 1, 0, xs.size - 1)
		jj = np.clip(np.searchsorted(ys, yq, side="right") - 1, 0, ys.size - 1)
		M = np.zeros((yq.size, xq.size), dtype=bool)
		for r, j0 in enumerate(jj):
			M[r, :] = mask[j0, ii]
		return M

	mask_e = _nearest_bool(problem["electrode_mask"], x, y, xu, yu)
	Exu[mask_e] = np.nan
	Eyu[mask_e] = np.nan
	ax.streamplot(xu, yu, Exu, Eyu, density=1.20, color="k", linewidth=0.9)
	ax.set_title("Streamlines")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")
	ax.contour(X, Y, problem["electrode_mask"], levels=[0.5], colors="white", linewidths=1.0)
	_draw_circles(ax, meta, color="white", lw=0.6, alpha=0.7)

	if title:
		fig.suptitle(title, fontsize=13)
	if output_path:
		fig.savefig(output_path, dpi=250)
	return fig


def plot_zoom_near_defect(problem, V, radius_factor=4.0, output_path=None, title=None):
	x = problem["x"]
	y = problem["y"]
	X = problem["X"]
	Y = problem["Y"]
	Ex, Ey, Em = compute_field(problem, V)
	meta = _extract_meta(problem)
	xd = float(meta.get("defect_cx", 0.5 * (x[0] + x[-1])))
	yd = float(meta.get("defect_cy", 0.5 * (y[0] + y[-1])))
	rd = float(meta.get("defect_radius", 0.0))
	if x.size > 1:
		dx = float(np.min(np.diff(x)))
	else:
		dx = 1.0
	if y.size > 1:
		dy = float(np.min(np.diff(y)))
	else:
		dy = 1.0
	rw = max(1.0, radius_factor) * max(rd, 3.0 * max(dx, dy))
	xmin = xd - rw
	xmax = xd + rw
	ymin = yd - rw
	ymax = yd + rw
	nx = int(problem["nx"])
	ny = int(problem["ny"])
	i0 = int(np.clip(np.searchsorted(x, xmin, side="left"), 0, nx - 2))
	i1 = int(np.clip(np.searchsorted(x, xmax, side="right"), 1, nx))
	j0 = int(np.clip(np.searchsorted(y, ymin, side="left"), 0, ny - 2))
	j1 = int(np.clip(np.searchsorted(y, ymax, side="right"), 1, ny))
	Xs, Ys = np.meshgrid(x[i0:i1], y[j0:j1], indexing="xy")
	Vs = V[j0:j1, i0:i1]
	Exs = Ex[j0:j1, i0:i1]
	Eys = Ey[j0:j1, i0:i1]
	Ems = Em[j0:j1, i0:i1]

	fig, axs = plt.subplots(1, 3, figsize=(14.2, 4.6), constrained_layout=True)

	ax = axs[0]
	c = ax.pcolormesh(Xs, Ys, Vs, shading="auto", cmap="viridis")
	ax.contour(Xs, Ys, Vs, colors="k", linewidths=0.35, levels=14)
	ax.set_title("V near defect")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")
	fig.colorbar(c, ax=ax, shrink=0.9)

	ax = axs[1]
	Em_plot = np.clip(Ems, 1e-1, np.nanmax(Ems))
	im = ax.pcolormesh(Xs, Ys, Em_plot, shading="auto", cmap="inferno", norm=LogNorm(vmin=1.0e-1, vmax=np.nanmax(Em_plot)))
	ax.set_title("|E| near defect (V/m)")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")
	fig.colorbar(im, ax=ax, shrink=0.9)

	ax = axs[2]
	ax.pcolormesh(Xs, Ys, Vs, shading="auto", cmap="viridis", alpha=0.65)
	xu = np.linspace(Xs[0, 0], Xs[0, -1], min(160, Xs.shape[1] * 2))
	yu = np.linspace(Ys[0, 0], Ys[-1, 0], min(160, Ys.shape[0] * 2))

	def _bilinear(F, xs, ys, xq, yq):
		Fi = np.zeros((yq.size, xq.size), dtype=float)
		ix1 = np.clip(np.searchsorted(xs, xq, side="right") - 1, 0, xs.size - 2)
		tx = (xq - xs[ix1]) / np.maximum(xs[ix1 + 1] - xs[ix1], 1.0e-16)
		iy1 = np.clip(np.searchsorted(ys, yq, side="right") - 1, 0, ys.size - 2)
		ty = (yq - ys[iy1]) / np.maximum(ys[iy1 + 1] - ys[iy1], 1.0e-16)
		for jj, (j0, wy) in enumerate(zip(iy1, ty)):
			f0 = (1.0 - wy) * F[j0, :] + wy * F[j0 + 1, :]
			left = f0[ix1]
			right = f0[ix1 + 1]
			Fi[jj, :] = (1.0 - tx) * left + tx * right
		return Fi

	Exu = _bilinear(Exs, x[i0:i1], y[j0:j1], xu, yu)
	Eyu = _bilinear(Eys, x[i0:i1], y[j0:j1], xu, yu)
	ax.streamplot(xu, yu, Exu, Eyu, density=1.15, color="k", linewidth=0.9)
	ax.set_title("Local streamlines")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")

	if title:
		fig.suptitle(title, fontsize=12)
	if output_path:
		fig.savefig(output_path, dpi=250)
	return fig


def plot_radial_diagnostics(problem, V, phi_deg=0.0, num_samples=400, output_path=None, title=None):
	x = problem["x"]
	y = problem["y"]
	meta = _extract_meta(problem)
	cx = float(meta.get("cx", 0.5 * (x[0] + x[-1])))
	cy = float(meta.get("cy", 0.5 * (y[0] + y[-1])))
	Rin = float(meta.get("Rin", 0.0))
	Rout = float(meta.get("Rout", 0.0))
	V0 = float(meta.get("voltage", 0.0))
	Ex, Ey, Em = compute_field(problem, V)
	phi = np.deg2rad(phi_deg)

	def _interp(M, px, py):
		xs = x
		ys = y
		i1 = int(np.clip(np.searchsorted(xs, px, side="right") - 1, 0, xs.size - 2))
		j1 = int(np.clip(np.searchsorted(ys, py, side="right") - 1, 0, ys.size - 2))
		x0 = xs[i1]
		x1 = xs[i1 + 1]
		y0 = ys[j1]
		y1 = ys[j1 + 1]
		if x1 == x0:
			tx = 0.0
		else:
			tx = (px - x0) / (x1 - x0)
		if y1 == y0:
			ty = 0.0
		else:
			ty = (py - y0) / (y1 - y0)
		f00 = M[j1, i1]
		f10 = M[j1, i1 + 1]
		f01 = M[j1 + 1, i1]
		f11 = M[j1 + 1, i1 + 1]
		return float((1.0 - ty) * ((1.0 - tx) * f00 + tx * f10) + ty * ((1.0 - tx) * f01 + tx * f11))

	r = np.linspace(max(Rin * 1.001, Rin + 1.0e-6), Rout * 0.999, num_samples)
	xs = cx + r * np.cos(phi)
	ys = cy + r * np.sin(phi)
	Es = np.array([_interp(Em, px, py) for px, py in zip(xs, ys)], dtype=float)
	if Rin > 0.0 and Rout > Rin:
		E0 = V0 / (r * np.log(Rout / Rin))
	else:
		E0 = np.zeros_like(r)

	fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.6), constrained_layout=True)
	ax.plot(r, Es, label="Simulated |E|(r) along ray", lw=1.6)
	ax.plot(r, E0, label="Ideal coax (no defect)", lw=1.2, linestyle="--")
	ax.set_xlabel("Radius r from cable center (m)")
	ax.set_ylabel("|E| (V/m)")
	ax.grid(True, alpha=0.3, linestyle="--")
	ax.legend()
	if title:
		ax.set_title(title)
	if output_path:
		fig.savefig(output_path, dpi=250)
	return fig













