from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Tuple, Callable, List

from .grid import Grid2D


def _grad_1d(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
	"""
	Compute first derivative along axis for each row/column using second-order
	non-uniform central differences where possible and one-sided at boundaries.
	values is shape (..., n)
	"""
	v = values
	n = axis.size
	d = np.zeros_like(v)
	if n == 1:
		return d
	# interior
	for i in range(1, n - 1):
		d[..., i] = (v[..., i + 1] - v[..., i - 1]) / (axis[i + 1] - axis[i - 1])
	# boundaries: one-sided second order
	d[..., 0] = (v[..., 1] - v[..., 0]) / (axis[1] - axis[0])
	d[..., -1] = (v[..., -1] - v[..., -2]) / (axis[-1] - axis[-2])
	return d


def compute_field(grid: Grid2D, potential: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Returns (Ex, Ey, |E|) on the node grid.
	Ex = -∂V/∂x, Ey = -∂V/∂y
	"""
	V = potential
	# Derivatives along x: operate across i index
	dVdx = _grad_1d(V, grid.x)
	# Along y: operate across j index, so transpose indices for reuse
	dVdy = _grad_1d(V.transpose(1, 0), grid.y).transpose(1, 0)
	Ex = -dVdx
	Ey = -dVdy
	Emag = np.hypot(Ex, Ey)
	return Ex, Ey, Emag


def find_peak_field(Emag: np.ndarray, mask_exclude: np.ndarray | None = None) -> Tuple[float, Tuple[int, int]]:
	E = Emag.copy()
	if mask_exclude is not None:
		E[mask_exclude] = -np.inf
	idx = np.nanargmax(E)
	j, i = np.unravel_index(idx, E.shape)
	return float(E[j, i]), (int(j), int(i))


def _bilinear_interpolate(x: float, y: float, grid: Grid2D, Fx: np.ndarray, Fy: np.ndarray) -> Tuple[float, float]:
	# Find indices bracketing x and y
	xs, ys = grid.x, grid.y
	if x <= xs[0]:
		i0 = 0
		i1 = 1
	elif x >= xs[-1]:
		i0 = len(xs) - 2
		i1 = len(xs) - 1
	else:
		i1 = int(np.searchsorted(xs, x, side="right"))
		i0 = i1 - 1
	if y <= ys[0]:
		j0 = 0
		j1 = 1
	elif y >= ys[-1]:
		j0 = len(ys) - 2
		j1 = len(ys) - 1
	else:
		j1 = int(np.searchsorted(ys, y, side="right"))
		j0 = j1 - 1
	x0, x1 = xs[i0], xs[i1]
	y0, y1 = ys[j0], ys[j1]
	tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
	ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
	# Corners
	fx00, fy00 = Fx[j0, i0], Fy[j0, i0]
	fx10, fy10 = Fx[j0, i1], Fy[j0, i1]
	fx01, fy01 = Fx[j1, i0], Fy[j1, i0]
	fx11, fy11 = Fx[j1, i1], Fy[j1, i1]
	# Bilinear
	fx0 = (1 - tx) * fx00 + tx * fx10
	fx1 = (1 - tx) * fx01 + tx * fx11
	fx = (1 - ty) * fx0 + ty * fx1
	fy0 = (1 - tx) * fy00 + tx * fy10
	fy1 = (1 - tx) * fy01 + tx * fy11
	fy = (1 - ty) * fy0 + ty * fy1
	return fx, fy


def trace_field_line(
	grid: Grid2D,
	Ex: np.ndarray,
	Ey: np.ndarray,
	start: Tuple[float, float],
	max_length: float,
	step_init: float,
	direction: int = 1,
	stop_region: np.ndarray | None = None,
) -> np.ndarray:
	"""
	Trace a streamline from `start` by integrating dx/ds = Ex/|E|, dy/ds = Ey/|E|.
	- direction: +1 follows E, -1 goes against E.
	- stop_region: boolean mask (ny, nx) where integration should stop (e.g., electrode).
	Returns array of points shape (m, 2).
	"""
	xs, ys = grid.x, grid.y
	xmin, xmax = xs[0], xs[-1]
	ymin, ymax = ys[0], ys[-1]
	points = [np.array(start, dtype=float)]
	h = float(step_init)
	total = 0.0
	while total < max_length:
		x, y = points[-1]
		if not (xmin <= x <= xmax and ymin <= y <= ymax):
			break
		fx, fy = _bilinear_interpolate(x, y, grid, Ex, Ey)
		Em = np.hypot(fx, fy)
		if Em < 1e-12:
			break
		ux, uy = direction * fx / Em, direction * fy / Em
		# RK4 step in arclength s
		def vel(px: float, py: float) -> Tuple[float, float]:
			_fx, _fy = _bilinear_interpolate(px, py, grid, Ex, Ey)
			_E = (np.hypot(_fx, _fy) + 1e-16)
			return direction * _fx / _E, direction * _fy / _E
		k1x, k1y = vel(x, y)
		k2x, k2y = vel(x + 0.5 * h * k1x, y + 0.5 * h * k1y)
		k3x, k3y = vel(x + 0.5 * h * k2x, y + 0.5 * h * k2y)
		k4x, k4y = vel(x + h * k3x, y + h * k3y)
		dx = (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
		dy = (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
		xn, yn = x + dx, y + dy
		points.append(np.array([xn, yn]))
		total += float(np.hypot(dx, dy))
		if stop_region is not None:
			# Stop if next point lands inside stop region
			# Find nearest indices
			i = int(np.clip(np.searchsorted(xs, xn, side="right") - 1, 0, len(xs) - 1))
			j = int(np.clip(np.searchsorted(ys, yn, side="right") - 1, 0, len(ys) - 1))
			if stop_region[j, i]:
				break
	return np.vstack(points)




