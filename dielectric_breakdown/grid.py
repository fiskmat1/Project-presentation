from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid2D:
	x: np.ndarray  # shape (nx,)
	y: np.ndarray  # shape (ny,)
	axisymmetric: bool = False  # treat x as radius r if True

	@property
	def nx(self) -> int:
		return int(self.x.size)

	@property
	def ny(self) -> int:
		return int(self.y.size)

	@property
	def X(self) -> np.ndarray:
		return np.meshgrid(self.x, self.y, indexing="xy")[0]

	@property
	def Y(self) -> np.ndarray:
		return np.meshgrid(self.x, self.y, indexing="xy")[1]


def stretched_axis(
	length: float,
	n: int,
	focus_points: list[float] | None = None,
	strength: float = 3.0,
) -> np.ndarray:
	"""
	Create a monotone axis [0, length] with optional clustering around focus points using
	a tanh-based mapping. strength ~3 is mild, 6 is strong.
	"""
	s = np.linspace(0.0, 1.0, n)
	if not focus_points:
		return length * s
	# Combine multiple focus mappings multiplicatively to avoid over-concentration
	u = s.copy()
	for fp in focus_points:
		c = fp / max(length, 1e-12)
		# map s toward c using a smooth tanh warp
		warp = 0.5 * (np.tanh(strength * (s - c)) / np.tanh(strength * max(c, 1 - c)) + 1.0)
		u = 0.5 * (u + warp)
	# Renormalize to exactly [0, 1] so endpoints are anchored at 0 and length
	u = (u - u[0]) / max(u[-1] - u[0], 1e-16)
	u[0] = 0.0
	u[-1] = 1.0
	return length * u


def make_grid(
	lx: float,
	ly: float,
	nx: int,
	ny: int,
	focus_x: list[float] | None = None,
	focus_y: list[float] | None = None,
	strength: float = 3.0,
	axisymmetric: bool = False,
) -> Grid2D:
	x = stretched_axis(lx, nx, focus_x, strength=strength)
	y = stretched_axis(ly, ny, focus_y, strength=strength)
	return Grid2D(x=x, y=y, axisymmetric=axisymmetric)


