from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from .grid import Grid2D, make_grid


@dataclass(frozen=True)
class GeometryProblem:
	grid: Grid2D
	dirichlet_mask: np.ndarray       # shape (ny, nx), True where V is fixed
	dirichlet_values: np.ndarray     # shape (ny, nx)
	neumann_sides: Dict[str, bool]   # zero-flux at sides not covered by Dirichlet
	electrode_mask: np.ndarray       # True where metal electrode lies (subset of dirichlet_mask)
	meta: Dict[str, float]           # geometry params for later reference
	# Optional spatially varying permittivity ε(x,y) on node grid (ny, nx).
	# If None, solver assumes ε = 1 everywhere (homogeneous Laplace).
	epsilon: np.ndarray | None = None


@dataclass(frozen=True)
class ParallelPlatesGeometry:
	width: float           # domain width (m)
	gap: float             # plate separation (m)
	voltage: float         # upper plate potential (V), bottom is 0 V
	nx: int = 200
	ny: int = 160

	def build(self) -> GeometryProblem:
		grid = make_grid(
			lx=self.width,
			ly=self.gap,
			nx=self.nx,
			ny=self.ny,
			focus_x=[0.5 * self.width],
			focus_y=None,
			strength=3.0,
			axisymmetric=False,
		)
		nx, ny = grid.nx, grid.ny
		# Masks are (ny, nx)
		dirichlet_mask = np.zeros((ny, nx), dtype=bool)
		dirichlet_values = np.zeros((ny, nx), dtype=float)
		electrode_mask = np.zeros((ny, nx), dtype=bool)
		# Bottom plate at y=0 -> j=0
		dirichlet_mask[0, :] = True
		dirichlet_values[0, :] = 0.0
		electrode_mask[0, :] = True
		# Top plate at y=gap -> j=ny-1
		dirichlet_mask[-1, :] = True
		dirichlet_values[-1, :] = self.voltage
		electrode_mask[-1, :] = True
		neumann_sides = {"left": True, "right": True, "top": False, "bottom": False}
		return GeometryProblem(
			grid=grid,
			dirichlet_mask=dirichlet_mask,
			dirichlet_values=dirichlet_values,
			neumann_sides=neumann_sides,
			electrode_mask=electrode_mask,
			meta={
				"width": self.width,
				"gap": self.gap,
				"voltage": self.voltage,
			},
		)


@dataclass(frozen=True)
class NeedlePlaneGeometry:
	width: float               # computational domain width (m), should be several gaps
	height: float              # computational domain height (m)
	gap: float                 # tip-to-plane distance (m)
	needle_tip_radius: float   # hemispherical tip radius (m)
	voltage: float             # needle potential (V), plane is ground
	nx: int = 280
	ny: int = 240
	axisymmetric: bool = True  # treat x as r, y as z (needle axis at x=0)

	def build(self) -> GeometryProblem:
		# Focus around the axis (x=0) and around the tip height (y=self.gap)
		grid = make_grid(
			lx=self.width,
			ly=self.height,
			nx=self.nx,
			ny=self.ny,
			focus_x=[0.0],  # cluster near axis exactly at r=0
			focus_y=[self.gap],
			strength=5.0,
			axisymmetric=self.axisymmetric,
		)
		X = grid.X
		Y = grid.Y
		nx, ny = grid.nx, grid.ny
		dirichlet_mask = np.zeros((ny, nx), dtype=bool)
		dirichlet_values = np.zeros((ny, nx), dtype=float)
		electrode_mask = np.zeros((ny, nx), dtype=bool)
		# Grounded plane at y=0 (bottom boundary)
		dirichlet_mask[0, :] = True
		dirichlet_values[0, :] = 0.0
		electrode_mask[0, :] = True
		# Needle electrode: hemispherical tip centered at (r=0, z=self.gap + r_tip)
		r_tip = float(self.needle_tip_radius)
		r = X  # if axisymmetric, x is radius
		z = Y
		# Hemisphere: (z - (gap + r_tip))^2 + r^2 <= r_tip^2, and z >= gap
		center_z = self.gap + r_tip
		# Robust painting of the electrode onto nodes: include nodes within a small band of the hemispherical surface
		# Distance to surface (positive outside)
		dist = np.sqrt((z - center_z) ** 2 + r ** 2) - r_tip
		# Band width tied to grid resolution
		dx_min = float(np.min(np.diff(grid.x)))
		dy_min = float(np.min(np.diff(grid.y)))
		band = 0.6 * max(dx_min, dy_min)
		tip_region = (z >= self.gap) & (np.abs(dist) <= band)
		# Slender shank above the hemisphere (a small cylinder on the axis)
		shank_region = (r <= max(0.75 * r_tip, 0.5 * dx_min)) & (z > center_z)
		needle_region = tip_region | shank_region
		# Ensure at least one node is painted near the apex
		if not np.any(needle_region):
			jcand, icand = np.unravel_index(np.argmin((np.maximum(z - self.gap, 0.0)) ** 2 + r ** 2), z.shape)
			needle_region[jcand, icand] = True
		dirichlet_mask[needle_region] = True
		dirichlet_values[needle_region] = self.voltage
		electrode_mask[needle_region] = True
		# Sides and top are zero-flux to emulate open space
		neumann_sides = {"left": True, "right": True, "top": True, "bottom": False}
		return GeometryProblem(
			grid=grid,
			dirichlet_mask=dirichlet_mask,
			dirichlet_values=dirichlet_values,
			neumann_sides=neumann_sides,
			electrode_mask=electrode_mask,
			meta={
				"width": self.width,
				"height": self.height,
				"gap": self.gap,
				"needle_tip_radius": self.needle_tip_radius,
				"voltage": self.voltage,
			},
		)


@dataclass(frozen=True)
class TripleJunctionGeometry:
	"""
	Metal–dielectric–air triple-junction geometry in 2D Cartesian cross-section.
	- Bottom boundary (y=0): grounded conductor.
	- Dielectric layer from y=0..t_d with relative permittivity eps_r.
	- Air above from y=t_d..height (eps_r=1).
	- High-voltage metal pad sits on top of dielectric, touching at y=t_d within
	  a finite width region centered at x=pad_center_x (default width/2).
	"""
	width: float
	height: float
	dielectric_thickness: float
	eps_r: float
	voltage: float
	pad_width: float
	pad_height: float
	pad_center_x: float | None = None
	nx: int = 420
	ny: int = 320

	def build(self) -> GeometryProblem:
		td = float(self.dielectric_thickness)
		xc = (self.width * 0.5) if self.pad_center_x is None else float(self.pad_center_x)
		xl = xc - 0.5 * self.pad_width
		xr = xc + 0.5 * self.pad_width
		# Focus near material interface and pad edges
		grid = make_grid(
			lx=self.width,
			ly=self.height,
			nx=self.nx,
			ny=self.ny,
			focus_x=[max(min(xl, self.width), 0.0), max(min(xr, self.width), 0.0)],
			focus_y=[td],
			strength=6.0,
			axisymmetric=False,
		)
		nx, ny = grid.nx, grid.ny
		X, Y = grid.X, grid.Y
		dirichlet_mask = np.zeros((ny, nx), dtype=bool)
		dirichlet_values = np.zeros((ny, nx), dtype=float)
		electrode_mask = np.zeros((ny, nx), dtype=bool)
		# Ground at bottom
		dirichlet_mask[0, :] = True
		dirichlet_values[0, :] = 0.0
		electrode_mask[0, :] = True

		# High-voltage pad: rectangle on top of dielectric
		pad_region = (Y >= td) & (Y <= td + self.pad_height) & (X >= xl) & (X <= xr)
		if not np.any(pad_region):
			# Guarantee at least one node lies on the pad for stability
			i = int(np.clip(np.searchsorted(grid.x, xc, side="left"), 0, nx - 1))
			j = int(np.clip(np.searchsorted(grid.y, td + 0.5 * self.pad_height, side="left"), 0, ny - 1))
			pad_region[j, i] = True
		dirichlet_mask[pad_region] = True
		dirichlet_values[pad_region] = self.voltage
		electrode_mask[pad_region] = True

		# Materials: ε map (ny, nx)
		epsilon = np.ones((ny, nx), dtype=float)
		epsilon[Y <= td] = float(self.eps_r)  # dielectric slab

		neumann_sides = {"left": True, "right": True, "top": True, "bottom": False}
		return GeometryProblem(
			grid=grid,
			dirichlet_mask=dirichlet_mask,
			dirichlet_values=dirichlet_values,
			neumann_sides=neumann_sides,
			electrode_mask=electrode_mask,
			meta={
				"width": self.width,
				"height": self.height,
				"dielectric_thickness": td,
				"eps_r": float(self.eps_r),
				"voltage": float(self.voltage),
				"pad_width": float(self.pad_width),
				"pad_height": float(self.pad_height),
				"pad_center_x": float(xc),
				"junction_x_left": float(xl),
				"junction_x_right": float(xr),
			},
			epsilon=epsilon,
		)


