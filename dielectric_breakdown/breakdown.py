from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np

from .grid import Grid2D
from .field import compute_field, trace_field_line


def townsend_alpha_air(E: np.ndarray, p: float = 101325.0) -> np.ndarray:
	"""
	First Townsend ionization coefficient α(E) for air using a common exponential fit:
	    α/p = A * exp(-B * p / E)
	in SI units. Constants chosen to match typical engineering practice.
	Defaults:
	  A ≈ 112.5 1/(Pa·m), B ≈ 2737 V/(Pa·m)
	These produce E_breakdown ~ 3 MV/m in uniform fields at 1 atm.
	"""
	A = 112.5  # 1/(Pa·m)
	B = 2737.0  # V/(Pa·m)
	with np.errstate(over="ignore", divide="ignore"):
		alpha = p * A * np.exp(-B * p / np.maximum(E, 1e-6))
	return alpha


def raether_meek_integral(
	grid: Grid2D,
	potential: np.ndarray,
	seeds: List[Tuple[float, float]],
	p: float = 101325.0,
	k_threshold: float = 20.0,
	max_length: float = 0.02,
	step_init: float = 1e-4,
	stop_region: np.ndarray | None = None,
) -> Dict[str, object]:
	"""
	Compute Raether–Meek ionization integral along field lines launched from `seeds`.
	Returns:
	  dict with keys:
	    'best_integral': float
	    'best_polyline': (m,2) array
	    'all_integrals': list of floats
	    'crossed_threshold': bool
	"""
	Ex, Ey, Emag = compute_field(grid, potential)
	all_integrals: List[float] = []
	best_I = -np.inf
	best_poly = None
	for s in seeds:
		line = trace_field_line(
			grid, Ex, Ey, start=s, max_length=max_length, step_init=step_init, direction=1, stop_region=stop_region
		)
		# Sample E along line by bilinear interpolation using grid lookup
		xs, ys = grid.x, grid.y
		i1 = np.clip(np.searchsorted(xs, line[:, 0], side="right") - 1, 0, len(xs) - 1)
		j1 = np.clip(np.searchsorted(ys, line[:, 1], side="right") - 1, 0, len(ys) - 1)
		Eline = Emag[j1, i1]
		alpha = townsend_alpha_air(Eline, p=p)
		# line arclengths
		ds = np.hypot(np.diff(line[:, 0]), np.diff(line[:, 1]))
		I = float(np.sum(0.5 * (alpha[:-1] + alpha[1:]) * ds))
		all_integrals.append(I)
		if I > best_I:
			best_I = I
			best_poly = line
	return {
		"best_integral": float(best_I),
		"best_polyline": np.array(best_poly) if best_poly is not None else None,
		"all_integrals": all_integrals,
		"crossed_threshold": bool(best_I >= k_threshold),
		"k_threshold": float(k_threshold),
	}




