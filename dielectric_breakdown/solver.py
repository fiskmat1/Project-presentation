from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Tuple

from .geometry import GeometryProblem
from .grid import Grid2D

# Optional SciPy sparse backend
_SCIPY_AVAILABLE = True
try:
	from scipy.sparse import lil_matrix, csc_matrix, diags
	from scipy.sparse.linalg import spsolve, cg, LinearOperator, spilu
except Exception:
	_SCIPY_AVAILABLE = False
	lil_matrix = csc_matrix = diags = spsolve = cg = LinearOperator = spilu = None


def _index(j: int, i: int, nx: int) -> int:
	return j * nx + i


def _spacings(axis: np.ndarray, i: int) -> Tuple[float, float]:
	if i <= 0 or i >= axis.size - 1:
		raise IndexError("Spacings requested at boundary index")
	return axis[i] - axis[i - 1], axis[i + 1] - axis[i]


def _assemble(problem: GeometryProblem) -> Tuple[csc_matrix, np.ndarray]:
	if not _SCIPY_AVAILABLE:
		raise RuntimeError("SciPy backend is not available; cannot assemble sparse system.")
	grid: Grid2D = problem.grid
	x, y = grid.x, grid.y
	nx, ny = grid.nx, grid.ny
	N = nx * ny
	dirichlet_mask = problem.dirichlet_mask
	dirichlet_values = problem.dirichlet_values
	neumann = problem.neumann_sides
	# Optional spatially varying permittivity map on nodes
	eps_map = getattr(problem, "epsilon", None)

	A = lil_matrix((N, N), dtype=float)
	b = np.zeros(N, dtype=float)

	def is_neumann_side(i: int, j: int) -> Tuple[bool, bool, bool, bool]:
		return (
			neumann.get("left", False) and i == 0,
			neumann.get("right", False) and i == nx - 1,
			neumann.get("bottom", False) and j == 0,
			neumann.get("top", False) and j == ny - 1,
		)
	# Harmonic average at faces to respect flux continuity across material jumps
	def _harmonic(a: float, b: float) -> float:
		da = max(a, 1e-16)
		db = max(b, 1e-16)
		return 2.0 * da * db / (da + db)

	for j in range(ny):
		for i in range(nx):
			row = _index(j, i, nx)
			if dirichlet_mask[j, i]:
				A[row, row] = 1.0
				b[row] = dirichlet_values[j, i]
				continue

			left_N, right_N, bottom_N, top_N = is_neumann_side(i, j)
			# Handle boundaries with Neumann if not Dirichlet
			if i == 0 and not left_N:
				# Physical boundary without Neumann means interior equation cannot be formed -> treat as natural Dirichlet 0 V
				A[row, row] = 1.0
				b[row] = 0.0
				continue
			if i == nx - 1 and not right_N:
				A[row, row] = 1.0
				b[row] = 0.0
				continue
			if j == 0 and not bottom_N:
				A[row, row] = 1.0
				b[row] = 0.0
				continue
			if j == ny - 1 and not top_N:
				A[row, row] = 1.0
				b[row] = 0.0
				continue

			# Build finite-volume Laplacian with possible axisymmetric r-weight in x-direction
			# We use cell-centered widths around node i: dxw, dxe; dyb, dyn
			# For Neumann sides, mirror ghost cells: value at ghost equals neighbor -> collapses to one-sided diff
			# Compute spacings; handle boundaries by equating dxw=dxe, etc.
			if i == 0:
				dxw = x[1] - x[0]
			else:
				dxw = x[i] - x[i - 1]
			if i == nx - 1:
				dxe = x[-1] - x[-2]
			else:
				dxe = x[i + 1] - x[i]
			if j == 0:
				dyb = y[1] - y[0]
			else:
				dyb = y[j] - y[j - 1]
			if j == ny - 1:
				dyn = y[-1] - y[-2]
			else:
				dyn = y[j + 1] - y[j]

			dx_c = 0.5 * (dxw + dxe)
			dy_c = 0.5 * (dyb + dyn)

			# Axisymmetric weighting: flux scales with r at faces
			if grid.axisymmetric:
				r_i = x[i]
				r_w = max(x[i] - 0.5 * dxw, 0.0)
				r_e = max(x[i] + 0.5 * dxe, 0.0)
				aw = r_w / max(dxw, 1e-16)
				ae = r_e / max(dxe, 1e-16)
				ax_scale = 1.0 / max(dx_c, 1e-16)
			else:
				aw = 1.0 / max(dxw, 1e-16)
				ae = 1.0 / max(dxe, 1e-16)
				ax_scale = 1.0 / max(dx_c, 1e-16)
			# y-direction (Cartesian both cases)
			as_ = 1.0 / max(dyb, 1e-16)
			an = 1.0 / max(dyn, 1e-16)
			ay_scale = 1.0 / max(dy_c, 1e-16)

			# Face permittivities
			if eps_map is None:
				eps_w = eps_e = eps_s = eps_n = 1.0
			else:
				e_c = float(eps_map[j, i])
				e_w = float(eps_map[j, i - 1]) if i > 0 else e_c
				e_e = float(eps_map[j, i + 1]) if i < nx - 1 else e_c
				e_s = float(eps_map[j - 1, i]) if j > 0 else e_c
				e_n = float(eps_map[j + 1, i]) if j < ny - 1 else e_c
				eps_w = _harmonic(e_c, e_w)
				eps_e = _harmonic(e_c, e_e)
				eps_s = _harmonic(e_c, e_s)
				eps_n = _harmonic(e_c, e_n)

			# Mirror for Neumann
			w_coeff = aw if i > 0 else (0.0 if not left_N else aw)
			e_coeff = ae if i < nx - 1 else (0.0 if not right_N else ae)
			s_coeff = as_ if j > 0 else (0.0 if not bottom_N else as_)
			n_coeff = an if j < ny - 1 else (0.0 if not top_N else an)

			# If at Neumann boundary, the missing neighbor contributes like an equal neighbor (mirror),
			# effectively doubling the opposite flux. We reflect this by zeroing the missing neighbor but
			# doubling the present side in the central balance below.
			neumann_w = (i == 0 and left_N)
			neumann_e = (i == nx - 1 and right_N)
			neumann_s = (j == 0 and bottom_N)
			neumann_n = (j == ny - 1 and top_N)

			# Coefficients for neighbors
			cW = eps_w * w_coeff * ax_scale
			cE = eps_e * e_coeff * ax_scale
			cS = eps_s * s_coeff * ay_scale
			cN = eps_n * n_coeff * ay_scale

			# Central coefficient balances neighbor coefficients; double the opposite if mirroring
			center = 0.0
			if i > 0:
				A[row, _index(j, i - 1, nx)] = -cW
				center += cW
			elif neumann_w:
				# mirror -> add to central as if ghost equals east neighbor; handled by doubling cE in center
				center += cW
			if i < nx - 1:
				A[row, _index(j, i + 1, nx)] = -cE
				center += cE
			elif neumann_e:
				center += cE
			if j > 0:
				A[row, _index(j - 1, i, nx)] = -cS
				center += cS
			elif neumann_s:
				center += cS
			if j < ny - 1:
				A[row, _index(j + 1, i, nx)] = -cN
				center += cN
			elif neumann_n:
				center += cN

			A[row, row] = center
			b[row] = 0.0

	# Impose Dirichlet nodes again to ensure exactness (in case of overlaps)
	for j in range(ny):
		for i in range(nx):
			if dirichlet_mask[j, i]:
				row = _index(j, i, nx)
				A.rows[row] = [row]
				A.data[row] = [1.0]
				b[row] = dirichlet_values[j, i]

	return A.tocsc(), b


def solve_potential(problem: GeometryProblem, method: str = "cg") -> np.ndarray:
	"""
	Solve Laplace's equation ∇·(k∇V)=0 with optional axisymmetric weighting encoded in the assembly.
	Returns potential V with shape (ny, nx).
	"""
	if _SCIPY_AVAILABLE and method in ("cg", "direct"):
		A, b = _assemble(problem)
		nx, ny = problem.grid.nx, problem.grid.ny
		if method == "direct":
			v = spsolve(A, b)
		else:
			# Preconditioned conjugate gradient
			# Add tiny diagonal regularization with correct shape
			Ap = A + diags([np.full(A.shape[0], 1e-12)], [0], shape=A.shape)
			try:
				M = spilu(Ap, drop_tol=1e-4, fill_factor=10.0)
				Mx = lambda x: M.solve(x)
				P = LinearOperator(Ap.shape, Mx)
				v, info = cg(Ap, b, M=P, maxiter=2000, atol=1e-10, rtol=1e-10)
			except Exception:
				v, info = cg(Ap, b, maxiter=4000, atol=1e-10, rtol=1e-10)
			if info != 0:
				# Fallback to direct if CG did not converge
				v = spsolve(A, b)
		return v.reshape((ny, nx))
	# Pure NumPy SOR fallback
	return _solve_potential_sor(problem)


def _solve_potential_sor(
	problem: GeometryProblem,
	max_iter: int = 20000,
	tol: float = 1e-6,
	omega: float = 1.8,
) -> np.ndarray:
	grid: Grid2D = problem.grid
	x, y = grid.x, grid.y
	nx, ny = grid.nx, grid.ny
	dirichlet_mask = problem.dirichlet_mask
	dirichlet_values = problem.dirichlet_values
	neumann = problem.neumann_sides
	eps_map = getattr(problem, "epsilon", None)
	V = np.zeros((ny, nx), dtype=float)
	# initialize at Dirichlet nodes
	V[dirichlet_mask] = dirichlet_values[dirichlet_mask]
	# Helpers for Neumann detection
	def is_neumann_side(i: int, j: int) -> Tuple[bool, bool, bool, bool]:
		return (
			neumann.get("left", False) and i == 0,
			neumann.get("right", False) and i == nx - 1,
			neumann.get("bottom", False) and j == 0,
			neumann.get("top", False) and j == ny - 1,
		)
	def _harmonic(a: float, b: float) -> float:
		da = max(a, 1e-16)
		db = max(b, 1e-16)
		return 2.0 * da * db / (da + db)
	for it in range(max_iter):
		max_delta = 0.0
		# Red-black SOR for better convergence on structured grids
		for parity in (0, 1):
			for j in range(ny):
				i_start = (parity - (j % 2)) % 2
				for i in range(i_start, nx, 2):
					if dirichlet_mask[j, i]:
						continue
					left_N, right_N, bottom_N, top_N = is_neumann_side(i, j)
					# local spacings
					dxw = (x[i] - x[i - 1]) if i > 0 else (x[1] - x[0])
					dxe = (x[i + 1] - x[i]) if i < nx - 1 else (x[-1] - x[-2])
					dyb = (y[j] - y[j - 1]) if j > 0 else (y[1] - y[0])
					dyn = (y[j + 1] - y[j]) if j < ny - 1 else (y[-1] - y[-2])
					dx_c = 0.5 * (dxw + dxe)
					dy_c = 0.5 * (dyb + dyn)
					if grid.axisymmetric:
						r_w = max(x[i] - 0.5 * dxw, 0.0)
						r_e = max(x[i] + 0.5 * dxe, 0.0)
						aw = r_w / max(dxw, 1e-16)
						ae = r_e / max(dxe, 1e-16)
						ax_scale = 1.0 / max(dx_c, 1e-16)
					else:
						aw = 1.0 / max(dxw, 1e-16)
						ae = 1.0 / max(dxe, 1e-16)
						ax_scale = 1.0 / max(dx_c, 1e-16)
					as_ = 1.0 / max(dyb, 1e-16)
					an = 1.0 / max(dyn, 1e-16)
					ay_scale = 1.0 / max(dy_c, 1e-16)
					# Face permittivities
					if eps_map is None:
						eps_w = eps_e = eps_s = eps_n = 1.0
					else:
						e_c = float(eps_map[j, i])
						e_w = float(eps_map[j, i - 1]) if i > 0 else e_c
						e_e = float(eps_map[j, i + 1]) if i < nx - 1 else e_c
						e_s = float(eps_map[j - 1, i]) if j > 0 else e_c
						e_n = float(eps_map[j + 1, i]) if j < ny - 1 else e_c
						eps_w = _harmonic(e_c, e_w)
						eps_e = _harmonic(e_c, e_e)
						eps_s = _harmonic(e_c, e_s)
						eps_n = _harmonic(e_c, e_n)
					cW = eps_w * aw * ax_scale
					cE = eps_e * ae * ax_scale
					cS = eps_s * as_ * ay_scale
					cN = eps_n * an * ay_scale
					center = 0.0
					num = 0.0
					# West
					if i > 0:
						center += cW
						num += cW * V[j, i - 1]
					elif left_N:
						center += cW
					# East
					if i < nx - 1:
						center += cE
						num += cE * V[j, i + 1]
					elif right_N:
						center += cE
					# South
					if j > 0:
						center += cS
						num += cS * V[j - 1, i]
					elif bottom_N:
						center += cS
					# North
					if j < ny - 1:
						center += cN
						num += cN * V[j + 1, i]
					elif top_N:
						center += cN
					if center <= 0:
						continue
					V_new = num / center
					old = V[j, i]
					V[j, i] = (1.0 - omega) * old + omega * V_new
					max_delta = max(max_delta, abs(V[j, i] - old))
		# re-impose Dirichlet strongly
		V[dirichlet_mask] = dirichlet_values[dirichlet_mask]
		if max_delta < tol:
			break
	return V


