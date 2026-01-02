import numpy as np

DOMAIN_PADDING = 1.06


def make_grid(nx, ny, outer_radius):
	Rout = float(outer_radius)
	L = 2.0 * DOMAIN_PADDING * Rout
	x = np.linspace(0.0, L, nx)
	y = np.linspace(0.0, L, ny)
	X, Y = np.meshgrid(x, y, indexing="xy")
	return {
		"x": x,
		"y": y,
		"X": X,
		"Y": Y,
		"nx": int(nx),
		"ny": int(ny),
		"L": L,
	}


def build_problem(
	voltage,
	inner_radius,
	outer_radius,
	eps_r,
	defect_type="none",
	defect_radius=0.0,
	defect_center_x=None,
	defect_center_y=None,
	defect_epsr=None,
	nx=400,
	ny=400,
):
	Rin = float(inner_radius)
	Rout = float(outer_radius)

	if not (Rout > Rin > 0.0):
		raise ValueError("outer_radius must be > inner_radius > 0")

	grid = make_grid(nx, ny, Rout)
	x = grid["x"]
	y = grid["y"]
	X = grid["X"]
	Y = grid["Y"]
	L = grid["L"]

	cx = 0.5 * L
	cy = 0.5 * L

	R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

	dirichlet_mask = np.zeros((ny, nx), dtype=bool)
	dirichlet_values = np.zeros((ny, nx), dtype=float)
	electrode_mask = np.zeros((ny, nx), dtype=bool)

	if nx > 1:
		dx = x[1] - x[0]
	else:
		dx = 1.0

	if ny > 1:
		dy = y[1] - y[0]
	else:
		dy = 1.0

	band = 0.6 * max(dx, dy)
	inner_region = R <= Rin + band
	outer_region = R >= Rout - band

	if not np.any(inner_region):
		j0, i0 = np.unravel_index(np.argmin((R - Rin) ** 2), R.shape)
		inner_region[j0, i0] = True

	if not np.any(outer_region):
		j1, i1 = np.unravel_index(np.argmin((R - Rout) ** 2), R.shape)
		outer_region[j1, i1] = True

	dirichlet_mask[inner_region] = True
	dirichlet_values[inner_region] = float(voltage)
	electrode_mask[inner_region] = True

	dirichlet_mask[outer_region] = True
	dirichlet_values[outer_region] = 0.0
	electrode_mask[outer_region] = True

	epsilon = np.ones((ny, nx), dtype=float)
	annulus = (R >= Rin) & (R <= Rout)
	epsilon[annulus] = float(eps_r)

	if defect_type is None:
		defect_type = "none"

	dt = str(defect_type).lower()

	if dt in ("bubble", "inclusion") and defect_radius > 0.0:
		if defect_center_x is None or defect_center_y is None:
			r_mid = 0.5 * (Rin + Rout)
			xd = cx + r_mid
			yd = cy
		else:
			xd = float(defect_center_x)
			yd = float(defect_center_y)

		rd = float(defect_radius)
		region_defect = (X - xd) ** 2 + (Y - yd) ** 2 <= rd ** 2
		region_defect = region_defect & annulus & (~electrode_mask)

		if defect_epsr is None:
			if dt == "bubble":
				eps_def = 1.0
			else:
				eps_def = 80.0
		else:
			eps_def = float(defect_epsr)

		epsilon[region_defect] = eps_def

		defect_cx = xd
		defect_cy = yd
		defect_type_code = 1 if dt == "bubble" else 2
		defect_eps_value = eps_def
		defect_radius_value = rd
	else:
		defect_cx = cx
		defect_cy = cy
		defect_type_code = 0
		defect_eps_value = 1.0
		defect_radius_value = float(defect_radius)

	meta = {
		"cx": float(cx),
		"cy": float(cy),
		"Rin": float(Rin),
		"Rout": float(Rout),
		"voltage": float(voltage),
		"eps_r": float(eps_r),
		"defect_type": float(defect_type_code),
		"defect_cx": float(defect_cx),
		"defect_cy": float(defect_cy),
		"defect_radius": float(defect_radius_value),
		"defect_epsr": float(defect_eps_value),
	}

	problem = {
		"x": x,
		"y": y,
		"X": X,
		"Y": Y,
		"nx": int(nx),
		"ny": int(ny),
		"dirichlet_mask": dirichlet_mask,
		"dirichlet_values": dirichlet_values,
		"electrode_mask": electrode_mask,
		"epsilon": epsilon,
		"meta": meta,
	}

	return problem


def solve_potential(problem, max_iter=20000, tol=1e-6, omega=1.6, progress=False, progress_every=250):
	"""
	Solve ∇·(ε∇V)=0 with Dirichlet electrodes and Neumann (no-flux) box boundary.

	Implementation notes:
	- Uses a red–black Gauss–Seidel SOR update (vectorized), which is much faster than
	  a pure Python double-loop for large grids.
	- Harmonic means of ε are used at faces to improve flux continuity at ε-jumps.
	"""
	ny = int(problem["ny"])
	nx = int(problem["nx"])
	dirichlet_mask = problem["dirichlet_mask"]
	dirichlet_values = problem["dirichlet_values"]
	eps = problem.get("epsilon")
	if eps is None:
		eps = np.ones((ny, nx), dtype=float)
	else:
		eps = np.asarray(eps, dtype=float)

	V = np.zeros((ny, nx), dtype=float)
	V[dirichlet_mask] = dirichlet_values[dirichlet_mask]

	# Reflection helper for Neumann boundary (zero normal derivative at box edges).
	def _neighbors(A):
		Aw = np.empty_like(A)
		Ae = np.empty_like(A)
		As = np.empty_like(A)
		An = np.empty_like(A)

		if nx > 1:
			Aw[:, 1:] = A[:, :-1]
			Aw[:, 0] = A[:, 1]
			Ae[:, :-1] = A[:, 1:]
			Ae[:, -1] = A[:, -2]
		else:
			Aw[:, 0] = A[:, 0]
			Ae[:, 0] = A[:, 0]

		if ny > 1:
			As[1:, :] = A[:-1, :]
			As[0, :] = A[1, :]
			An[:-1, :] = A[1:, :]
			An[-1, :] = A[-2, :]
		else:
			As[0, :] = A[0, :]
			An[0, :] = A[0, :]

		return Aw, Ae, As, An

	def _harm(a, b):
		return (2.0 * a * b) / (a + b + 1.0e-30)

	# Precompute face coefficients (ε does not change during iterations).
	ew, ee, es, en = _neighbors(eps)
	cW = _harm(eps, ew)
	cE = _harm(eps, ee)
	cS = _harm(eps, es)
	cN = _harm(eps, en)
	den = cW + cE + cS + cN
	den = np.maximum(den, 1.0e-30)

	jj, ii = np.indices((ny, nx))
	red = ((ii + jj) % 2) == 0
	black = ~red
	red_update = red & (~dirichlet_mask)
	black_update = black & (~dirichlet_mask)

	for it in range(max_iter):
		max_delta = 0.0

		# --- Red update (depends only on black neighbors on a 4-neighbour stencil) ---
		Vw, Ve, Vs, Vn = _neighbors(V)
		v_new = (cW * Vw + cE * Ve + cS * Vs + cN * Vn) / den
		old = V[red_update].copy()
		V[red_update] = (1.0 - omega) * V[red_update] + omega * v_new[red_update]
		V[dirichlet_mask] = dirichlet_values[dirichlet_mask]
		delta = np.max(np.abs(V[red_update] - old)) if old.size else 0.0
		max_delta = max(max_delta, float(delta))

		# --- Black update (depends only on updated red neighbors) ---
		Vw, Ve, Vs, Vn = _neighbors(V)
		v_new = (cW * Vw + cE * Ve + cS * Vs + cN * Vn) / den
		old = V[black_update].copy()
		V[black_update] = (1.0 - omega) * V[black_update] + omega * v_new[black_update]
		V[dirichlet_mask] = dirichlet_values[dirichlet_mask]
		delta = np.max(np.abs(V[black_update] - old)) if old.size else 0.0
		max_delta = max(max_delta, float(delta))

		if progress and (it % int(progress_every) == 0 or it == max_iter - 1):
			print(f"iter {it:6d}  max_delta={max_delta:.3e}")

		if max_delta < tol:
			break

	return V


def compute_field(problem, V):
	x = problem["x"]
	y = problem["y"]
	ny, nx = V.shape
	Ex = np.zeros_like(V)
	Ey = np.zeros_like(V)
	if x.size > 1:
		dx = x[1] - x[0]
	else:
		dx = 1.0
	if y.size > 1:
		dy = y[1] - y[0]
	else:
		dy = 1.0
	for j in range(ny):
		for i in range(nx):
			if i == 0:
				if nx > 1:
					Ex[j, i] = -(V[j, i + 1] - V[j, i]) / dx
				else:
					Ex[j, i] = 0.0
			elif i == nx - 1:
				Ex[j, i] = -(V[j, i] - V[j, i - 1]) / dx
			else:
				Ex[j, i] = -(V[j, i + 1] - V[j, i - 1]) / (2.0 * dx)
			if j == 0:
				if ny > 1:
					Ey[j, i] = -(V[j + 1, i] - V[j, i]) / dy
				else:
					Ey[j, i] = 0.0
			elif j == ny - 1:
				Ey[j, i] = -(V[j, i] - V[j - 1, i]) / dy
			else:
				Ey[j, i] = -(V[j + 1, i] - V[j - 1, i]) / (2.0 * dy)
	Emag = np.sqrt(Ex ** 2 + Ey ** 2)
	return Ex, Ey, Emag


def find_peak_field(Emag, mask_exclude=None):
	E = Emag.copy()
	if mask_exclude is not None:
		E[mask_exclude] = -1.0e30
	index = int(np.argmax(E))
	j, i = np.unravel_index(index, E.shape)
	return float(E[j, i]), (int(j), int(i))
