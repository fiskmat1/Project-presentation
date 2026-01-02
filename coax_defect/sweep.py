import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from .geometry import build_problem, solve_potential, compute_field


def run_sweep():
    Rin = 2.0e-3
    Rout = 10.0e-3
    V0 = 15000.0
    epsr = 2.3
    defect_type = "bubble"
    defect_epsr = 1.0
    defect_radii = [0.2e-3, 0.5e-3, 1.0e-3]
    phi_deg_list = [0.0, 45.0, 90.0, 135.0]
    radial_frac_list = [0.4, 0.6, 0.8]
    nx = 560
    ny = 560
    outdir = "coax_defect/outputs/sweeps"
    padding = 1.06

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    rows = []

    for rd in tqdm(defect_radii, desc="defect radius"):
        for phi_deg in tqdm(phi_deg_list, desc="angle deg"):
            for frac in tqdm(radial_frac_list, desc="radial frac"):
                r = Rin + float(frac) * (Rout - Rin)
                phi = np.deg2rad(float(phi_deg))

                L = 2.0 * padding * Rout
                cx = 0.5 * L
                cy = 0.5 * L

                xd = cx + r * np.cos(phi)
                yd = cy + r * np.sin(phi)

                if defect_type == "bubble":
                    eps_def = None
                else:
                    eps_def = defect_epsr

                problem = build_problem(
                    voltage=V0,
                    inner_radius=Rin,
                    outer_radius=Rout,
                    eps_r=epsr,
                    defect_type=defect_type,
                    defect_radius=rd,
                    defect_center_x=xd,
                    defect_center_y=yd,
                    defect_epsr=eps_def,
                    nx=nx,
                    ny=ny,
                )

                V = solve_potential(
                    problem,
                    max_iter=15000,
                    tol=1.0e-5,
                    omega=1.6,
                )
                Ex, Ey, Em = compute_field(problem, V)

                Emax = float(np.nanmax(Em))
                Eideal = V0 / (Rin * np.log(Rout / Rin))

                if defect_type == "bubble" and eps_def is None:
                    stored_eps = 1.0
                else:
                    stored_eps = float(defect_epsr)

                rows.append(
                    {
                        "defect_type": defect_type,
                        "defect_epsr": stored_eps,
                        "defect_radius_m": float(rd),
                        "phi_deg": float(phi_deg),
                        "radial_frac": float(frac),
                        "Emax": Emax,
                        "Eideal_Rin": float(Eideal),
                        "enhancement": Emax / (Eideal + 1.0e-25),
                    }
                )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


if __name__ == "__main__":
    path = run_sweep()
    print("Saved sweep CSV to:", path)
