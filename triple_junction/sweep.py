from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Iterable, List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

from dielectric_breakdown.geometry import TripleJunctionGeometry
from dielectric_breakdown.solver import solve_potential
from dielectric_breakdown.field import compute_field
from .run import _compute_metrics


@dataclass(frozen=True)
class SweepConfig:
	width: float = 0.02
	height: float = 0.02
	epsr_list: Iterable[float] = (2.0, 4.0, 8.0, 16.0)
	td_list: Iterable[float] = (0.2e-3, 0.5e-3, 1.0e-3, 2.0e-3)
	voltage: float = 10e3
	pad_width: float = 3e-3
	pad_height: float = 0.6e-3
	nx: int = 520
	ny: int = 380
	method: str = "cg"
	outdir: str = "triple_junction/outputs/sweeps"


def run_sweep(cfg: SweepConfig) -> str:
	os.makedirs(cfg.outdir, exist_ok=True)
	rows: List[Dict[str, float]] = []
	for epsr in tqdm(cfg.epsr_list, desc="epsr", leave=False):
		for td in tqdm(cfg.td_list, desc="td", leave=False):
			geom = TripleJunctionGeometry(
				width=cfg.width,
				height=cfg.height,
				dielectric_thickness=float(td),
				eps_r=float(epsr),
				voltage=cfg.voltage,
				pad_width=cfg.pad_width,
				pad_height=cfg.pad_height,
				nx=cfg.nx,
				ny=cfg.ny,
			)
			problem = geom.build()
			V = solve_potential(problem, method=cfg.method)
			metrics = _compute_metrics(problem, V)
			row = {
				"epsr": float(epsr),
				"td_m": float(td),
				"Emax_global": metrics["Emax_global"],
				"Emax_air_near_junction": metrics["Emax_air_near_junction"],
				"Dn_ratio_die_over_air": metrics["Dn_ratio_die_over_air"],
			}
			rows.append(row)
	df = pd.DataFrame(rows)
	csv_path = os.path.join(cfg.outdir, "sweep_results.csv")
	df.to_csv(csv_path, index=False)
	return csv_path


if __name__ == "__main__":
	path = run_sweep(SweepConfig())
	print(f"Wrote sweep: {path}")



