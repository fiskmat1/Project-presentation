import os
import json
import numpy as np
import argparse
from geometry import build_problem, solve_potential, compute_field, find_peak_field
from plots import plot_overview, plot_zoom_near_defect, plot_radial_diagnostics

def _m_to_mm_str(v):
    return f'{v * 1000.0:.3f}mm'

def case_folder(Rin, Rout, V0, epsr, defect_type, defect_radius, defect_epsr, outdir, name):
    Rin_mm = _m_to_mm_str(Rin)
    Rout_mm = _m_to_mm_str(Rout)
    VkV = f'{int(round(V0 / 1000.0))}kV'
    base = f'coax_epsr{epsr:.2f}_Rin{Rin_mm}_Rout{Rout_mm}_V{VkV}'
  
    if defect_type != 'none':
        rd_mm = _m_to_mm_str(defect_radius)
        if defect_type == 'bubble':
            base += f'_bubble_r{rd_mm}'
        else:
            de = 80.0 if defect_epsr is None else float(defect_epsr)
            base += f'_incl_eps{de:.1f}_r{rd_mm}'
    if name:
        base += '_' + str(name)
    return os.path.join(outdir, base)

def compute_metrics(problem, V, Em):
    meta = problem.get('meta', {})
    Emax, pos = find_peak_field(Em, None)
    j_max, i_max = pos
    x = problem['x']
    y = problem['y']
    metrics = {}
    metrics['Emax_global'] = float(Emax)
    metrics['x_at_Emax'] = float(x[i_max])
    metrics['y_at_Emax'] = float(y[j_max])
    Rin = float(meta.get('Rin', 0.0))
    Rout = float(meta.get('Rout', 1.0))
    V0 = float(meta.get('voltage', 0.0))
  
    if Rin > 0.0 and Rout > Rin:
        E_ideal_max = V0 / (Rin * np.log(Rout / Rin))
    else:
        E_ideal_max = 0.0
    metrics['E_ideal_Rin'] = float(E_ideal_max)
  
    if E_ideal_max != 0.0:
        metrics['enhancement_vs_ideal'] = float(Emax / (E_ideal_max + 1e-25))
    else:
        metrics['enhancement_vs_ideal'] = 0.0
   
    rd = float(meta.get('defect_radius', 0.0))
  
    if rd > 0.0:
        metrics['defect_radius_m'] = float(rd)
        metrics['defect_center_x'] = float(meta.get('defect_cx', 0.0))
        metrics['defect_center_y'] = float(meta.get('defect_cy', 0.0))
    return metrics

def run_case(Rin=0.002, Rout=0.01, V0=15000.0, epsr=2.3, defect_type='bubble', defect_radius=0.0005, defect_center_x=None, defect_center_y=None, defect_epsr=None, nx=720, ny=720, outdir='outputs', name=None, make_plots=True, max_iter=20000, tol=1e-06, omega=1.6):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
   
    out_case = case_folder(Rin, Rout, V0, epsr, defect_type, defect_radius, defect_epsr, outdir, name)
   
    if not os.path.exists(out_case):
        os.makedirs(out_case)
  
    problem = build_problem(voltage=V0, inner_radius=Rin, outer_radius=Rout, eps_r=epsr, defect_type=defect_type, defect_radius=defect_radius, defect_center_x=defect_center_x, defect_center_y=defect_center_y, defect_epsr=defect_epsr, nx=nx, ny=ny)
    V = solve_potential(problem, max_iter=max_iter, tol=tol, omega=omega)
    Ex, Ey, Emag = compute_field(problem, V)
    np.savetxt(os.path.join(out_case, 'potential.csv'), V, delimiter=',')
    np.savetxt(os.path.join(out_case, 'Emag.csv'), Emag, delimiter=',')
    metrics = compute_metrics(problem, V, Emag)
   
    with open(os.path.join(out_case, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    if make_plots:
        title = f'Coaxial cable V0={V0 / 1000.0:.0f} kV, Rin={Rin * 1000.0:.2f} mm, Rout={Rout * 1000.0:.2f} mm, epsr={epsr:.2f}, defect={defect_type}'
       
        plot_overview(problem, V, title=title, output_path=os.path.join(out_case, 'overview.png'))
        plot_zoom_near_defect(problem, V, output_path=os.path.join(out_case, 'zoom_defect.png'), title='Zoom near defect')
        plot_radial_diagnostics(problem, V, phi_deg=0.0, output_path=os.path.join(out_case, 'radial_profiles.png'), title='Radial |E|(r) vs ideal')
    return out_case

def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--Rin', type=float, default=0.002)
    p.add_argument('--Rout', type=float, default=0.01)
    p.add_argument('--V0', type=float, default=15000.0)
    p.add_argument('--epsr', type=float, default=2.3)
    p.add_argument('--defect-type', type=str, default='bubble', choices=['none', 'bubble', 'inclusion'])
    p.add_argument('--defect-radius', type=float, default=0.0005)
    p.add_argument('--defect-epsr', type=float, default=None)
    p.add_argument('--nx', type=int, default=720)
    p.add_argument('--ny', type=int, default=720)
    p.add_argument('--outdir', type=str, default='outputs')
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--max-iter', type=int, default=20000)
    p.add_argument('--tol', type=float, default=1e-06)
    return p.parse_args(argv)

if __name__ == '__main__':
   
    args = _parse_args()
    defect_type = str(args.defect_type)
    if defect_type == 'none':
        defect_radius = 0.0
        defect_epsr = None
    else:
        defect_radius = float(args.defect_radius)
        defect_epsr = float(args.defect_epsr) if (defect_type == 'inclusion' and args.defect_epsr is not None) else None
   
    path = run_case(
        Rin=float(args.Rin),
        Rout=float(args.Rout),
        V0=float(args.V0),
        epsr=float(args.epsr),
        defect_type=defect_type,
        defect_radius=defect_radius,
        defect_center_x=None,
        defect_center_y=None,
        defect_epsr=defect_epsr,
        nx=int(args.nx),
        ny=int(args.ny),
        outdir=str(args.outdir),
        name=None,
        make_plots=not args.no_plots,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
    )
 
