import numpy as np
import subprocess
import tempfile
import os
from multiprocessing import Pool, cpu_count
import time

ASYMPTOTIC_ALPHA_C = 4.267

def generate_3sat(n_vars, alpha, seed=None):
    if seed is not None:
        np.random.seed(seed)
    m = int(alpha * n_vars)
    clauses = []
    for _ in range(m):
        vars_chosen = np.random.choice(range(1, n_vars + 1), 3, replace=False)
        signs = np.random.choice([-1, 1], 3)
        clause = (vars_chosen * signs).tolist()
        clauses.append(clause)
    return clauses

def write_cnf_file(n_vars, clauses, filename):
    with open(filename, 'w') as f:
        f.write(f"p cnf {n_vars} {len(clauses)}\n")
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")

def run_minisat(cnf_file, timeout_sec=90):
    try:
        result = subprocess.run(['minisat', cnf_file], capture_output=True, text=True, timeout=timeout_sec)
        return 'SATISFIABLE' in result.stdout
    except:
        return False

def _sat_worker(args):
    i, n_vars, alpha = args
    with tempfile.NamedTemporaryFile(suffix='.cnf', delete=False) as tmp:
        cnf_path = tmp.name
        clauses = generate_3sat(n_vars, alpha, seed=i)
        write_cnf_file(n_vars, clauses, cnf_path)
        sat = run_minisat(cnf_path)
        os.unlink(cnf_path)
        return sat

def estimate_p_sat(n_vars, alpha, num_trials=30, timeout_sec=90):
    args_list = [(i, n_vars, alpha) for i in range(num_trials)]
    with Pool(processes=cpu_count()) as p:
        results = p.map(_sat_worker, args_list)
    return sum(results) / num_trials

def binary_search_center(n_vars, low=3.5, high=4.5, trials=20, tol=0.05):
    """Binary search for α where P_sat ≈ 0.5"""
    centers = []
    for _ in range(5):  # 5 iterations → precision ~0.03
        mid = (low + high) / 2
        p = estimate_p_sat(n_vars, mid, num_trials=trials)
        print(f"  Binary step: α={mid:.3f}, P_sat={p:.3f}")
        if p > 0.5:
            high = mid
        else:
            low = mid
        centers.append(mid)
    return np.mean(centers[-3:])  # average last 3 for stability

def coarse_scan_width(n_vars, center, trials=20):
    """Quick 6-point scan around center to estimate rough width"""
    alphas = np.linspace(center - 0.3, center + 0.3, 6)
    p_sats = []
    for a in alphas:
        p = estimate_p_sat(n_vars, a, num_trials=trials)
        p_sats.append(p)
    p_sats = np.array(p_sats)
    # Rough width: distance between first >0.8 and first <0.2
    high_idx = np.where(p_sats > 0.8)[0]
    low_idx = np.where(p_sats < 0.2)[0]
    if len(high_idx) > 0 and len(low_idx) > 0:
        width = alphas[low_idx[0]] - alphas[high_idx[-1]]
    else:
        width = np.nan
    return width, alphas.mean()

if __name__ == "__main__":
    ns = [500, 1000, 2000, 4000]
    shifts = []
    diameters = []

    print("Efficient orbit hunt: measuring deviation / diameter growth")

    for n in ns:
        print(f"\n=== n = {n} ===")
        center = binary_search_center(n, trials=20)
        print(f"Estimated center α_c(n) ≈ {center:.4f}")
        shift = ASYMPTOTIC_ALPHA_C - center
        shifts.append(shift)
        print(f"Shift = {shift:.4f}")

        rough_width, avg_alpha = coarse_scan_width(n, center, trials=20)
        diameters.append(rough_width)
        print(f"Rough diameter (width) ≈ {rough_width:.4f}")

    # Summary table
    print("\n" + "="*60)
    print("n     | α_c(n)   | shift     | rough diameter")
    print("-"*60)
    for n, c, s, d in zip(ns, fitted_centers if 'fitted_centers' in locals() else [np.nan]*len(ns), shifts, diameters):
        print(f"{n:5d} | {c:.4f} | {s:.4f}   | {d:.4f}")

    # Fit shift growth (allow positive θ for growing diameter)
    if len(shifts) >= 3:
        log_n = np.log(ns)
        log_shift = np.log(np.abs(shifts))
        def power(logn, theta, logc):
            return logc + theta * logn   # note +theta so positive = growth
        popt, _ = curve_fit(power, log_n, log_shift, p0=[0.1, -2])
        theta_fit = popt[0]
        c_fit = np.exp(popt[1])
        print(f"\nFitted shift growth: shift ≈ {c_fit:.4f} * n^{theta_fit:.4f}")
        if theta_fit > 0:
            print("→ DIAMETER GROWING with n (supports diverging orbit)")
        else:
            print("→ Diameter shrinking (classical scaling)")

    print("\nNext: add n=6000–10000 if shift keeps growing.")
