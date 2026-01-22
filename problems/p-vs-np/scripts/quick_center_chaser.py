import numpy as np
import subprocess
import tempfile
import os
from multiprocessing import Pool, cpu_count

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

def estimate_p_sat(n_vars, alpha, num_trials=15):
    args_list = [(i, n_vars, alpha) for i in range(num_trials)]
    with Pool(processes=cpu_count()) as p:
        results = p.map(_sat_worker, args_list)
    return sum(results) / num_trials

def binary_search_center(n_vars, low=3.0, high=4.5, trials=15):
    print(f"Binary search for n={n_vars}")
    for step in range(8):
        mid = (low + high) / 2
        p = estimate_p_sat(n_vars, mid, num_trials=trials)
        print(f"  α={mid:.4f} → P_sat={p:.3f}")
        if p > 0.5:
            high = mid
        else:
            low = mid
    center = (low + high) / 2
    return center

if __name__ == "__main__":
    ns = [2000, 4000, 8000]
    centers = []
    shifts = []

    print("Quick center chaser: measuring orbit radius growth")

    for n in ns:
        center = binary_search_center(n)
        centers.append(center)
        shift = ASYMPTOTIC_ALPHA_C - center
        shifts.append(shift)
        print(f"n={n:4d} | center ≈ {center:.4f} | shift = {shift:.4f}\n")

    print("\nSummary:")
    print("n     center     shift")
    for n, c, s in zip(ns, centers, shifts):
        print(f"{n:4d}  {c:.4f}   {s:.4f}")

    print("\nIf shifts keep increasing: orbit diameter growing — middle receding!")
