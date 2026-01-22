import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import subprocess
import tempfile
import os
from multiprocessing import Pool, cpu_count
import time

# Constants
ASYMPTOTIC_ALPHA_C = 4.267
PHI = (1 + np.sqrt(5)) / 2
L4, L5 = 7, 11
DAT_CONSTANTS = {
    '1/φ': 1/PHI,
    '7/12': 7/12,
    '1/(2φ)': 1/(2*PHI),
    '11/18': 11/18,
    'L(5)/(L(6)+1)': 11/(18+1),
}

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
        result = subprocess.run(
            ['minisat', cnf_file],
            capture_output=True,
            text=True,
            timeout=timeout_sec
        )
        return 'SATISFIABLE' in result.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        print(f"minisat error on {cnf_file}: {e}")
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

def estimate_p_sat(n_vars, alpha, num_trials=60, timeout_sec=90):
    print(f"  Estimating P_sat(n={n_vars}, α={alpha:.3f}) with {num_trials} trials...")
    start = time.time()
    args_list = [(i, n_vars, alpha) for i in range(num_trials)]
    with Pool(processes=cpu_count()) as p:
        results = p.map(_sat_worker, args_list)
    p_sat = sum(results) / num_trials
    elapsed = time.time() - start
    print(f"  → P_sat = {p_sat:.3f}  (took {elapsed:.1f} sec)")
    return p_sat

def fit_transition_width(n_vars, alphas_range, num_trials=60):
    p_sats = []
    for alpha in alphas_range:
        p = estimate_p_sat(n_vars, alpha, num_trials=num_trials)
        p_sats.append(p)
    p_sats = np.array(p_sats)

    def sigmoid(x, alpha_c, width, height=1.0, shift=0.0):
        return height / (1 + np.exp((x - alpha_c - shift) / width))

    try:
        popt, _ = curve_fit(
            sigmoid, alphas_range, p_sats,
            p0=[alphas_range.mean(), (alphas_range.max()-alphas_range.min())/4, 1.0, 0.0],
            bounds=([alphas_range.min()-0.5, 0.01, 0.5, -0.5],
                    [alphas_range.max()+0.5, 1.0, 1.5, 0.5]),
            maxfev=5000
        )
        delta_alpha = popt[1] * 2.2
        return delta_alpha, popt
    except Exception as e:
        print(f"Fit failed for n={n_vars}: {e}")
        return np.nan, None

if __name__ == "__main__":
    ns = [500, 1000, 2000, 4000]  # adjust as needed
    num_trials_per_point = 60

    deltas = []
    fitted_centers = []
    shifts = []

    for n in ns:
        print(f"\n=== n = {n} variables ===")

        theta = 0.62
        c_shift = 0.65
        expected_width = 1.4 / (n ** theta)
        alpha_center = ASYMPTOTIC_ALPHA_C - c_shift / (n ** theta)
        alpha_low = max(3.0, alpha_center - 5 * expected_width)
        alpha_high = alpha_center + 5 * expected_width
        alphas_range = np.linspace(alpha_low, alpha_high, 13)

        print(f"  Predicted center ≈ {alpha_center:.3f}, scanning {alpha_low:.3f} – {alpha_high:.3f}")

        delta, popt = fit_transition_width(n, alphas_range, num_trials_per_point)
        if not np.isnan(delta):
            deltas.append(delta)
            fitted_center = popt[0] if popt is not None else np.nan
            fitted_centers.append(fitted_center)
            shift = ASYMPTOTIC_ALPHA_C - fitted_center if not np.isnan(fitted_center) else np.nan
            shifts.append(shift)
            print(f"Δα ≈ {delta:.4f}   fitted center ≈ {fitted_center:.3f}   shift ≈ {shift:.4f}")
        else:
            deltas.append(np.nan)
            fitted_centers.append(np.nan)
            shifts.append(np.nan)

    # Fit shift law
    valid_mask = ~np.isnan(shifts)
    ns_valid = np.array(ns)[valid_mask]
    shifts_valid = np.array(shifts)[valid_mask]

    if len(ns_valid) >= 3:
        log_ns = np.log(ns_valid)
        log_shifts = np.log(shifts_valid)

        def shift_power(logn, theta, logC):
            return logC - theta * logn

        popt_shift, _ = curve_fit(shift_power, log_ns, log_shifts, p0=[0.62, np.mean(log_shifts)])

        theta_fit = popt_shift[0]
        c_fit = np.exp(popt_shift[1])

        print("\n" + "="*80)
        print(f"Fitted shift law: shift(n) ≈ {c_fit:.4f} / n^{theta_fit:.4f}")
        print("="*80)

        print("\nDAT constant matches for c:")
        for name, target in DAT_CONSTANTS.items():
            dev = abs(c_fit - float(target)) / float(target) * 100
            print(f"{name:<15}: {float(target):.4f} (dev {dev:.2f}%)")

        plt.figure(figsize=(10, 6))
        plt.loglog(ns_valid, shifts_valid, 'o-', label='Measured shift')
        plt.loglog(ns_valid, c_fit * ns_valid**(-theta_fit), 'r--', label=f'Fit c={c_fit:.3f}, θ={theta_fit:.3f}')
        plt.xlabel('n (variables)')
        plt.ylabel('Shift = 4.267 - α_c(n)')
        plt.title('Finite-Size Shift in 3-SAT Threshold')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.savefig('3sat_shift_plot.png')
        plt.show()

        print("Shift plot saved: 3sat_shift_plot.png")
    else:
        print("Not enough valid shift measurements to fit.")
