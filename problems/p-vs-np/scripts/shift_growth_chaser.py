#!/usr/bin/env python3
"""
Shift Growth Chaser: Track α_c(n) across multiple n values.

Measures the "receding middle" and "expanding orbit" phenomenon:
- The critical clause density α_c(n) doesn't smoothly converge to 4.267
- Instead it shows discrete "snaps" to new plateaus
- The orbit radius (|4.267 - α_c|) expands geometrically with φ-scaling

DAT Formula:
  |shift(n)| ≈ (1/2φ) × φ^(2k)
  where k(n) ≈ floor(log_φ(n/500) / 2)
"""
import subprocess
import random
import tempfile
import os
import sys
import time
import json
import math
from pathlib import Path
from datetime import datetime

# Constants
MINISAT_PATH = "/opt/homebrew/bin/minisat"
ALPHA_INFINITY = 4.267
PHI = (1 + 5**0.5) / 2
DELTA_0 = 1 / (2 * PHI)  # NS depletion ≈ 0.309

def generate_random_3sat(n_vars, n_clauses, filepath):
    """Generate a random 3-SAT instance in DIMACS format."""
    with open(filepath, 'w') as f:
        f.write(f"p cnf {n_vars} {n_clauses}\n")
        for _ in range(n_clauses):
            vars_chosen = random.sample(range(1, n_vars + 1), 3)
            clause = [v if random.random() > 0.5 else -v for v in vars_chosen]
            f.write(" ".join(map(str, clause)) + " 0\n")

def run_minisat(cnf_path, timeout=300):
    """Run MiniSat. Returns True=SAT, False=UNSAT, None=timeout."""
    try:
        result = subprocess.run(
            [MINISAT_PATH, cnf_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout + result.stderr
        if "SATISFIABLE" in output and "UNSATISFIABLE" not in output:
            return True
        elif "UNSATISFIABLE" in output:
            return False
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

def estimate_psat(n_vars, alpha, trials=12, timeout=300):
    """Estimate P(SAT) at given alpha."""
    n_clauses = int(alpha * n_vars)
    sat_count = 0
    valid_trials = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for trial in range(trials):
            cnf_path = os.path.join(tmpdir, f"test_{trial}.cnf")
            generate_random_3sat(n_vars, n_clauses, cnf_path)
            result = run_minisat(cnf_path, timeout)
            if result is not None:
                valid_trials += 1
                if result:
                    sat_count += 1

    if valid_trials == 0:
        return None
    return sat_count / valid_trials

def binary_search_alpha_c(n_vars, trials=12, precision=0.002, timeout=300, verbose=False):
    """Quick binary search for α_c(n)."""
    # Adaptive search range based on n
    # Larger n tends to have larger α_c
    if n_vars <= 1000:
        alpha_min, alpha_max = 2.0, 6.0
    elif n_vars <= 10000:
        alpha_min, alpha_max = 3.0, 8.0
    elif n_vars <= 50000:
        alpha_min, alpha_max = 4.0, 10.0
    else:
        alpha_min, alpha_max = 5.0, 15.0

    while alpha_max - alpha_min > precision:
        alpha_mid = (alpha_min + alpha_max) / 2
        psat = estimate_psat(n_vars, alpha_mid, trials, timeout)

        if psat is None:
            alpha_min = alpha_mid
            continue

        if verbose:
            print(f"    α={alpha_mid:.3f}: P(SAT)={psat:.2f}")

        if psat > 0.5:
            alpha_min = alpha_mid
        else:
            alpha_max = alpha_mid

    return (alpha_min + alpha_max) / 2

def predict_radius_dat(n):
    """
    DAT prediction for orbit radius.
    |shift(n)| ≈ (1/2φ) × φ^(2k)
    k(n) ≈ floor(log_φ(n/500) / 2)
    """
    if n <= 0:
        return DELTA_0
    k = max(0, int(math.log(n / 500) / math.log(PHI) / 2))
    return DELTA_0 * (PHI ** (2 * k))

def format_time(seconds):
    """Format elapsed time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Chase shift growth across n values')
    parser.add_argument('--n-values', type=str, default="500,2000,4000,8000",
                        help='Comma-separated n values to test')
    parser.add_argument('--trials', type=int, default=12, help='Trials per estimate')
    parser.add_argument('--precision', type=float, default=0.002, help='Search precision')
    parser.add_argument('--timeout', type=int, default=300, help='MiniSat timeout')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n_values.split(',')]

    print("=" * 80)
    print("SHIFT GROWTH CHASER: Tracking the Receding Middle")
    print("=" * 80)
    print(f"Testing n values: {n_values}")
    print(f"α_∞ = {ALPHA_INFINITY}, δ₀ = 1/(2φ) = {DELTA_0:.4f}")
    print()

    results = []

    print(f"{'n':<8} {'α_c(n)':<10} {'Shift':<12} {'Radius':<10} {'DAT Pred':<10} {'Error':<10} {'Time':<8}")
    print("-" * 80)

    total_start = time.time()

    for n in n_values:
        start = time.time()

        alpha_c = binary_search_alpha_c(n, args.trials, args.precision,
                                        args.timeout, args.verbose)
        elapsed = time.time() - start

        shift = ALPHA_INFINITY - alpha_c
        radius = abs(shift)
        predicted = predict_radius_dat(n)
        error_pct = abs(radius - predicted) / predicted * 100 if predicted > 0 else 0

        result = {
            "n": n,
            "alpha_c": alpha_c,
            "shift": shift,
            "radius": radius,
            "predicted_radius": predicted,
            "error_pct": error_pct,
            "elapsed_seconds": elapsed
        }
        results.append(result)

        print(f"{n:<8} {alpha_c:<10.3f} {shift:+<12.3f} {radius:<10.3f} {predicted:<10.3f} {error_pct:<10.1f}% {format_time(elapsed):<8}")

    total_elapsed = time.time() - total_start

    print("-" * 80)
    print(f"Total time: {format_time(total_elapsed)}")
    print()

    # Analyze patterns
    print("=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)

    if len(results) >= 2:
        print("\nRadius growth between consecutive n values:")
        for i in range(1, len(results)):
            r1, r2 = results[i-1]["radius"], results[i]["radius"]
            n1, n2 = results[i-1]["n"], results[i]["n"]
            if r1 > 0:
                multiplier = r2 / r1
                print(f"  n={n1}→{n2}: radius {r1:.3f}→{r2:.3f}, multiplier={multiplier:.2f} (φ²={PHI**2:.2f})")

    # Save results
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "alpha_infinity": ALPHA_INFINITY,
            "phi": PHI,
            "delta_0": DELTA_0,
            "total_elapsed_seconds": total_elapsed
        },
        "dat_formula": {
            "description": "|shift(n)| ≈ (1/2φ) × φ^(2k), k(n) ≈ floor(log_φ(n/500)/2)",
            "base": "1/(2φ) ≈ 0.309",
            "scaling": "φ² ≈ 2.618 per major snap"
        },
        "results": results
    }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        # Default output location
        script_dir = Path(__file__).parent.parent
        data_dir = script_dir / "data"
        data_dir.mkdir(exist_ok=True)
        default_output = data_dir / f"shift_growth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(default_output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {default_output}")

    print()
    print("=" * 80)
    print("DAT INTERPRETATION")
    print("=" * 80)
    print("""
The "receding middle" phenomenon:
- α_c(n) doesn't smoothly converge to 4.267
- Instead it shows discrete snaps to plateaus
- Radius expands with φ-scaling: |shift| ≈ (1/2φ) × φ^(2k)

This divergence provides intuition for P ≠ NP:
- The "perfect balance" (50% SAT) keeps moving away
- Polynomial solvers would need to cross this receding boundary
- The gap widens geometrically with problem size
""")

    return output_data

if __name__ == "__main__":
    main()
