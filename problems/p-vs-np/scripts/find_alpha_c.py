#!/usr/bin/env python3
"""
Binary search to find α_c(n) where P(SAT) ≈ 0.5 using MiniSat.

This script locates the critical clause density for random 3-SAT at a given n,
which is where the phase transition occurs (50% satisfiable).
"""
import subprocess
import random
import tempfile
import os
import sys
import time
import json
from pathlib import Path

# Constants
MINISAT_PATH = "/opt/homebrew/bin/minisat"
ALPHA_INFINITY = 4.267  # Theoretical asymptotic critical ratio
PHI = (1 + 5**0.5) / 2

def generate_random_3sat(n_vars, n_clauses, filepath):
    """Generate a random 3-SAT instance in DIMACS format."""
    with open(filepath, 'w') as f:
        f.write(f"p cnf {n_vars} {n_clauses}\n")
        for _ in range(n_clauses):
            # Pick 3 distinct variables
            vars_chosen = random.sample(range(1, n_vars + 1), 3)
            # Randomly negate each
            clause = [v if random.random() > 0.5 else -v for v in vars_chosen]
            f.write(" ".join(map(str, clause)) + " 0\n")

def run_minisat(cnf_path, timeout=300):
    """Run MiniSat on a CNF file. Returns True if SAT, False if UNSAT, None if timeout."""
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
    except Exception as e:
        print(f"Error running MiniSat: {e}")
        return None

def estimate_psat(n_vars, alpha, trials=15, timeout=300):
    """Estimate P(SAT) at given alpha using multiple trials."""
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

def binary_search_alpha_c(n_vars, target_psat=0.5, trials=15,
                          alpha_min=1.0, alpha_max=15.0,
                          precision=0.001, timeout=300, verbose=True):
    """
    Binary search to find α where P(SAT) ≈ target_psat.

    Returns: (alpha_c, final_psat)
    """
    if verbose:
        print(f"\nBinary search for n={n_vars}, target P(SAT)={target_psat}")
        print(f"  Search range: [{alpha_min:.3f}, {alpha_max:.3f}]")
        print("-" * 50)

    iteration = 0
    while alpha_max - alpha_min > precision:
        iteration += 1
        alpha_mid = (alpha_min + alpha_max) / 2

        psat = estimate_psat(n_vars, alpha_mid, trials, timeout)

        if psat is None:
            if verbose:
                print(f"  [{iteration}] α={alpha_mid:.4f}: timeout/error")
            # If we get timeouts, narrow from below
            alpha_min = alpha_mid
            continue

        if verbose:
            print(f"  [{iteration}] α={alpha_mid:.4f}: P(SAT)={psat:.3f}")

        if psat > target_psat:
            # Too easy (too many SAT), increase alpha
            alpha_min = alpha_mid
        else:
            # Too hard (too many UNSAT), decrease alpha
            alpha_max = alpha_mid

    alpha_c = (alpha_min + alpha_max) / 2
    final_psat = estimate_psat(n_vars, alpha_c, trials * 2, timeout)  # Double trials for final estimate

    return alpha_c, final_psat

def compute_shift(alpha_c):
    """Compute shift from asymptotic critical ratio."""
    return ALPHA_INFINITY - alpha_c

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Find α_c(n) via binary search')
    parser.add_argument('--n', type=int, default=500, help='Number of variables')
    parser.add_argument('--trials', type=int, default=15, help='Trials per alpha estimate')
    parser.add_argument('--precision', type=float, default=0.001, help='Search precision')
    parser.add_argument('--timeout', type=int, default=300, help='MiniSat timeout (seconds)')
    parser.add_argument('--alpha-min', type=float, default=1.0, help='Min search alpha')
    parser.add_argument('--alpha-max', type=float, default=15.0, help='Max search alpha')
    parser.add_argument('--output', type=str, help='Output JSON file')

    args = parser.parse_args()

    print("=" * 70)
    print(f"FINDING α_c FOR n = {args.n}")
    print("=" * 70)

    start_time = time.time()

    alpha_c, final_psat = binary_search_alpha_c(
        n_vars=args.n,
        trials=args.trials,
        precision=args.precision,
        timeout=args.timeout,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        verbose=True
    )

    elapsed = time.time() - start_time
    shift = compute_shift(alpha_c)
    radius = abs(shift)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  n            = {args.n}")
    print(f"  α_c(n)       = {alpha_c:.4f}")
    print(f"  Final P(SAT) = {final_psat:.3f}" if final_psat else "  Final P(SAT) = N/A")
    print(f"  Shift        = {shift:+.4f}  (4.267 - α_c)")
    print(f"  Radius       = {radius:.4f}  (|shift|)")
    print(f"  Time         = {elapsed:.1f}s")
    print("=" * 70)

    # Save results
    result = {
        "n": args.n,
        "alpha_c": alpha_c,
        "final_psat": final_psat,
        "shift": shift,
        "radius": radius,
        "alpha_infinity": ALPHA_INFINITY,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")

    return result

if __name__ == "__main__":
    main()
