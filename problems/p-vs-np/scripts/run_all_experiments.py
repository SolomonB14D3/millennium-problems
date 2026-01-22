#!/usr/bin/env python3
"""
Master script to run all P vs NP shift growth experiments.

This replicates the key findings:
1. Runs binary search for α_c(n) across multiple n values
2. Tracks the "receding middle" phenomenon
3. Verifies the DAT formula: |shift| ≈ (1/2φ) × φ^(2k)
4. Saves results to data folder
"""
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Experiment configurations
SMALL_N = [500, 2000, 4000, 8000]      # Quick experiments
LARGE_N = [12000, 24000, 32000, 64000]  # Longer experiments

# Constants
PHI = (1 + 5**0.5) / 2
DELTA_0 = 1 / (2 * PHI)
ALPHA_INFINITY = 4.267

def get_script_dir():
    return Path(__file__).parent

def get_data_dir():
    data_dir = get_script_dir().parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir

def run_shift_chaser(n_values, output_name, trials=12, timeout=300):
    """Run shift growth chaser for given n values."""
    script = get_script_dir() / "shift_growth_chaser.py"
    output = get_data_dir() / output_name

    n_str = ",".join(map(str, n_values))
    cmd = [
        sys.executable, str(script),
        "--n-values", n_str,
        "--trials", str(trials),
        "--timeout", str(timeout),
        "--output", str(output)
    ]

    print(f"\nRunning shift chaser for n={n_str}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 70)

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def run_formula_verification(input_file=None):
    """Run DAT formula verification."""
    script = get_script_dir() / "verify_dat_formula.py"

    cmd = [sys.executable, str(script)]
    if input_file:
        cmd.extend(["--input", str(input_file)])
    else:
        cmd.append("--use-reference")

    print("\nRunning DAT formula verification")
    print("-" * 70)

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def combine_results(small_file, large_file, output_file):
    """Combine small and large n results into a single file."""
    combined_results = []

    for filepath in [small_file, large_file]:
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                combined_results.extend(data.get("results", []))

    # Sort by n
    combined_results.sort(key=lambda x: x["n"])

    # Compute additional metrics
    for i, r in enumerate(combined_results):
        # Compute k value
        n = r["n"]
        k = max(0, int((n / 500) ** 0.5 / PHI)) if n > 500 else 0
        r["k"] = k
        r["dat_predicted"] = DELTA_0 * (PHI ** (2 * k))

        # Compute multiplier from previous
        if i > 0:
            prev_radius = combined_results[i-1]["radius"]
            if prev_radius > 0 and r["radius"] != prev_radius:
                r["multiplier"] = r["radius"] / prev_radius

    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "description": "Combined shift growth experimental results",
            "alpha_infinity": ALPHA_INFINITY,
            "phi": PHI,
            "delta_0": DELTA_0,
            "dat_formula": "|shift(n)| ≈ (1/2φ) × φ^(2k)"
        },
        "results": combined_results,
        "summary": {
            "n_range": [combined_results[0]["n"], combined_results[-1]["n"]] if combined_results else [],
            "total_points": len(combined_results)
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nCombined results saved to {output_file}")
    return output_data

def print_summary_table(results):
    """Print a summary table of results."""
    print("\n" + "=" * 90)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 90)

    print(f"\n{'n':<8} {'α_c(n)':<10} {'Shift':<12} {'Radius':<10} {'k':<4} {'DAT Pred':<10} {'Error':<10}")
    print("-" * 90)

    for r in results:
        error = abs(r["radius"] - r.get("dat_predicted", 0))
        error_pct = error / r.get("dat_predicted", 1) * 100 if r.get("dat_predicted", 0) > 0 else 0
        print(f"{r['n']:<8} {r['alpha_c']:<10.3f} {r['shift']:+<12.3f} {r['radius']:<10.3f} {r.get('k', '?'):<4} {r.get('dat_predicted', 0):<10.3f} {error_pct:<10.1f}%")

    print("-" * 90)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run all P vs NP experiments')
    parser.add_argument('--small-only', action='store_true', help='Only run small n experiments')
    parser.add_argument('--large-only', action='store_true', help='Only run large n experiments')
    parser.add_argument('--verify-only', action='store_true', help='Only run verification on reference data')
    parser.add_argument('--trials', type=int, default=12, help='Trials per estimate')
    parser.add_argument('--timeout', type=int, default=300, help='MiniSat timeout')

    args = parser.parse_args()

    print("=" * 90)
    print("P vs NP SHIFT GROWTH EXPERIMENTS")
    print("Replicating the 'Receding Middle' Finding")
    print("=" * 90)
    print(f"DAT Formula: |shift(n)| ≈ (1/2φ) × φ^(2k)")
    print(f"             where 1/(2φ) = {DELTA_0:.4f}, φ² = {PHI**2:.4f}")
    print()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_dir = get_data_dir()

    if args.verify_only:
        run_formula_verification()
        return

    small_file = data_dir / f"small_n_{timestamp}.json"
    large_file = data_dir / f"large_n_{timestamp}.json"
    combined_file = data_dir / f"combined_results_{timestamp}.json"

    # Run experiments
    if not args.large_only:
        print("\n" + "=" * 90)
        print("PHASE 1: Small n experiments (n = 500-8000)")
        print("=" * 90)
        run_shift_chaser(SMALL_N, small_file.name, args.trials, args.timeout)

    if not args.small_only:
        print("\n" + "=" * 90)
        print("PHASE 2: Large n experiments (n = 12000-64000)")
        print("Note: These may take significantly longer")
        print("=" * 90)
        run_shift_chaser(LARGE_N, large_file.name, args.trials, min(args.timeout, 600))

    # Combine results
    if small_file.exists() or large_file.exists():
        print("\n" + "=" * 90)
        print("PHASE 3: Combining results")
        print("=" * 90)
        combined = combine_results(small_file, large_file, combined_file)
        print_summary_table(combined.get("results", []))

        # Run verification on combined results
        print("\n" + "=" * 90)
        print("PHASE 4: DAT Formula Verification")
        print("=" * 90)
        run_formula_verification(combined_file)

    print("\n" + "=" * 90)
    print("EXPERIMENTS COMPLETE")
    print("=" * 90)
    print(f"Results saved in: {data_dir}")

if __name__ == "__main__":
    main()
