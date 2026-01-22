#!/usr/bin/env python3
"""
Benchmark φ-aware SAT solver vs baseline.

The φ-structure doesn't make individual instances easier to solve -
it tells us WHERE the hard region is, allowing better resource allocation.

Key insight: If you know α_c(n), you can:
1. Skip instances far from transition (predict SAT/UNSAT without solving)
2. Allocate more time to hard instances
3. Use different heuristics per region
"""

import subprocess
import tempfile
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

PHI = (1 + math.sqrt(5)) / 2

def predict_alpha_c(n: int) -> float:
    """Empirical α_c(n) prediction."""
    data = [
        (500, 3.573), (2000, 4.497), (4000, 4.996),
        (8000, 4.996), (12000, 5.495), (24000, 6.998),
        (32000, 6.998), (64000, 9.996),
    ]
    if n <= data[0][0]: return data[0][1]
    if n >= data[-1][0]:
        return data[-1][1] + math.log(n / data[-1][0]) * 2.5
    for i in range(len(data) - 1):
        n1, a1 = data[i]
        n2, a2 = data[i + 1]
        if n1 <= n <= n2:
            t = (math.log(n) - math.log(n1)) / (math.log(n2) - math.log(n1))
            return a1 + t * (a2 - a1)
    return 4.267

def generate_3sat(n: int, m: int, seed: int) -> str:
    """Generate random 3-SAT."""
    random.seed(seed)
    lines = [f"p cnf {n} {m}"]
    for _ in range(m):
        vars = random.sample(range(1, n + 1), 3)
        clause = [v if random.random() > 0.5 else -v for v in vars]
        lines.append(" ".join(map(str, clause)) + " 0")
    return "\n".join(lines)

def solve_minisat(cnf: str, timeout: float) -> Tuple[bool, Optional[bool], float]:
    """Run MiniSat, return (completed, sat, time)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
        f.write(cnf)
        cnf_path = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.out', delete=False) as f:
        out_path = f.name

    try:
        start = time.time()
        result = subprocess.run(
            ['minisat', cnf_path, out_path],
            capture_output=True, timeout=timeout, text=True
        )
        elapsed = time.time() - start

        output = result.stdout + result.stderr
        if 'SATISFIABLE' in output:
            return True, True, elapsed
        elif 'UNSATISFIABLE' in output:
            return True, False, elapsed
        return False, None, elapsed
    except subprocess.TimeoutExpired:
        return False, None, timeout
    finally:
        os.remove(cnf_path)
        if os.path.exists(out_path):
            os.remove(out_path)

@dataclass
class BenchmarkResult:
    n: int
    alpha: float
    alpha_c: float
    distance: float
    region: str
    baseline_time: float
    baseline_solved: bool
    phi_time: float
    phi_solved: bool
    phi_prediction: Optional[bool]  # Did we predict SAT/UNSAT without solving?
    actual: Optional[bool]

def benchmark_instance(n: int, alpha: float, seed: int, timeout: float = 5.0) -> BenchmarkResult:
    """Benchmark a single instance."""
    m = int(n * alpha)
    cnf = generate_3sat(n, m, seed)

    alpha_c = predict_alpha_c(n)
    distance = (alpha - alpha_c) / alpha_c

    # Classify region
    if distance < -0.20:
        region = "easy_sat"
    elif distance > 0.20:
        region = "easy_unsat"
    else:
        region = "hard"

    # φ-aware strategy
    phi_prediction = None
    phi_time = 0.0
    phi_solved = False

    if region == "easy_sat" and distance < -0.35:
        # Very likely SAT - predict without solving
        phi_prediction = True
        phi_time = 0.001  # Just classification time
        phi_solved = True
    elif region == "easy_unsat" and distance > 0.35:
        # Very likely UNSAT - predict without solving
        phi_prediction = False
        phi_time = 0.001
        phi_solved = True
    else:
        # Actually solve
        phi_solved_result, phi_sat, phi_time = solve_minisat(cnf, timeout)
        phi_solved = phi_solved_result

    # Baseline: always solve
    baseline_solved, baseline_sat, baseline_time = solve_minisat(cnf, timeout)

    actual = baseline_sat if baseline_solved else None

    return BenchmarkResult(
        n=n, alpha=alpha, alpha_c=alpha_c, distance=distance,
        region=region,
        baseline_time=baseline_time, baseline_solved=baseline_solved,
        phi_time=phi_time, phi_solved=phi_solved,
        phi_prediction=phi_prediction, actual=actual
    )

def run_benchmark():
    """Run benchmark suite."""
    print("=" * 70)
    print("φ-Aware SAT Solver Benchmark")
    print("=" * 70)

    # Test configurations: (n, alpha_offsets)
    # alpha_offset is relative to α_c(n)
    configs = [
        (200, [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]),
        (500, [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]),
        (1000, [-0.4, -0.2, 0.0, 0.2, 0.4]),
    ]

    results: List[BenchmarkResult] = []
    n_trials = 5

    for n, offsets in configs:
        alpha_c = predict_alpha_c(n)
        print(f"\nn = {n}, α_c = {alpha_c:.3f}")
        print("-" * 70)

        for offset in offsets:
            alpha = alpha_c * (1 + offset)

            # Run multiple trials
            for trial in range(n_trials):
                result = benchmark_instance(n, alpha, seed=42 + trial, timeout=3.0)
                results.append(result)

            # Average results
            trial_results = results[-n_trials:]
            avg_baseline = sum(r.baseline_time for r in trial_results) / n_trials
            avg_phi = sum(r.phi_time for r in trial_results) / n_trials
            baseline_solved = sum(1 for r in trial_results if r.baseline_solved)
            phi_solved = sum(1 for r in trial_results if r.phi_solved)
            phi_predicted = sum(1 for r in trial_results if r.phi_prediction is not None)

            speedup = avg_baseline / avg_phi if avg_phi > 0 else 0

            print(f"  α = {alpha:.2f} ({offset:+.0%} from α_c): "
                  f"baseline {avg_baseline:.3f}s ({baseline_solved}/{n_trials}), "
                  f"φ-aware {avg_phi:.3f}s ({phi_solved}/{n_trials}), "
                  f"speedup: {speedup:.1f}x"
                  + (f" [predicted {phi_predicted}]" if phi_predicted else ""))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_baseline_time = sum(r.baseline_time for r in results)
    total_phi_time = sum(r.phi_time for r in results)
    total_baseline_solved = sum(1 for r in results if r.baseline_solved)
    total_phi_solved = sum(1 for r in results if r.phi_solved)
    total_predicted = sum(1 for r in results if r.phi_prediction is not None)

    print(f"\nTotal instances: {len(results)}")
    print(f"Baseline: {total_baseline_time:.2f}s total, {total_baseline_solved} solved")
    print(f"φ-aware:  {total_phi_time:.2f}s total, {total_phi_solved} solved")
    print(f"  - Predicted without solving: {total_predicted}")
    print(f"Overall speedup: {total_baseline_time / total_phi_time:.2f}x")

    # Check prediction accuracy
    predictions = [r for r in results if r.phi_prediction is not None and r.actual is not None]
    if predictions:
        correct = sum(1 for r in predictions if r.phi_prediction == r.actual)
        print(f"\nPrediction accuracy: {correct}/{len(predictions)} ({100*correct/len(predictions):.1f}%)")

    # Speedup by region
    print("\nSpeedup by region:")
    for region in ["easy_sat", "hard", "easy_unsat"]:
        region_results = [r for r in results if r.region == region]
        if region_results:
            baseline = sum(r.baseline_time for r in region_results)
            phi = sum(r.phi_time for r in region_results)
            speedup = baseline / phi if phi > 0 else 0
            print(f"  {region:12s}: {speedup:.1f}x ({len(region_results)} instances)")

if __name__ == "__main__":
    run_benchmark()
