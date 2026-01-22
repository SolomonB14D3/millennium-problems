#!/usr/bin/env python3
"""
Test: Does φ-prefiltering help a SOTA solver (Kissat)?

Experiment:
1. Generate mix of easy and hard instances
2. Baseline: Run Kissat on all
3. φ-filtered: Predict easy cases, only run Kissat on uncertain ones
4. Compare total time and accuracy
"""

import subprocess
import tempfile
import os
import random
import time
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

# φ-structure constants
ALPHA_C_DATA = [
    (500, 3.573), (2000, 4.497), (4000, 4.996),
    (8000, 4.996), (64000, 9.996),
]

def predict_alpha_c(n: int) -> float:
    if n <= ALPHA_C_DATA[0][0]: return ALPHA_C_DATA[0][1]
    if n >= ALPHA_C_DATA[-1][0]: return ALPHA_C_DATA[-1][1]
    for i in range(len(ALPHA_C_DATA) - 1):
        n1, a1 = ALPHA_C_DATA[i]
        n2, a2 = ALPHA_C_DATA[i + 1]
        if n1 <= n <= n2:
            t = (math.log(n) - math.log(n1)) / (math.log(n2) - math.log(n1))
            return a1 + t * (a2 - a1)
    return 4.267

def generate_3sat(n: int, m: int, seed: int) -> str:
    random.seed(seed)
    lines = [f"p cnf {n} {m}"]
    for _ in range(m):
        vars = random.sample(range(1, n + 1), 3)
        clause = [v if random.random() > 0.5 else -v for v in vars]
        lines.append(" ".join(map(str, clause)) + " 0")
    return "\n".join(lines)

def solve_kissat(cnf_path: str, timeout: float) -> Tuple[Optional[bool], float]:
    """Run Kissat solver."""
    try:
        start = time.time()
        result = subprocess.run(
            ['kissat', '--quiet', cnf_path],
            capture_output=True, timeout=timeout, text=True
        )
        elapsed = time.time() - start

        if result.returncode == 10:  # SAT
            return True, elapsed
        elif result.returncode == 20:  # UNSAT
            return False, elapsed
        return None, elapsed
    except subprocess.TimeoutExpired:
        return None, timeout

def phi_predict(n: int, m: int) -> Tuple[str, Optional[bool], float]:
    """
    φ-based prediction.
    Returns: (category, prediction, confidence)
    """
    alpha = m / n
    alpha_c = predict_alpha_c(n)
    distance = (alpha - alpha_c) / alpha_c

    if distance < -0.25:
        return "easy_sat", True, min(0.95, 0.7 - distance)
    elif distance > 0.25:
        return "easy_unsat", False, min(0.95, 0.7 + distance)
    else:
        return "hard", None, 0.5

@dataclass
class Instance:
    n: int
    m: int
    seed: int
    alpha: float
    category: str

@dataclass
class Result:
    instance: Instance
    baseline_result: Optional[bool]
    baseline_time: float
    filtered_result: Optional[bool]
    filtered_time: float
    prediction: Optional[bool]
    prediction_correct: Optional[bool]

def run_experiment():
    print("=" * 70)
    print("φ-PREFILTER TEST: Does knowing α_c help Kissat?")
    print("=" * 70)

    # Generate diverse instance mix
    instances: List[Instance] = []

    # Mix of easy and hard instances
    configs = [
        # (n, alpha_relative_to_ac, count)
        (200, -0.40, 5),   # Easy SAT
        (200, -0.20, 5),   # Moderate SAT
        (200, 0.00, 5),    # Hard (at transition)
        (200, +0.20, 5),   # Moderate UNSAT
        (200, +0.40, 5),   # Easy UNSAT

        (400, -0.30, 3),
        (400, 0.00, 3),
        (400, +0.30, 3),
    ]

    for n, offset, count in configs:
        alpha_c = predict_alpha_c(n)
        alpha = alpha_c * (1 + offset)
        m = int(n * alpha)

        category = "easy_sat" if offset < -0.2 else "easy_unsat" if offset > 0.2 else "hard"

        for i in range(count):
            instances.append(Instance(n, m, seed=1000 + len(instances), alpha=alpha, category=category))

    random.shuffle(instances)  # Randomize order

    print(f"\nGenerated {len(instances)} instances:")
    print(f"  Easy SAT: {sum(1 for i in instances if i.category == 'easy_sat')}")
    print(f"  Hard: {sum(1 for i in instances if i.category == 'hard')}")
    print(f"  Easy UNSAT: {sum(1 for i in instances if i.category == 'easy_unsat')}")

    timeout = 5.0
    results: List[Result] = []

    print(f"\nRunning with timeout = {timeout}s per instance...")
    print("-" * 70)

    baseline_total_time = 0
    filtered_total_time = 0

    for i, inst in enumerate(instances):
        # Generate CNF
        cnf = generate_3sat(inst.n, inst.m, inst.seed)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
            f.write(cnf)
            cnf_path = f.name

        try:
            # φ prediction
            pred_category, prediction, confidence = phi_predict(inst.n, inst.m)

            # Baseline: always run Kissat
            baseline_result, baseline_time = solve_kissat(cnf_path, timeout)
            baseline_total_time += baseline_time

            # Filtered: use prediction for high-confidence cases
            if confidence > 0.8 and prediction is not None:
                # Skip solver, use prediction
                filtered_result = prediction
                filtered_time = 0.001  # Just prediction time
            else:
                # Run solver
                filtered_result, filtered_time = solve_kissat(cnf_path, timeout)
            filtered_total_time += filtered_time

            # Check prediction accuracy
            if prediction is not None and baseline_result is not None:
                pred_correct = (prediction == baseline_result)
            else:
                pred_correct = None

            results.append(Result(
                instance=inst,
                baseline_result=baseline_result,
                baseline_time=baseline_time,
                filtered_result=filtered_result,
                filtered_time=filtered_time,
                prediction=prediction,
                prediction_correct=pred_correct
            ))

            # Progress
            status = "." if baseline_result is not None else "T"
            print(status, end="", flush=True)

        finally:
            os.remove(cnf_path)

    print("\n")

    # Analysis
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nTotal time:")
    print(f"  Baseline (always solve): {baseline_total_time:.2f}s")
    print(f"  φ-filtered:              {filtered_total_time:.2f}s")
    print(f"  Speedup:                 {baseline_total_time / filtered_total_time:.2f}x")

    # Accuracy
    predictions_made = [r for r in results if r.prediction is not None and r.baseline_result is not None]
    if predictions_made:
        correct = sum(1 for r in predictions_made if r.prediction_correct)
        print(f"\nPrediction accuracy: {correct}/{len(predictions_made)} ({100*correct/len(predictions_made):.1f}%)")

    # Breakdown by category
    print(f"\nBy category:")
    for cat in ["easy_sat", "hard", "easy_unsat"]:
        cat_results = [r for r in results if r.instance.category == cat]
        if cat_results:
            base_time = sum(r.baseline_time for r in cat_results)
            filt_time = sum(r.filtered_time for r in cat_results)
            solved = sum(1 for r in cat_results if r.baseline_result is not None)
            speedup = base_time / filt_time if filt_time > 0 else 0
            print(f"  {cat:12s}: baseline={base_time:.2f}s, filtered={filt_time:.2f}s, speedup={speedup:.1f}x, solved={solved}/{len(cat_results)}")

    # Errors (wrong predictions)
    errors = [r for r in results if r.prediction_correct == False]
    if errors:
        print(f"\nPrediction errors: {len(errors)}")
        for r in errors[:5]:
            print(f"  n={r.instance.n}, α={r.instance.alpha:.2f}: predicted {r.prediction}, actual {r.baseline_result}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    speedup = baseline_total_time / filtered_total_time
    if speedup > 1.5:
        print(f"\n✓ φ-prefiltering provides {speedup:.1f}x speedup")
        print("  Worth using for batch processing / job scheduling")
    elif speedup > 1.1:
        print(f"\n~ Modest {speedup:.1f}x speedup from φ-prefiltering")
        print("  Marginal benefit, may not be worth the complexity")
    else:
        print(f"\n✗ No significant speedup ({speedup:.1f}x)")
        print("  Kissat's internal heuristics are already adaptive")

if __name__ == "__main__":
    run_experiment()
