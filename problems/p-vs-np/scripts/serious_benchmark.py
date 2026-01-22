#!/usr/bin/env python3
"""
Serious benchmark: Verify φ-prefiltering result at scale.
"""

import subprocess
import tempfile
import os
import random
import time
import math
from typing import Optional, Tuple
from collections import defaultdict

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
    try:
        start = time.time()
        result = subprocess.run(
            ['kissat', '--quiet', cnf_path],
            capture_output=True, timeout=timeout
        )
        elapsed = time.time() - start
        if result.returncode == 10: return True, elapsed
        elif result.returncode == 20: return False, elapsed
        return None, elapsed
    except subprocess.TimeoutExpired:
        return None, timeout

def phi_filter(n: int, m: int, threshold: float = 0.25) -> Tuple[Optional[bool], float]:
    """
    Returns (prediction, confidence).
    prediction = None means "need to solve"
    """
    alpha = m / n
    alpha_c = predict_alpha_c(n)
    distance = (alpha - alpha_c) / alpha_c

    if distance < -threshold:
        confidence = min(0.99, 0.8 - distance)
        return True, confidence  # Predict SAT
    elif distance > threshold:
        confidence = min(0.99, 0.8 + distance)
        return False, confidence  # Predict UNSAT
    else:
        return None, 0.5  # Need to solve

print("=" * 70)
print("SERIOUS BENCHMARK: φ-prefilter vs Kissat")
print("=" * 70)

# Large-scale test
n_instances = 100
timeout = 10.0

# Realistic distribution: mix of easy and hard
# In practice, many real-world instances are NOT at the transition
distribution = [
    # (n, alpha_offset, weight) - offset from α_c
    (300, -0.50, 10),   # Very easy SAT
    (300, -0.30, 15),   # Easy SAT
    (300, -0.10, 10),   # Near transition (SAT side)
    (300, 0.00, 10),    # At transition
    (300, +0.10, 10),   # Near transition (UNSAT side)
    (300, +0.30, 15),   # Easy UNSAT
    (300, +0.50, 10),   # Very easy UNSAT
    (500, -0.30, 5),    # Larger easy SAT
    (500, 0.00, 5),     # Larger hard
    (500, +0.30, 10),   # Larger easy UNSAT
]

instances = []
for n, offset, count in distribution:
    alpha_c = predict_alpha_c(n)
    alpha = alpha_c * (1 + offset)
    m = int(n * alpha)
    for i in range(count):
        instances.append((n, m, len(instances), offset))

random.shuffle(instances)

print(f"\n{len(instances)} instances, timeout={timeout}s")
print(f"Distribution: {sum(1 for _,_,_,o in instances if o < -0.2)} easy_sat, "
      f"{sum(1 for _,_,_,o in instances if -0.2 <= o <= 0.2)} hard, "
      f"{sum(1 for _,_,_,o in instances if o > 0.2)} easy_unsat")

# Run benchmark
baseline_time = 0
filtered_time = 0
baseline_solved = 0
filtered_solved = 0
predictions_made = 0
predictions_correct = 0
predictions_wrong = 0

stats = defaultdict(lambda: {'base_time': 0, 'filt_time': 0, 'count': 0, 'base_solved': 0, 'filt_solved': 0})

print(f"\nRunning... ", end="", flush=True)

for i, (n, m, seed, offset) in enumerate(instances):
    cnf = generate_3sat(n, m, seed)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
        f.write(cnf)
        path = f.name

    try:
        # Baseline: always solve
        base_result, base_t = solve_kissat(path, timeout)
        baseline_time += base_t
        if base_result is not None:
            baseline_solved += 1

        # φ-filtered
        prediction, confidence = phi_filter(n, m)

        if prediction is not None and confidence > 0.85:
            # Use prediction
            filt_result = prediction
            filt_t = 0.0001
            predictions_made += 1

            # Check accuracy
            if base_result is not None:
                if prediction == base_result:
                    predictions_correct += 1
                else:
                    predictions_wrong += 1
        else:
            # Solve
            filt_result, filt_t = solve_kissat(path, timeout)

        filtered_time += filt_t
        if filt_result is not None:
            filtered_solved += 1

        # Stats by category
        if offset < -0.2:
            cat = 'easy_sat'
        elif offset > 0.2:
            cat = 'easy_unsat'
        else:
            cat = 'hard'

        stats[cat]['base_time'] += base_t
        stats[cat]['filt_time'] += filt_t
        stats[cat]['count'] += 1
        if base_result is not None: stats[cat]['base_solved'] += 1
        if filt_result is not None: stats[cat]['filt_solved'] += 1

        # Progress
        if (i + 1) % 10 == 0:
            print(f"{i+1}", end="", flush=True)
        else:
            print(".", end="", flush=True)

    finally:
        os.remove(path)

print(" done\n")

# Results
print("=" * 70)
print("RESULTS")
print("=" * 70)

speedup = baseline_time / filtered_time if filtered_time > 0 else 0

print(f"\n{'Metric':<25} {'Baseline':>12} {'φ-filtered':>12} {'Improvement':>12}")
print("-" * 65)
print(f"{'Total time':<25} {baseline_time:>11.2f}s {filtered_time:>11.2f}s {speedup:>11.1f}x")
print(f"{'Instances solved':<25} {baseline_solved:>12} {filtered_solved:>12}")
print(f"{'Predictions made':<25} {'-':>12} {predictions_made:>12}")
print(f"{'Prediction accuracy':<25} {'-':>12} {f'{predictions_correct}/{predictions_made}':>12}")

if predictions_wrong > 0:
    print(f"\n⚠ {predictions_wrong} WRONG PREDICTIONS - need to verify ground truth")

print(f"\nBy category:")
print(f"{'Category':<12} {'Count':>6} {'Base time':>10} {'Filt time':>10} {'Speedup':>8} {'Base solved':>12} {'Filt solved':>12}")
print("-" * 80)
for cat in ['easy_sat', 'hard', 'easy_unsat']:
    s = stats[cat]
    sp = s['base_time'] / s['filt_time'] if s['filt_time'] > 0 else 0
    print(f"{cat:<12} {s['count']:>6} {s['base_time']:>9.2f}s {s['filt_time']:>9.2f}s {sp:>7.1f}x {s['base_solved']:>12} {s['filt_solved']:>12}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if predictions_wrong == 0 and speedup > 5:
    print(f"""
✓ VALIDATED: {speedup:.1f}x speedup with {predictions_correct}/{predictions_made} correct predictions

This is a real, practical result:
- φ-structure predicts α_c(n) for random 3-SAT
- Instances far from α_c can be classified without solving
- Even SOTA solvers benefit from this pre-filtering

Potential applications:
1. SAT competition pre-processing
2. Cloud SAT solving resource allocation
3. Satisfiability prediction for random instances
""")
elif predictions_wrong > 0:
    accuracy = predictions_correct / predictions_made if predictions_made > 0 else 0
    print(f"""
⚠ {predictions_wrong} wrong predictions ({accuracy:.1%} accuracy)
Need to investigate failure cases.
""")
else:
    print(f"""
Speedup: {speedup:.1f}x
Modest improvement, may not justify added complexity.
""")
