#!/usr/bin/env python3
"""Sweep α to find the actual phase transition."""

import subprocess
import tempfile
import os
import random
import time

def generate_3sat(n, m, seed):
    random.seed(seed)
    lines = [f"p cnf {n} {m}"]
    for _ in range(m):
        vars = random.sample(range(1, n + 1), 3)
        clause = [v if random.random() > 0.5 else -v for v in vars]
        lines.append(" ".join(map(str, clause)) + " 0")
    return "\n".join(lines)

def solve(cnf_path, timeout):
    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as f:
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
            return True, elapsed
        elif 'UNSATISFIABLE' in output:
            return False, elapsed
        return None, elapsed
    except subprocess.TimeoutExpired:
        return None, timeout
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)

print("=" * 70)
print("PHASE TRANSITION SWEEP")
print("=" * 70)

n = 400
timeout = 20.0
trials = 10

# Sweep α from 3.0 to 5.5
alphas = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.8, 5.0, 5.5]

print(f"\nn = {n} variables, {trials} trials per α, timeout = {timeout}s")
print(f"\n{'α':>6} {'m':>6} {'P(SAT)':>8} {'P(UNSAT)':>8} {'P(TO)':>8} {'avg_time':>10}")
print("-" * 55)

for alpha in alphas:
    m = int(n * alpha)
    sat_count = 0
    unsat_count = 0
    timeout_count = 0
    total_time = 0

    for trial in range(trials):
        cnf = generate_3sat(n, m, seed=1000 + trial)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
            f.write(cnf)
            path = f.name
        try:
            result, elapsed = solve(path, timeout)
            total_time += elapsed
            if result == True:
                sat_count += 1
            elif result == False:
                unsat_count += 1
            else:
                timeout_count += 1
        finally:
            os.remove(path)

    p_sat = sat_count / trials
    p_unsat = unsat_count / trials
    p_timeout = timeout_count / trials
    avg_time = total_time / trials

    # Mark the transition region
    marker = ""
    if 0.3 < p_sat < 0.7 and p_timeout < 0.3:
        marker = " ← TRANSITION"
    elif p_timeout > 0.5:
        marker = " ← HARD"

    print(f"{alpha:>6.2f} {m:>6} {p_sat:>8.0%} {p_unsat:>8.0%} {p_timeout:>8.0%} {avg_time:>9.2f}s{marker}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("""
The phase transition occurs where P(SAT) drops from ~100% to ~0%.
The HARD region is where instances timeout.

φ-structure predicts α_c(400) ≈ 3.57 (interpolated from data).

If the actual transition is at a different α, this helps calibrate
the prediction model for more accurate triage.
""")
