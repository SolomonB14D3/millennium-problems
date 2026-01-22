#!/usr/bin/env python3
"""
Test the limitations of φ-SAT:
1. Does it work on CRAFTED instances (not just random)?
2. Does the buffer need to shrink as n grows?
"""

import subprocess
import tempfile
import os
import random
import time
import math
from typing import Optional, Tuple

# φ-SAT predictor
ALPHA_C_TABLE = [
    (500, 3.573), (2000, 4.497), (4000, 4.996),
    (8000, 4.996), (64000, 9.996),
]

def predict_alpha_c(n):
    if n <= ALPHA_C_TABLE[0][0]: return ALPHA_C_TABLE[0][1]
    if n >= ALPHA_C_TABLE[-1][0]: return ALPHA_C_TABLE[-1][1]
    for i in range(len(ALPHA_C_TABLE) - 1):
        n1, a1 = ALPHA_C_TABLE[i]
        n2, a2 = ALPHA_C_TABLE[i + 1]
        if n1 <= n <= n2:
            t = (math.log(n) - math.log(n1)) / (math.log(n2) - math.log(n1))
            return a1 + t * (a2 - a1)
    return 4.267

def phi_predict(n, m, threshold=0.25):
    alpha = m / n
    alpha_c = predict_alpha_c(n)
    distance = (alpha - alpha_c) / alpha_c
    if distance < -threshold:
        return True, abs(distance)  # SAT
    elif distance > threshold:
        return False, abs(distance)  # UNSAT
    return None, 0.0

def solve_kissat(cnf_path, timeout=30.0):
    try:
        start = time.time()
        result = subprocess.run(['kissat', '--quiet', cnf_path],
                                capture_output=True, timeout=timeout)
        elapsed = time.time() - start
        if result.returncode == 10: return True, elapsed
        elif result.returncode == 20: return False, elapsed
        return None, elapsed
    except subprocess.TimeoutExpired:
        return None, timeout

def generate_random_3sat(n, m, seed):
    random.seed(seed)
    lines = [f"p cnf {n} {m}"]
    for _ in range(m):
        vars = random.sample(range(1, n + 1), 3)
        clause = [v if random.random() > 0.5 else -v for v in vars]
        lines.append(" ".join(map(str, clause)) + " 0")
    return "\n".join(lines)

def generate_crafted_sat(n, m):
    """
    Generate a CRAFTED instance that LOOKS under-constrained
    but has hidden structure making it hard.

    Strategy: Create chains of implications that look sparse
    but create deep dependencies.
    """
    lines = [f"p cnf {n} {m}"]
    clauses_added = 0

    # Create implication chains: x1 -> x2 -> x3 -> ... -> xk -> -x1
    # This creates a contradiction that's hard to find
    chain_length = min(n // 2, 50)

    # Add the implication chain (each implication is 2 clauses)
    for i in range(chain_length - 1):
        # xi -> x(i+1) is equivalent to (-xi OR x(i+1))
        lines.append(f"-{i+1} {i+2} 0")
        clauses_added += 1

    # Close the loop: xk -> -x1
    lines.append(f"-{chain_length} -{1} 0")
    clauses_added += 1

    # Force x1 to be true (so the chain must propagate)
    lines.append(f"{1} 0")
    clauses_added += 1

    # Fill remaining clauses with random "noise"
    while clauses_added < m:
        vars = random.sample(range(1, n + 1), 3)
        clause = [v if random.random() > 0.5 else -v for v in vars]
        lines.append(" ".join(map(str, clause)) + " 0")
        clauses_added += 1

    # Update header with actual clause count
    lines[0] = f"p cnf {n} {clauses_added}"
    return "\n".join(lines)

def generate_crafted_unsat(n, m):
    """
    Generate a CRAFTED UNSAT instance that LOOKS over-constrained normally
    but has a specific small unsatisfiable core.
    """
    lines = [f"p cnf {n} {m}"]

    # Unsatisfiable core: x AND -x (using 3-SAT clauses)
    # (x OR x OR x) AND (-x OR -x OR -x)
    lines.append("1 1 1 0")
    lines.append("-1 -1 -1 0")

    # Fill with random clauses
    for _ in range(m - 2):
        vars = random.sample(range(2, n + 1), 3)  # Avoid var 1
        clause = [v if random.random() > 0.5 else -v for v in vars]
        lines.append(" ".join(map(str, clause)) + " 0")

    return "\n".join(lines)


print("=" * 70)
print("TEST 1: CRAFTED INSTANCES")
print("=" * 70)
print("""
Testing if φ-SAT fails on CRAFTED instances where:
- The α ratio looks "easy" (far from α_c)
- But hidden structure makes it hard/different than predicted
""")

# Test crafted SAT-looking instances
print("\n--- Crafted instances that LOOK like easy SAT ---")
print(f"{'n':>6} {'m':>6} {'α':>6} {'α_c':>6} {'φ-pred':>8} {'actual':>8} {'match':>6}")
print("-" * 55)

crafted_errors = 0
for n in [100, 200, 300]:
    alpha_c = predict_alpha_c(n)
    # Make α look low (should predict SAT)
    alpha = alpha_c * 0.6  # 40% below α_c
    m = int(n * alpha)

    cnf = generate_crafted_sat(n, m)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
        f.write(cnf)
        path = f.name

    try:
        phi_pred, _ = phi_predict(n, m)
        actual, _ = solve_kissat(path, timeout=30)

        phi_str = "SAT" if phi_pred else "UNSAT" if phi_pred is False else "?"
        act_str = "SAT" if actual else "UNSAT" if actual is False else "T/O"
        match = "✓" if phi_pred == actual else "✗"
        if phi_pred != actual and actual is not None:
            crafted_errors += 1

        print(f"{n:>6} {m:>6} {alpha:>6.2f} {alpha_c:>6.2f} {phi_str:>8} {act_str:>8} {match:>6}")
    finally:
        os.remove(path)

# Test crafted UNSAT-looking instances
print("\n--- Crafted instances that LOOK like easy UNSAT ---")
print(f"{'n':>6} {'m':>6} {'α':>6} {'α_c':>6} {'φ-pred':>8} {'actual':>8} {'match':>6}")
print("-" * 55)

for n in [100, 200, 300]:
    alpha_c = predict_alpha_c(n)
    alpha = alpha_c * 1.5  # 50% above α_c
    m = int(n * alpha)

    cnf = generate_crafted_unsat(n, m)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
        f.write(cnf)
        path = f.name

    try:
        phi_pred, _ = phi_predict(n, m)
        actual, _ = solve_kissat(path, timeout=30)

        phi_str = "SAT" if phi_pred else "UNSAT" if phi_pred is False else "?"
        act_str = "SAT" if actual else "UNSAT" if actual is False else "T/O"
        match = "✓" if phi_pred == actual else "✗"
        if phi_pred != actual and actual is not None:
            crafted_errors += 1

        print(f"{n:>6} {m:>6} {alpha:>6.2f} {alpha_c:>6.2f} {phi_str:>8} {act_str:>8} {match:>6}")
    finally:
        os.remove(path)

print(f"\nCrafted instance errors: {crafted_errors}")


print("\n" + "=" * 70)
print("TEST 2: BUFFER SHARPNESS vs n")
print("=" * 70)
print("""
Testing if the 25% buffer stays valid as n increases.
If the transition sharpens, we might need a smaller buffer for large n.
""")

# Test at different n with different buffers
print(f"\n{'n':>6} {'buffer':>8} {'trials':>7} {'correct':>8} {'accuracy':>10}")
print("-" * 45)

buffers = [0.30, 0.25, 0.20, 0.15]
n_values = [100, 300, 500, 800]

for n in n_values:
    alpha_c = predict_alpha_c(n)

    for buffer in buffers:
        correct = 0
        total = 0

        # Test instances at the buffer boundary
        for offset in [-buffer - 0.05, -buffer + 0.05, buffer - 0.05, buffer + 0.05]:
            alpha = alpha_c * (1 + offset)
            m = int(n * alpha)

            for seed in range(5):
                cnf = generate_random_3sat(n, m, seed)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
                    f.write(cnf)
                    path = f.name

                try:
                    phi_pred, _ = phi_predict(n, m, threshold=buffer)
                    actual, _ = solve_kissat(path, timeout=15)

                    if phi_pred is not None and actual is not None:
                        total += 1
                        if phi_pred == actual:
                            correct += 1
                finally:
                    os.remove(path)

        accuracy = correct / total if total > 0 else 0
        print(f"{n:>6} {buffer:>7.0%} {total:>7} {correct:>8} {accuracy:>9.0%}")


print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)

if crafted_errors > 0:
    print(f"""
⚠ CRAFTED INSTANCES: {crafted_errors} errors

The φ-structure is specific to RANDOM 3-SAT.
Crafted/industrial instances can violate predictions because:
1. They don't follow the random variable distribution
2. Hidden structure can make "easy-looking" α values hard
3. Real-world SAT is NOT random

RECOMMENDATION: Only use φ-SAT for random/uniform instances.
""")
else:
    print("""
✓ Crafted instance tests passed (sample size is small though)
""")

print("""
BUFFER ANALYSIS:
- If accuracy drops as n increases for a fixed buffer → need adaptive buffer
- If accuracy stays high → the 25% buffer is stable

KEY INSIGHT: φ-SAT is a heuristic for RANDOM 3-SAT, not a general SAT predictor.
For the P vs NP Millennium Prize, we'd need it to work on ALL instances.
""")
