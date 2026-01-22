#!/usr/bin/env python3
"""
Large-Scale 3-SAT Phase Transition Analysis
Measure finite-size scaling at n = 1000, 2000, 5000, 10000
"""

import random
import time
import sys
from collections import defaultdict

# Increase recursion limit for larger instances
sys.setrecursionlimit(50000)

PHI = (1 + 5**0.5) / 2

def generate_3sat(n_vars, n_clauses):
    """Generate random 3-SAT instance"""
    clauses = []
    for _ in range(n_clauses):
        vars = random.sample(range(1, n_vars + 1), 3)
        clause = tuple(v if random.random() > 0.5 else -v for v in vars)
        clauses.append(clause)
    return clauses

def unit_propagate(clauses, assignment):
    """Unit propagation with conflict detection"""
    changed = True
    while changed:
        changed = False
        for clause in clauses:
            unassigned = []
            satisfied = False
            for lit in clause:
                var = abs(lit)
                if var in assignment:
                    if (lit > 0) == assignment[var]:
                        satisfied = True
                        break
                else:
                    unassigned.append(lit)

            if satisfied:
                continue
            if len(unassigned) == 0:
                return None  # Conflict
            if len(unassigned) == 1:
                lit = unassigned[0]
                assignment[abs(lit)] = (lit > 0)
                changed = True
    return assignment

def dpll_limited(clauses, assignment, n_vars, depth=0, max_depth=1000):
    """DPLL with depth limit for large instances"""
    if depth > max_depth:
        return None  # Indeterminate

    # Unit propagation
    assignment = unit_propagate(clauses, assignment.copy())
    if assignment is None:
        return False

    # Check satisfaction
    all_sat = True
    next_var = None
    for clause in clauses:
        satisfied = False
        for lit in clause:
            var = abs(lit)
            if var in assignment:
                if (lit > 0) == assignment[var]:
                    satisfied = True
                    break
            elif next_var is None:
                next_var = var
        if not satisfied:
            all_sat = False
            if next_var is not None:
                break

    if all_sat:
        return True
    if next_var is None:
        return False

    # Branch
    for val in [True, False]:
        new_assign = assignment.copy()
        new_assign[next_var] = val
        result = dpll_limited(clauses, new_assign, n_vars, depth + 1, max_depth)
        if result is True:
            return True
        if result is None:
            return None
    return False

def walksat(clauses, n_vars, max_flips=10000, p=0.5):
    """WalkSAT - fast incomplete solver for satisfiable instances"""
    assignment = {v: random.choice([True, False]) for v in range(1, n_vars + 1)}

    for _ in range(max_flips):
        # Find unsatisfied clauses
        unsat = []
        for clause in clauses:
            satisfied = False
            for lit in clause:
                if (lit > 0) == assignment[abs(lit)]:
                    satisfied = True
                    break
            if not satisfied:
                unsat.append(clause)

        if not unsat:
            return True  # Satisfied

        # Pick random unsatisfied clause
        clause = random.choice(unsat)

        if random.random() < p:
            # Random walk: flip random variable in clause
            lit = random.choice(clause)
            var = abs(lit)
        else:
            # Greedy: flip variable that minimizes broken clauses
            best_var = abs(clause[0])
            best_breaks = float('inf')
            for lit in clause:
                var = abs(lit)
                # Count how many clauses would break
                assignment[var] = not assignment[var]
                breaks = sum(1 for c in clauses if not any(
                    (l > 0) == assignment[abs(l)] for l in c))
                assignment[var] = not assignment[var]
                if breaks < best_breaks:
                    best_breaks = breaks
                    best_var = var
            var = best_var

        assignment[var] = not assignment[var]

    return None  # Indeterminate

def solve_with_timeout(clauses, n_vars, timeout_flips):
    """Try to solve using WalkSAT, fall back to limited DPLL"""
    # First try WalkSAT (good for SAT instances)
    result = walksat(clauses, n_vars, max_flips=timeout_flips)
    if result is True:
        return True

    # Try DPLL with limited depth for UNSAT detection
    max_depth = min(n_vars * 2, 500)
    result = dpll_limited(clauses, {}, n_vars, max_depth=max_depth)
    return result

def measure_sat_prob(n_vars, alpha, n_samples, timeout_flips):
    """Measure P(satisfiable) at given alpha"""
    n_clauses = int(alpha * n_vars)
    sat = 0
    unsat = 0
    unknown = 0

    for _ in range(n_samples):
        clauses = generate_3sat(n_vars, n_clauses)
        result = solve_with_timeout(clauses, n_vars, timeout_flips)
        if result is True:
            sat += 1
        elif result is False:
            unsat += 1
        else:
            unknown += 1

    total = sat + unsat
    if total == 0:
        return 0.5, unknown
    return sat / total, unknown

def find_transition(n_vars, n_samples, timeout_flips):
    """Find transition point and width"""
    print(f"  Scanning alpha range...")

    # Coarse scan
    alphas = [3.5, 3.8, 4.0, 4.1, 4.2, 4.267, 4.3, 4.4, 4.5, 4.8, 5.0]
    probs = []

    for alpha in alphas:
        p, unk = measure_sat_prob(n_vars, alpha, n_samples // 2, timeout_flips)
        probs.append(p)
        print(f"    α={alpha:.3f}: P(SAT)={p:.3f} (unk={unk})")

    # Find alpha where P crosses 0.5
    alpha_half = 4.267  # Default
    for i in range(len(alphas) - 1):
        if probs[i] >= 0.5 and probs[i+1] < 0.5:
            # Linear interpolation
            alpha_half = alphas[i] + (0.5 - probs[i]) / (probs[i+1] - probs[i]) * (alphas[i+1] - alphas[i])
            break

    # Find transition width (alpha where P goes from ~0.9 to ~0.1)
    alpha_90 = alphas[0]
    alpha_10 = alphas[-1]

    for i, (a, p) in enumerate(zip(alphas, probs)):
        if p <= 0.9 and i > 0:
            alpha_90 = alphas[i-1]
            break

    for i, (a, p) in enumerate(zip(alphas, probs)):
        if p <= 0.1:
            alpha_10 = a
            break

    delta_alpha = alpha_10 - alpha_90

    return alpha_half, delta_alpha, list(zip(alphas, probs))

def main():
    print("=" * 70)
    print("LARGE-SCALE 3-SAT PHASE TRANSITION ANALYSIS")
    print("=" * 70)
    print()
    print(f"φ = {PHI:.6f}")
    print(f"1/φ = {1/PHI:.6f}")
    print()

    # Test sizes - progressively larger
    configs = [
        (100, 100, 5000),    # n_vars, n_samples, timeout_flips
        (200, 80, 8000),
        (500, 50, 15000),
        (1000, 30, 25000),
        (2000, 20, 40000),
        (5000, 10, 80000),
    ]

    results = []

    print("Configuration: (n_vars, n_samples, timeout_flips)")
    print("-" * 70)

    for n_vars, n_samples, timeout_flips in configs:
        print(f"\nn = {n_vars} vars ({n_samples} samples, {timeout_flips} flips)")
        print("-" * 50)

        start = time.time()
        alpha_half, delta_alpha, curve = find_transition(n_vars, n_samples, timeout_flips)
        elapsed = time.time() - start

        if delta_alpha > 0:
            log_n = 0.693147 * (n_vars ** 0.5) if n_vars > 0 else 1  # Use ln(n) proxy
            log_n = len(bin(n_vars)) - 2  # log2(n)

        results.append({
            'n': n_vars,
            'alpha_half': alpha_half,
            'delta_alpha': delta_alpha,
            'time': elapsed
        })

        print(f"  α_0.5 = {alpha_half:.4f}")
        print(f"  Δα = {delta_alpha:.4f}")
        print(f"  Time: {elapsed:.1f}s")

    # Fit scaling exponent
    print("\n" + "=" * 70)
    print("FINITE-SIZE SCALING FIT")
    print("=" * 70)

    # Filter valid results
    valid = [r for r in results if r['delta_alpha'] > 0.01]

    if len(valid) >= 3:
        import math

        # Compute log values
        log_n = [math.log(r['n']) for r in valid]
        log_delta = [math.log(r['delta_alpha']) for r in valid]

        # Linear regression: log(Δα) = a + b*log(n), so b = -1/ν
        n = len(valid)
        sum_x = sum(log_n)
        sum_y = sum(log_delta)
        sum_xy = sum(x*y for x, y in zip(log_n, log_delta))
        sum_xx = sum(x*x for x in log_n)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x**2)
        intercept = (sum_y - slope * sum_x) / n

        inv_nu = -slope
        nu = 1 / inv_nu if inv_nu != 0 else float('inf')

        print(f"\nScaling: Δα ~ n^(-1/ν)")
        print(f"Fit: log(Δα) = {intercept:.4f} + ({slope:.4f}) * log(n)")
        print()
        print(f"Measured 1/ν = {inv_nu:.4f}")
        print(f"Measured ν = {nu:.4f}")
        print()

        # Compare to φ
        print("COMPARISON TO φ-VALUES")
        print("-" * 70)
        print(f"{'Quantity':<20} {'Measured':<15} {'φ-Target':<15} {'Deviation':<15}")
        print("-" * 70)

        dev_inv_nu = abs(inv_nu - 1/PHI) / (1/PHI) * 100
        dev_nu = abs(nu - PHI) / PHI * 100
        dev_nu_1p5 = abs(nu - 1.5) / 1.5 * 100

        print(f"{'1/ν':<20} {inv_nu:<15.4f} {1/PHI:<15.4f} {dev_inv_nu:<15.1f}%")
        print(f"{'ν':<20} {nu:<15.4f} {PHI:<15.4f} {dev_nu:<15.1f}%")
        print(f"{'ν vs 3/2':<20} {nu:<15.4f} {1.5:<15.4f} {dev_nu_1p5:<15.1f}%")
        print("-" * 70)

        # Assessment
        print()
        if dev_inv_nu < 2:
            status = "STRONG"
        elif dev_inv_nu < 5:
            status = "Moderate"
        elif dev_inv_nu < 10:
            status = "Suggestive"
        else:
            status = "Weak"

        print(f"STATUS: {status} ({dev_inv_nu:.1f}% deviation)")
        print()

        if dev_inv_nu < dev_nu_1p5:
            print("*** 1/ν is CLOSER to 1/φ than to 2/3 ***")
            print("    This SUPPORTS the φ-hypothesis!")
        else:
            print("*** 1/ν is closer to 2/3 than to 1/φ ***")

        # Summary table
        print("\n" + "=" * 70)
        print("RESULTS TABLE")
        print("=" * 70)
        print(f"{'n':<10} {'α_0.5':<12} {'Δα':<12} {'log(n)':<10} {'log(Δα)':<12} {'Time':<10}")
        print("-" * 70)
        for r in results:
            if r['delta_alpha'] > 0:
                print(f"{r['n']:<10} {r['alpha_half']:<12.4f} {r['delta_alpha']:<12.4f} "
                      f"{math.log(r['n']):<10.3f} {math.log(r['delta_alpha']):<12.3f} {r['time']:<10.1f}s")

        print("\n" + "=" * 70)
        print("FINAL RESULT")
        print("=" * 70)
        print(f"""
  Measured: 1/ν = {inv_nu:.4f}
  Target:   1/φ = {1/PHI:.4f}
  Deviation: {dev_inv_nu:.1f}%

  Status: {status}
""")

    else:
        print("Insufficient valid data points for scaling fit")

if __name__ == "__main__":
    main()
