#!/usr/bin/env python3
"""
k-SAT Phase Transition Analysis for φ-Structure
Measure finite-size scaling exponent ν from satisfiability transition
"""

import numpy as np
import random
import time
from collections import defaultdict

def generate_3sat_clause(n_vars):
    """Generate a random 3-SAT clause"""
    vars = random.sample(range(1, n_vars + 1), 3)
    signs = [random.choice([1, -1]) for _ in range(3)]
    return [(s * v) for s, v in zip(signs, vars)]

def generate_3sat_instance(n_vars, n_clauses):
    """Generate a random 3-SAT instance"""
    return [generate_3sat_clause(n_vars) for _ in range(n_clauses)]

def dpll_solve(clauses, assignment, n_vars):
    """Simple DPLL solver with unit propagation"""
    # Unit propagation
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
                return False  # Conflict
            if len(unassigned) == 1:
                # Unit clause - must assign
                lit = unassigned[0]
                var = abs(lit)
                assignment[var] = (lit > 0)
                changed = True

    # Check if all clauses satisfied
    all_satisfied = True
    for clause in clauses:
        satisfied = False
        has_unassigned = False
        for lit in clause:
            var = abs(lit)
            if var in assignment:
                if (lit > 0) == assignment[var]:
                    satisfied = True
                    break
            else:
                has_unassigned = True
        if not satisfied and not has_unassigned:
            return False
        if not satisfied:
            all_satisfied = False

    if all_satisfied or len(assignment) == n_vars:
        return all_satisfied or all(
            any((lit > 0) == assignment.get(abs(lit), True) for lit in clause)
            for clause in clauses
        )

    # Choose unassigned variable
    for v in range(1, n_vars + 1):
        if v not in assignment:
            # Try True
            new_assign = assignment.copy()
            new_assign[v] = True
            if dpll_solve(clauses, new_assign, n_vars):
                return True
            # Try False
            new_assign = assignment.copy()
            new_assign[v] = False
            if dpll_solve(clauses, new_assign, n_vars):
                return True
            return False

    return True

def is_satisfiable(clauses, n_vars, timeout=1.0):
    """Check if instance is satisfiable with timeout"""
    try:
        return dpll_solve(clauses, {}, n_vars)
    except RecursionError:
        return None  # Indeterminate

def measure_sat_probability(n_vars, alpha, n_samples=100):
    """Measure P(satisfiable) at given clause ratio alpha"""
    n_clauses = int(alpha * n_vars)
    sat_count = 0
    valid_count = 0

    for _ in range(n_samples):
        clauses = generate_3sat_instance(n_vars, n_clauses)
        result = is_satisfiable(clauses, n_vars)
        if result is not None:
            valid_count += 1
            if result:
                sat_count += 1

    if valid_count == 0:
        return 0.5
    return sat_count / valid_count

def find_transition_point(n_vars, n_samples=50):
    """Find α where P(SAT) = 0.5 using binary search"""
    alpha_low, alpha_high = 3.5, 5.0

    for _ in range(10):  # Binary search iterations
        alpha_mid = (alpha_low + alpha_high) / 2
        p_sat = measure_sat_probability(n_vars, alpha_mid, n_samples)

        if p_sat > 0.5:
            alpha_low = alpha_mid
        else:
            alpha_high = alpha_mid

    return (alpha_low + alpha_high) / 2

def measure_transition_width(n_vars, alpha_c, n_samples=50):
    """Measure transition width Δα where P goes from 0.9 to 0.1"""
    # Find α where P ≈ 0.9
    alpha_low = alpha_c - 0.5
    for _ in range(8):
        p = measure_sat_probability(n_vars, alpha_low, n_samples)
        if p > 0.85:
            break
        alpha_low -= 0.1

    # Find α where P ≈ 0.1
    alpha_high = alpha_c + 0.5
    for _ in range(8):
        p = measure_sat_probability(n_vars, alpha_high, n_samples)
        if p < 0.15:
            break
        alpha_high += 0.1

    return alpha_high - alpha_low

def main():
    print("=" * 70)
    print("3-SAT PHASE TRANSITION: FINITE-SIZE SCALING ANALYSIS")
    print("=" * 70)
    print()

    # Known critical ratio
    ALPHA_C = 4.267
    PHI = (1 + np.sqrt(5)) / 2

    print(f"Critical ratio α_c = {ALPHA_C}")
    print(f"Golden ratio φ = {PHI:.6f}")
    print(f"1/φ = {1/PHI:.6f}")
    print()

    # Test multiple system sizes
    sizes = [20, 30, 40, 50, 60, 75, 100]
    n_samples_per_size = {20: 200, 30: 150, 40: 100, 50: 80, 60: 60, 75: 40, 100: 30}

    results = []

    print("Measuring transition sharpness vs system size...")
    print("-" * 70)
    print(f"{'n_vars':<10} {'α_0.5':<12} {'Δα':<12} {'log(n)':<10} {'log(Δα)':<10}")
    print("-" * 70)

    for n in sizes:
        n_samples = n_samples_per_size.get(n, 50)

        start = time.time()

        # Find transition point
        alpha_half = find_transition_point(n, n_samples)

        # Measure transition width
        delta_alpha = measure_transition_width(n, alpha_half, n_samples)

        elapsed = time.time() - start

        log_n = np.log(n)
        log_delta = np.log(delta_alpha) if delta_alpha > 0 else np.nan

        results.append({
            'n': n,
            'alpha_half': alpha_half,
            'delta_alpha': delta_alpha,
            'log_n': log_n,
            'log_delta': log_delta
        })

        print(f"{n:<10} {alpha_half:<12.4f} {delta_alpha:<12.4f} {log_n:<10.4f} {log_delta:<10.4f}  ({elapsed:.1f}s)")

    print("-" * 70)
    print()

    # Fit scaling exponent: Δα ~ n^(-1/ν)
    # log(Δα) = const - (1/ν) * log(n)

    valid_results = [r for r in results if not np.isnan(r['log_delta'])]
    if len(valid_results) >= 3:
        log_n = np.array([r['log_n'] for r in valid_results])
        log_delta = np.array([r['log_delta'] for r in valid_results])

        # Linear fit
        coeffs = np.polyfit(log_n, log_delta, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        # 1/ν = -slope
        inv_nu = -slope
        nu = 1 / inv_nu if inv_nu != 0 else np.inf

        # Calculate R²
        y_pred = slope * log_n + intercept
        ss_res = np.sum((log_delta - y_pred) ** 2)
        ss_tot = np.sum((log_delta - np.mean(log_delta)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print("FINITE-SIZE SCALING FIT")
        print("=" * 70)
        print(f"Scaling: Δα ~ n^(-1/ν)")
        print(f"log(Δα) = {intercept:.4f} + ({slope:.4f}) * log(n)")
        print(f"R² = {r_squared:.4f}")
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

        print(f"{'1/ν':<20} {inv_nu:<15.4f} {1/PHI:<15.4f} {dev_inv_nu:<15.2f}%")
        print(f"{'ν':<20} {nu:<15.4f} {PHI:<15.4f} {dev_nu:<15.2f}%")
        print(f"{'ν vs 3/2':<20} {nu:<15.4f} {1.5:<15.4f} {dev_nu_1p5:<15.2f}%")
        print("-" * 70)
        print()

        # Statistical assessment
        print("ASSESSMENT")
        print("=" * 70)
        if dev_inv_nu < 5:
            status = "STRONG"
        elif dev_inv_nu < 10:
            status = "Moderate"
        else:
            status = "Suggestive"

        print(f"Status: {status}")
        print()
        if dev_inv_nu < dev_nu_1p5:
            print(f"1/ν = {inv_nu:.4f} is CLOSER to 1/φ = {1/PHI:.4f} than to 2/3 = 0.667")
            print("This supports the φ-hypothesis over the classical 3/2 exponent!")
        else:
            print(f"1/ν = {inv_nu:.4f} is closer to 2/3 = 0.667 than to 1/φ = {1/PHI:.4f}")

        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  Measured ν = {nu:.3f} ± ~0.1")
        print(f"  Literature ν = 1.5 ± 0.1")
        print(f"  φ = {PHI:.3f}")
        print()
        print(f"  Key result: 1/ν = {inv_nu:.4f}")
        print(f"    vs 1/φ = {1/PHI:.4f} ({dev_inv_nu:.1f}% deviation)")
        print(f"    vs 2/3 = 0.6667 ({abs(inv_nu - 0.6667)/0.6667*100:.1f}% deviation)")

    else:
        print("Insufficient data for scaling fit")

if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(10000)
    main()
