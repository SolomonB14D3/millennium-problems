#!/usr/bin/env python3
"""
Ultra-fast 3-SAT scaling using statistical approximation
Key: We only need to estimate P(SAT) and find transition width
"""
import random
import math
import time

print("=" * 70)
print("3-SAT PHASE TRANSITION - FAST SCALING ANALYSIS")
print("=" * 70)
print(flush=True)

PHI = (1 + 5**0.5) / 2

def fast_sat_check(n, m, max_tries=3, max_flips=None):
    """Very fast SAT approximation using multiple WalkSAT runs"""
    if max_flips is None:
        max_flips = n * 10  # Reduced flips for speed

    for _ in range(max_tries):
        # Generate instance
        clauses = []
        for _ in range(m):
            vs = random.sample(range(n), 3)
            clause = tuple((v, random.random() > 0.5) for v in vs)
            clauses.append(clause)

        # Fast WalkSAT
        assign = [random.random() > 0.5 for _ in range(n)]

        for flip in range(max_flips):
            # Find unsatisfied clause
            unsat_idx = None
            for i, clause in enumerate(clauses):
                sat = any(assign[v] == s for v, s in clause)
                if not sat:
                    unsat_idx = i
                    break

            if unsat_idx is None:
                return True  # Satisfied!

            # Flip random var in unsatisfied clause
            v, _ = random.choice(clauses[unsat_idx])
            assign[v] = not assign[v]

    return False  # Probably UNSAT or hard

def estimate_psat(n, alpha, samples=50):
    """Estimate P(SAT) at given alpha"""
    m = int(alpha * n)
    sat_count = sum(1 for _ in range(samples) if fast_sat_check(n, m))
    return sat_count / samples

def find_transition_width(n, samples_per_alpha=30):
    """Find transition width for given n"""
    # Known critical ratio
    alpha_c = 4.267

    # Scan around critical point
    alphas = [alpha_c - 0.5, alpha_c - 0.25, alpha_c, alpha_c + 0.25, alpha_c + 0.5, alpha_c + 0.75]
    probs = []

    for alpha in alphas:
        p = estimate_psat(n, alpha, samples_per_alpha)
        probs.append(p)

    # Find width where P drops from ~0.8 to ~0.2
    # Using linear interpolation
    p_high, p_low = 0.75, 0.25

    alpha_high = alphas[0]
    alpha_low = alphas[-1]

    for i in range(len(alphas) - 1):
        if probs[i] >= p_high and probs[i+1] < p_high:
            f = (p_high - probs[i+1]) / (probs[i] - probs[i+1])
            alpha_high = alphas[i+1] + f * (alphas[i] - alphas[i+1])

        if probs[i] >= p_low and probs[i+1] < p_low:
            f = (p_low - probs[i+1]) / (probs[i] - probs[i+1])
            alpha_low = alphas[i+1] + f * (alphas[i] - alphas[i+1])

    delta = max(0.05, alpha_low - alpha_high)
    return delta, list(zip(alphas, probs))

# Run scaling analysis
sizes = [30, 50, 75, 100, 150, 200, 300, 400, 500]
results = []

print(f"{'n':<8} {'Δα':<10} {'log(n)':<10} {'log(Δα)':<12} {'time':<8}")
print("-" * 55)

for n in sizes:
    start = time.time()
    samples = max(20, 100 - n//10)  # Fewer samples for large n

    delta, curve = find_transition_width(n, samples)
    elapsed = time.time() - start

    log_n = math.log(n)
    log_delta = math.log(delta) if delta > 0 else -5

    results.append((n, delta, log_n, log_delta))
    print(f"{n:<8} {delta:<10.4f} {log_n:<10.3f} {log_delta:<12.3f} {elapsed:<8.1f}s", flush=True)

# Fit power law: Δα ~ n^(-1/ν)
# log(Δα) = const - (1/ν)*log(n)
print()
print("=" * 70)
print("SCALING FIT: Δα ~ n^(-1/ν)")
print("=" * 70)

log_n = [r[2] for r in results]
log_delta = [r[3] for r in results]

# Linear regression
n_pts = len(results)
sx = sum(log_n)
sy = sum(log_delta)
sxy = sum(x*y for x,y in zip(log_n, log_delta))
sxx = sum(x*x for x in log_n)

slope = (n_pts * sxy - sx * sy) / (n_pts * sxx - sx * sx)
intercept = (sy - slope * sx) / n_pts

# Calculate R²
y_mean = sy / n_pts
ss_tot = sum((y - y_mean)**2 for y in log_delta)
ss_res = sum((y - (slope*x + intercept))**2 for x, y in zip(log_n, log_delta))
r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0

inv_nu = -slope
nu = 1 / inv_nu if abs(inv_nu) > 0.01 else float('inf')

print(f"Fit: log(Δα) = {intercept:.3f} + ({slope:.4f}) × log(n)")
print(f"R² = {r_squared:.4f}")
print()

print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Measured 1/ν = {inv_nu:.4f}")
print(f"Measured ν   = {nu:.4f}")
print()

# Compare to targets
dev_phi = abs(inv_nu - 1/PHI) / (1/PHI) * 100
dev_23 = abs(inv_nu - 2/3) / (2/3) * 100

print(f"{'Target':<20} {'Value':<12} {'Deviation':<12}")
print("-" * 45)
print(f"{'1/φ':<20} {1/PHI:<12.4f} {dev_phi:<12.1f}%")
print(f"{'2/3':<20} {2/3:<12.4f} {dev_23:<12.1f}%")
print()

if dev_phi < 2:
    status = "STRONG"
elif dev_phi < 5:
    status = "Moderate"
elif dev_phi < 10:
    status = "Suggestive"
else:
    status = "Weak"

print(f"STATUS: {status} (φ-deviation: {dev_phi:.1f}%)")
print()

if dev_phi < dev_23:
    print("★ 1/ν is CLOSER to 1/φ than to 2/3!")
    print("  This SUPPORTS the φ-hypothesis over classical 3/2.")
else:
    print("  1/ν is closer to 2/3 than to 1/φ")

print()
print("=" * 70)
print(f"SUMMARY: 1/ν = {inv_nu:.4f}, deviation from 1/φ = {dev_phi:.1f}%")
print("=" * 70)
