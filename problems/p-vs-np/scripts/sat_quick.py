#!/usr/bin/env python3
"""Quick 3-SAT scaling analysis with immediate output"""
import random
import sys
import math

print("3-SAT FINITE-SIZE SCALING", flush=True)
print("=" * 60, flush=True)

PHI = (1 + 5**0.5) / 2
print(f"Target: 1/φ = {1/PHI:.4f}", flush=True)
print(flush=True)

def gen_3sat(n, m):
    return [tuple(random.choice([-1,1])*v for v in random.sample(range(1,n+1),3)) for _ in range(m)]

def walksat(clauses, n, flips=5000):
    a = {v: random.random()>0.5 for v in range(1,n+1)}
    for _ in range(flips):
        unsat = [c for c in clauses if not any((l>0)==a[abs(l)] for l in c)]
        if not unsat: return True
        c = random.choice(unsat)
        v = abs(random.choice(c))
        a[v] = not a[v]
    return None

def measure(n, alpha, samples, flips):
    m = int(alpha * n)
    sat = sum(1 for _ in range(samples) if walksat(gen_3sat(n,m), n, flips) is True)
    return sat / samples

# Test sizes
sizes = [50, 100, 200, 400, 800]
results = []

print(f"{'n':<8} {'Δα':<10} {'log(n)':<10} {'log(Δα)':<10}", flush=True)
print("-" * 40, flush=True)

for n in sizes:
    samples = max(20, 200 // (n // 50))
    flips = n * 50

    # Find transition width by scanning
    alphas = [3.5, 4.0, 4.267, 4.5, 5.0]
    probs = [measure(n, a, samples, flips) for a in alphas]

    # Find where P crosses 0.9 and 0.1
    a90 = 3.5
    a10 = 5.0
    for a, p in zip(alphas, probs):
        if p < 0.9 and a90 == 3.5: a90 = a
        if p < 0.1: a10 = a; break

    delta = max(0.1, a10 - a90)
    results.append((n, delta))

    print(f"{n:<8} {delta:<10.3f} {math.log(n):<10.3f} {math.log(delta):<10.3f}", flush=True)

# Fit scaling
if len(results) >= 3:
    log_n = [math.log(r[0]) for r in results]
    log_d = [math.log(r[1]) for r in results]

    # Linear fit
    n = len(results)
    sx, sy = sum(log_n), sum(log_d)
    sxy = sum(x*y for x,y in zip(log_n, log_d))
    sxx = sum(x*x for x in log_n)

    slope = (n*sxy - sx*sy) / (n*sxx - sx*sx)
    inv_nu = -slope
    nu = 1/inv_nu if inv_nu != 0 else 999

    dev = abs(inv_nu - 1/PHI) / (1/PHI) * 100

    print(flush=True)
    print("=" * 60, flush=True)
    print("RESULT", flush=True)
    print("=" * 60, flush=True)
    print(f"Measured 1/ν = {inv_nu:.4f}", flush=True)
    print(f"Target   1/φ = {1/PHI:.4f}", flush=True)
    print(f"Deviation: {dev:.1f}%", flush=True)
    print(flush=True)

    if dev < 2: status = "STRONG"
    elif dev < 5: status = "Moderate"
    elif dev < 10: status = "Suggestive"
    else: status = "Weak"

    print(f"Status: {status}", flush=True)
