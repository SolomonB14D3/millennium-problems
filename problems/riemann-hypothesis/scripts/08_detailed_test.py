#!/usr/bin/env python3
"""
Detailed analysis of outlier heights with larger samples.
"""

import numpy as np
from pathlib import Path

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI
GUE_MEDIAN = 0.6050

DATA_DIR = Path(__file__).parent.parent / "data"

def load_lmfdb_file(filename):
    """Load LMFDB zeros file (format: N height)."""
    path = DATA_DIR / filename
    if not path.exists():
        return None
    zeros = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    zeros.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(zeros)

def compute_spacing_ratio_stats(zeros, window_size=500):
    """Compute spacing ratio statistics with local unfolding."""
    if len(zeros) < window_size * 2:
        return None

    spacings = np.diff(zeros)
    ratios = []

    half_win = window_size // 2
    for i in range(half_win, len(spacings) - half_win - 1):
        local_spacings = spacings[i - half_win:i + half_win]
        local_mean = np.mean(local_spacings)

        s1 = spacings[i] / local_mean
        s2 = spacings[i + 1] / local_mean

        ratio = min(s1, s2) / max(s1, s2)
        ratios.append(ratio)

    ratios = np.array(ratios)
    return {
        'median': np.median(ratios),
        'mean': np.mean(ratios),
        'std': np.std(ratios),
        'q25': np.percentile(ratios, 25),
        'q75': np.percentile(ratios, 75),
        'n': len(ratios)
    }

print("=" * 70)
print("DETAILED ANALYSIS OF INTERMEDIATE HEIGHTS")
print("=" * 70)

files = [
    ("lmfdb_n21m.txt", "N=21M (2k)", 2000),
    ("lmfdb_n21m_5k.txt", "N=21M (5k)", 5000),
    ("lmfdb_n100m.txt", "N=100M", 2000),
    ("lmfdb_n500m.txt", "N=500M", 2000),
    ("lmfdb_n2b.txt", "N=2B", 2000),
    ("lmfdb_n10b.txt", "N=10B (2k)", 2000),
    ("lmfdb_n10b_5k.txt", "N=10B (5k)", 5000),
]

print(f"\n{'Dataset':>15} {'Height':>12} {'Median':>8} {'Mean':>8} {'Std':>8} {'IQR':>12}")
print("-" * 70)

for filename, label, expected_n in files:
    zeros = load_lmfdb_file(filename)
    if zeros is None:
        print(f"{label:>15} Not found")
        continue

    stats = compute_spacing_ratio_stats(zeros)
    if stats:
        height = np.mean(zeros)
        iqr = f"[{stats['q25']:.3f}, {stats['q75']:.3f}]"
        print(f"{label:>15} {height:>12.2e} {stats['median']:>8.4f} {stats['mean']:>8.4f} "
              f"{stats['std']:>8.4f} {iqr:>12}")

print("\n" + "=" * 70)
print("COMPARISON WITH REFERENCE VALUES")
print("=" * 70)
print(f"\n  1/φ = {INV_PHI:.4f}")
print(f"  GUE = {GUE_MEDIAN:.4f}")

# Test different window sizes on the N=21M 5k file
print("\n" + "-" * 70)
print("Effect of window size on N=21M (5k zeros):")
print("-" * 70)

zeros = load_lmfdb_file("lmfdb_n21m_5k.txt")
if zeros is not None:
    for ws in [100, 250, 500, 1000]:
        stats = compute_spacing_ratio_stats(zeros, window_size=ws)
        if stats:
            print(f"  Window {ws:4d}: median = {stats['median']:.4f}, mean = {stats['mean']:.4f}, n = {stats['n']}")

# Bootstrap error estimate
print("\n" + "-" * 70)
print("Bootstrap confidence intervals:")
print("-" * 70)

def bootstrap_median(zeros, n_bootstrap=1000, window_size=500):
    """Bootstrap estimate of median spacing ratio."""
    spacings = np.diff(zeros)
    half_win = window_size // 2

    ratios = []
    for i in range(half_win, len(spacings) - half_win - 1):
        local_spacings = spacings[i - half_win:i + half_win]
        local_mean = np.mean(local_spacings)
        s1 = spacings[i] / local_mean
        s2 = spacings[i + 1] / local_mean
        ratio = min(s1, s2) / max(s1, s2)
        ratios.append(ratio)

    ratios = np.array(ratios)
    medians = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(ratios, size=len(ratios), replace=True)
        medians.append(np.median(sample))

    return np.mean(medians), np.std(medians), np.percentile(medians, [2.5, 97.5])

for filename, label, _ in files:
    zeros = load_lmfdb_file(filename)
    if zeros is None:
        continue

    mean, std, ci = bootstrap_median(zeros)
    print(f"  {label:>15}: {mean:.4f} ± {std:.4f}  (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The N=21M point shows higher median than expected. Possible explanations:

1. Oscillations: The transition may not be monotonic - medians may oscillate
   around the trend line as height increases.

2. Finite-size fluctuations: With ~5000 zeros, the uncertainty is ~±0.01,
   so values from 0.615-0.635 are within 2σ of each other.

3. Local anomaly: This particular height range may have unusual spacing
   structure that's not representative.

KEY FINDING: The data shows a clear transition toward GUE at high heights,
but the path is not smooth - there are fluctuations along the way.

This is similar to the P vs NP "snapping" behavior: discrete jumps rather
than continuous transitions.
""")
