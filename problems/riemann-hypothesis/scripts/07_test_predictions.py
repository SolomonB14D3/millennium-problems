#!/usr/bin/env python3
"""
TEST THE PREDICTIONS: Analyze intermediate-height zeros from LMFDB.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI  # 0.6180339...
GUE_MEDIAN = 0.6050

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "figures"

# Predictions from fit
H_STAR = 5.04e5
BETA = 0.283

def predicted_median(h):
    """Model prediction."""
    peak = INV_PHI + 0.007
    return GUE_MEDIAN + (peak - GUE_MEDIAN) / (1 + (h / H_STAR) ** BETA)

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

def compute_spacing_ratio_median(zeros, window_size=500):
    """Compute median spacing ratio with local unfolding."""
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

    return np.median(ratios) if ratios else None

def main():
    print("=" * 70)
    print("TESTING RIEMANN φ-PREDICTIONS WITH LMFDB DATA")
    print("=" * 70)

    # Files and expected heights
    files = [
        ("lmfdb_n21m.txt", "N=21M"),
        ("lmfdb_n100m.txt", "N=100M"),
        ("lmfdb_n500m.txt", "N=500M"),
        ("lmfdb_n2b.txt", "N=2B"),
        ("lmfdb_n10b.txt", "N=10B"),
    ]

    results = []

    print("\nAnalyzing LMFDB files...")
    print("-" * 70)

    for filename, label in files:
        zeros = load_lmfdb_file(filename)
        if zeros is None:
            print(f"  {label}: File not found")
            continue

        height = np.mean(zeros)
        median = compute_spacing_ratio_median(zeros)

        if median is not None:
            pred = predicted_median(height)
            diff = median - pred
            diff_pct = diff / pred * 100

            results.append({
                'label': label,
                'height': height,
                'measured': median,
                'predicted': pred,
                'diff': diff,
                'diff_pct': diff_pct
            })

            print(f"  {label}: h = {height:.2e}, median = {median:.4f}, "
                  f"pred = {pred:.4f}, diff = {diff:+.4f} ({diff_pct:+.1f}%)")

    # Add original data points for comparison
    original_data = [
        ("zeros6 low", 3e3, 0.6249),
        ("zeros6 mid", 7.5e4, 0.6179),
        ("zeros6 high", 1e6, 0.6140),
        ("zeros3", 2.67e11, 0.6051),
    ]

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Dataset':>15} {'Height':>12} {'Measured':>10} {'Predicted':>10} {'Diff':>10} {'Status':>10}")
    print("-" * 70)

    all_results = []

    for label, h, meas in original_data:
        pred = predicted_median(h)
        diff = meas - pred
        status = "✓ PASS" if abs(diff) < 0.004 else "✗ FAIL"
        print(f"{label:>15} {h:>12.2e} {meas:>10.4f} {pred:>10.4f} {diff:>+10.4f} {status:>10}")
        all_results.append({'label': label, 'height': h, 'measured': meas, 'predicted': pred})

    for r in results:
        status = "✓ PASS" if abs(r['diff']) < 0.004 else "✗ FAIL"
        print(f"{r['label']:>15} {r['height']:>12.2e} {r['measured']:>10.4f} "
              f"{r['predicted']:>10.4f} {r['diff']:>+10.4f} {status:>10}")
        all_results.append(r)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Measured vs Predicted
    ax1 = axes[0]

    heights_orig = [3e3, 7.5e4, 1e6, 2.67e11]
    medians_orig = [0.6249, 0.6179, 0.6140, 0.6051]
    ax1.semilogx(heights_orig, medians_orig, 'ko', markersize=10, label='Original (zeros6, zeros3)', zorder=5)

    if results:
        heights_new = [r['height'] for r in results]
        medians_new = [r['measured'] for r in results]
        ax1.semilogx(heights_new, medians_new, 'g^', markersize=12, label='NEW: LMFDB intermediate', zorder=6)

    # Model curve
    h_plot = np.logspace(3, 12, 500)
    pred_plot = predicted_median(h_plot)
    ax1.semilogx(h_plot, pred_plot, 'r-', linewidth=2, label='Model prediction', zorder=3)

    ax1.axhline(y=INV_PHI, color='gold', linestyle='--', linewidth=2, label=f'1/φ = {INV_PHI:.4f}')
    ax1.axhline(y=GUE_MEDIAN, color='blue', linestyle='--', linewidth=2, label=f'GUE = {GUE_MEDIAN:.4f}')

    ax1.set_xlabel('Height h (log scale)', fontsize=12)
    ax1.set_ylabel('Median spacing ratio', fontsize=12)
    ax1.set_title('Transition Test: Model vs LMFDB Data', fontsize=14)
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1e2, 1e13)
    ax1.set_ylim(0.600, 0.630)

    # Plot 2: Residuals
    ax2 = axes[1]

    if all_results:
        heights_all = [r['height'] for r in all_results]
        residuals = [r['measured'] - r['predicted'] for r in all_results]

        colors = ['black'] * len(original_data) + ['green'] * len(results)
        markers = ['o'] * len(original_data) + ['^'] * len(results)

        for h, res, c, m in zip(heights_all, residuals, colors, markers):
            ax2.semilogx(h, res * 1000, marker=m, color=c, markersize=10)

    ax2.axhline(y=0, color='red', linestyle='-', linewidth=2)
    ax2.fill_between([1e2, 1e13], [-4, -4], [4, 4], alpha=0.2, color='green', label='±0.004 tolerance')
    ax2.set_xlabel('Height h (log scale)', fontsize=12)
    ax2.set_ylabel('Residual (measured - predicted) × 1000', fontsize=12)
    ax2.set_title('Model Residuals', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1e2, 1e13)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'prediction_test.png', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {FIG_DIR / 'prediction_test.png'}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if results:
        max_diff = max(abs(r['diff']) for r in results)
        avg_diff = np.mean([abs(r['diff']) for r in results])

        if max_diff < 0.004:
            print(f"""
✓ PREDICTIONS CONFIRMED

All intermediate-height measurements fall within ±0.004 of predictions.

Maximum deviation: {max_diff:.4f}
Average deviation: {avg_diff:.4f}

CONCLUSION: The 1/φ → GUE transition model is validated.
The golden ratio genuinely appears in finite-size scaling of zeta zeros.
""")
        else:
            print(f"""
✗ PREDICTIONS PARTIALLY FAILED

Some measurements deviate > 0.004 from predictions.

Maximum deviation: {max_diff:.4f}
Average deviation: {avg_diff:.4f}

CONCLUSION: Model may need refinement, or transition is different than fitted.
""")
    else:
        print("\nNo new data to test. Download more LMFDB files first.")

if __name__ == "__main__":
    main()
