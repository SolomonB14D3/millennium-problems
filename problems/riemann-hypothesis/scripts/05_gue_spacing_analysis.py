#!/usr/bin/env python3
"""
Improved GUE Spacing Ratio Analysis - Fixed padding & better GUE approx
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse

# Golden ratio constants
PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI                   # ≈ 0.6180339887
INV_PHI_SQ = (3 - np.sqrt(5)) / 2   # ≈ 0.3819660113


def load_zeros(filename):
    try:
        with open(filename, 'r') as f:
            zeros = [float(line.strip()) for line in f if line.strip() and not line.startswith('#')]
        print(f"Loaded {len(zeros)} zeta zeros")
        return np.sort(np.array(zeros))
    except Exception as e:
        print(f"Error loading file: {e}")
        exit(1)


def compute_local_unfolded_ratios(zeros, window=30):
    if len(zeros) < window + 2:
        print(f"Need at least {window + 2} zeros for window={window}")
        exit(1)

    deltas = np.diff(zeros)  # shape: N-1 where N = len(zeros)

    # Rolling mean (valid mode shortens by window-1)
    kernel = np.ones(window) / window
    local_means_valid = np.convolve(deltas, kernel, mode='valid')

    # Pad to match original deltas length (window-1 total, symmetric)
    pad_total = window - 1
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    # Use 'edge' mode for better boundary behavior
    local_means = np.pad(local_means_valid, (pad_left, pad_right), mode='edge')

    # If still off by 1 (rare rounding/odd window), trim or pad last element
    if len(local_means) < len(deltas):
        local_means = np.pad(local_means, (0, 1), mode='edge')
    elif len(local_means) > len(deltas):
        local_means = local_means[:len(deltas)]

    unfolded = deltas / local_means
    ratios = unfolded[1:] / unfolded[:-1]          # r_n = s_{n+1} / s_n
    folded = np.minimum(ratios, 1 / ratios)        # folded ∈ [0,1]

    return folded


def approximate_gue_folded_ratio_density(x):
    """Improved approx for folded GUE ratio density (peak ~1.5 at ~0.60)"""
    # Rough analytic-inspired form: quadratic repulsion + Gaussian-like peak
    repulsion = 32 * x**2 / np.pi**2   # low-r behavior
    peak = np.exp(- (x - 0.599)**2 / (2 * 0.18**2))
    density = 1.52 * repulsion * peak
    density /= np.trapz(density, x) / 1.0   # normalize over [0,1] approx
    return density


def plot_and_analyze(ratios, local_window=30, integ_window=0.03, nbins=150, savefig=True):
    fig, ax = plt.subplots(figsize=(12, 7))

    kde = gaussian_kde(ratios, bw_method=0.025)
    x = np.linspace(0, 1.2, 1500)
    density = kde(x)

    ax.plot(x, density, 'b-', lw=2.2, label='KDE (zeta zeros)')
    ax.hist(ratios, bins=nbins, density=True, alpha=0.25, color='gray', label='Histogram')

    # GUE approx overlay
    gue_approx = approximate_gue_folded_ratio_density(x)
    ax.plot(x, gue_approx, 'g--', lw=1.8, label='Approx. GUE folded ratio density')

    ax.axvline(INV_PHI, color='r', linestyle='--', lw=1.8, label=f'1/φ ≈ {INV_PHI:.6f}')
    ax.axvline(INV_PHI_SQ, color='orange', linestyle=':', lw=1.8, label=f'1/φ² ≈ {INV_PHI_SQ:.6f}')

    ax.set_xlabel('Folded spacing ratio r = min(s_{n+1}/s_n, s_n/s_{n+1})')
    ax.set_ylabel('Density')
    ax.set_title(f'Spacing Ratio Distribution (local window={local_window}, n={len(ratios)} ratios)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Integrated excess
    mask_phi = (x >= INV_PHI - integ_window) & (x <= INV_PHI + integ_window)
    integ_phi = np.trapz(density[mask_phi], x[mask_phi])
    integ_gue = np.trapz(gue_approx[mask_phi], x[mask_phi])
    integ_uniform = 2 * integ_window * 1.0

    excess_vs_uniform = integ_phi / integ_uniform if integ_uniform > 0 else 0
    excess_vs_gue = integ_phi / integ_gue if integ_gue > 0 else 0

    peak_idx = np.argmax(density)
    peak_val = x[peak_idx]

    print(f"\nResults (n={len(ratios)} ratios, local window={local_window}):")
    print(f"  KDE peak location: {peak_val:.6f}")
    print(f"  Integrated density near 1/φ (±{integ_window*100:.1f}%): {integ_phi:.4f}")
    print(f"  Approx GUE integrated in same window: {integ_gue:.4f}")
    print(f"  Excess vs uniform baseline: ~{excess_vs_uniform:.2f}x")
    print(f"  Excess vs approx GUE: ~{excess_vs_gue:.2f}x")

    if savefig:
        plt.savefig("spacing_ratios_phi_fixed.png", dpi=300, bbox_inches='tight')
        print("Figure saved as spacing_ratios_phi_fixed.png")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Improved zeta zero spacing ratio analysis (fixed)")
    parser.add_argument("filename", help="Text file with zeta zero imaginary parts")
    parser.add_argument("--window", type=int, default=30, help="Local mean window size")
    parser.add_argument("--integ", type=float, default=0.03, help="Integration window around 1/phi")
    parser.add_argument("--nbins", type=int, default=150, help="Histogram bins")
    args = parser.parse_args()

    zeros = load_zeros(args.filename)
    ratios = compute_local_unfolded_ratios(zeros, window=args.window)
    plot_and_analyze(ratios, local_window=args.window, integ_window=args.integ, nbins=args.nbins)


if __name__ == "__main__":
    main()
