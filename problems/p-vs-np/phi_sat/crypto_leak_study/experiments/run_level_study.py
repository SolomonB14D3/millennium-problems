#!/usr/bin/env python3
"""
Main Experiment: Level-by-Level Cryptographic Leak Study

DAT Framework Approach:
1. Understand variables first
2. Find which variables leak
3. Build up level by level
4. Look for φ-structure at boundaries

This script orchestrates the full study:
- SHA256 alone
- ECDSA alone
- Full Bitcoin pipeline
- Cross-level pattern analysis
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.level_trainer import LevelTrainer, LeakReport
from features.bit_effects import compute_bit_effect_matrix, find_phi_structure
from primitives.sha256_study import sha256_instrumented, verify_implementation as verify_sha256
from primitives.ecdsa_study import verify_implementation as verify_ecdsa


def run_sha256_study(levels: list, samples: int, output_dir: str) -> LeakReport:
    """Run level-by-level SHA256 study."""
    print("\n" + "="*70)
    print("PHASE 1: SHA256 STUDY")
    print("="*70)

    # Verify implementation first
    if not verify_sha256():
        raise RuntimeError("SHA256 implementation verification failed")

    trainer = LevelTrainer(target='sha256', output_dir=os.path.join(output_dir, 'sha256'))
    return trainer.train_all_levels(levels=levels, samples_per_level=samples)


def run_ecdsa_study(levels: list, samples: int, output_dir: str) -> LeakReport:
    """Run level-by-level ECDSA study."""
    print("\n" + "="*70)
    print("PHASE 2: ECDSA STUDY")
    print("="*70)

    # Verify implementation first
    if not verify_ecdsa():
        raise RuntimeError("ECDSA implementation verification failed")

    trainer = LevelTrainer(target='ecdsa', output_dir=os.path.join(output_dir, 'ecdsa'))
    return trainer.train_all_levels(levels=levels, samples_per_level=samples)


def run_bit_effect_study(output_dir: str):
    """Detailed bit effect analysis."""
    import hashlib

    print("\n" + "="*70)
    print("PHASE 3: BIT EFFECT ANALYSIS")
    print("="*70)

    def sha256_fn(data: bytes) -> bytes:
        return hashlib.sha256(data).digest()

    results = {}

    for input_bits in [8, 16, 32, 64]:
        print(f"\n--- {input_bits}-bit input ---")
        bem = compute_bit_effect_matrix(sha256_fn, input_bits, 256, samples_per_bit=500)

        max_dev, i, j = bem.max_deviation()
        phi_result = find_phi_structure(bem.matrix)

        results[input_bits] = {
            'max_deviation': float(max_dev),
            'max_deviation_position': (int(i), int(j)),
            'leak_score': float(bem.leak_score()),
            'phi_structure': phi_result,
        }

        print(f"  Max deviation: {max_dev:.4f}")
        print(f"  Leak score: {bem.leak_score():.4f}")
        print(f"  φ-structure: {phi_result['phi_structure']}")

    # Save results
    with open(os.path.join(output_dir, 'bit_effect_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def cross_level_analysis(sha256_report: LeakReport, ecdsa_report: LeakReport) -> dict:
    """
    Analyze patterns across levels and primitives.

    DAT perspective: Look for universal scaling laws.
    """
    print("\n" + "="*70)
    print("CROSS-LEVEL ANALYSIS")
    print("="*70)

    PHI = 1.618033988749895

    analysis = {
        'sha256': {},
        'ecdsa': {},
        'cross_primitive': {},
    }

    # SHA256 scaling
    sha_levels = [r.level for r in sha256_report.results]
    sha_leaks = [r.leak_strength for r in sha256_report.results]

    print("\nSHA256 leak strength by level:")
    for level, leak in zip(sha_levels, sha_leaks):
        print(f"  {level:3d} bits: {leak:+.4f}")

    # ECDSA scaling
    ecdsa_levels = [r.level for r in ecdsa_report.results]
    ecdsa_leaks = [r.leak_strength for r in ecdsa_report.results]

    print("\nECDSA leak strength by level:")
    for level, leak in zip(ecdsa_levels, ecdsa_leaks):
        print(f"  {level:3d} bits: {leak:+.4f}")

    # Check for φ-scaling in leak decay
    def check_phi_scaling(levels, values):
        """Check if values scale by φ when levels double."""
        ratios = []
        for i in range(len(levels) - 1):
            if levels[i+1] == 2 * levels[i] and values[i] > 0.001:
                ratio = values[i] / values[i+1]
                ratios.append(ratio)
        return ratios

    sha_ratios = check_phi_scaling(sha_levels, sha_leaks)
    ecdsa_ratios = check_phi_scaling(ecdsa_levels, ecdsa_leaks)

    print("\nLeak decay ratios (level doubling):")
    if sha_ratios:
        print(f"  SHA256: {sha_ratios} (φ = {PHI:.3f})")
        phi_matches = sum(1 for r in sha_ratios if abs(r - PHI) < 0.5)
        print(f"    φ-like ratios: {phi_matches}/{len(sha_ratios)}")

    if ecdsa_ratios:
        print(f"  ECDSA: {ecdsa_ratios} (φ = {PHI:.3f})")
        phi_matches = sum(1 for r in ecdsa_ratios if abs(r - PHI) < 0.5)
        print(f"    φ-like ratios: {phi_matches}/{len(ecdsa_ratios)}")

    # Cross-primitive comparison
    common_levels = set(sha_levels) & set(ecdsa_levels)
    print(f"\nCommon levels: {sorted(common_levels)}")

    for level in sorted(common_levels):
        sha_leak = next(r.leak_strength for r in sha256_report.results if r.level == level)
        ecdsa_leak = next(r.leak_strength for r in ecdsa_report.results if r.level == level)
        ratio = sha_leak / ecdsa_leak if ecdsa_leak > 0.001 else float('inf')
        print(f"  Level {level}: SHA256={sha_leak:+.4f}, ECDSA={ecdsa_leak:+.4f}, ratio={ratio:.2f}")

    analysis['sha256']['levels'] = sha_levels
    analysis['sha256']['leak_strengths'] = sha_leaks
    analysis['sha256']['decay_ratios'] = sha_ratios
    analysis['ecdsa']['levels'] = ecdsa_levels
    analysis['ecdsa']['leak_strengths'] = ecdsa_leaks
    analysis['ecdsa']['decay_ratios'] = ecdsa_ratios

    return analysis


def main():
    parser = argparse.ArgumentParser(description='Level-by-level crypto leak study')
    parser.add_argument('--levels', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32],
                        help='Bit lengths to test')
    parser.add_argument('--samples', type=int, default=2000,
                        help='Samples per level')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--skip-sha256', action='store_true',
                        help='Skip SHA256 study')
    parser.add_argument('--skip-ecdsa', action='store_true',
                        help='Skip ECDSA study')
    parser.add_argument('--skip-bit-effects', action='store_true',
                        help='Skip bit effect analysis')

    args = parser.parse_args()

    # Setup output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', f'study_{timestamp}'
        )
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("CRYPTOGRAPHIC LEAK STUDY")
    print("DAT Framework Level-by-Level Analysis")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Levels to test: {args.levels}")
    print(f"Samples per level: {args.samples}")

    sha256_report = None
    ecdsa_report = None

    # Run studies
    if not args.skip_sha256:
        sha256_report = run_sha256_study(args.levels, args.samples, output_dir)

    if not args.skip_ecdsa:
        # ECDSA is slower, use fewer samples for higher levels
        ecdsa_samples = min(args.samples, 1000)
        ecdsa_report = run_ecdsa_study(args.levels, ecdsa_samples, output_dir)

    if not args.skip_bit_effects:
        run_bit_effect_study(output_dir)

    # Cross-level analysis
    if sha256_report and ecdsa_report:
        cross_analysis = cross_level_analysis(sha256_report, ecdsa_report)

        with open(os.path.join(output_dir, 'cross_analysis.json'), 'w') as f:
            json.dump(cross_analysis, f, indent=2, default=str)

    print("\n" + "="*70)
    print("STUDY COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")

    print("\nKEY INSIGHTS:")
    print("-" * 50)

    if sha256_report:
        print(f"SHA256 strongest leak: {sha256_report.strongest_leak_strength:+.4f} at {sha256_report.strongest_leak_level} bits")

    if ecdsa_report:
        print(f"ECDSA strongest leak: {ecdsa_report.strongest_leak_strength:+.4f} at {ecdsa_report.strongest_leak_level} bits")

    print("\nNEXT STEPS:")
    print("1. If leaks detected, investigate feature importance")
    print("2. Train deeper models on promising levels")
    print("3. Look for φ-structure in leak patterns")
    print("4. Test on full Bitcoin pipeline")


if __name__ == "__main__":
    main()
