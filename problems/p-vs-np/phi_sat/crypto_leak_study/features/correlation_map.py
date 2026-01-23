#!/usr/bin/env python3
"""
Output→Input Correlation Map

Find which output bits correlate with which input bits.
This reveals the STRUCTURE of how information is encoded.

Then use that structure for transfer learning.
"""

import numpy as np
import hashlib
import time
from typing import Dict, Tuple
from dataclasses import dataclass


def generate_samples(num_samples: int, input_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate input/output pairs."""
    input_bytes = (input_bits + 7) // 8

    inputs = np.zeros((num_samples, input_bits), dtype=np.int8)
    outputs = np.zeros((num_samples, 256), dtype=np.int8)

    for i in range(num_samples):
        data = bytearray(np.random.bytes(input_bytes))
        if input_bits % 8 != 0:
            mask = (1 << (input_bits % 8)) - 1
            data[0] &= mask
        data = bytes(data)

        h = hashlib.sha256(data).digest()

        inp_bits = np.unpackbits(np.frombuffer(bytes(data[:input_bytes]), dtype=np.uint8))
        inputs[i, :min(len(inp_bits), input_bits)] = inp_bits[:input_bits].astype(np.int8)
        outputs[i] = np.unpackbits(np.frombuffer(h, dtype=np.uint8)).astype(np.int8)

        if (i + 1) % 10000 == 0:
            print(f"  {i+1:,}/{num_samples:,}")

    # Convert to -1/+1 for correlation
    inputs = 2 * inputs - 1
    outputs = 2 * outputs - 1

    return inputs, outputs


def compute_correlation_matrix(inputs: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """
    Compute correlation between each output bit and each input bit.

    Returns: (256, input_bits) matrix of correlations
    """
    # Correlation = E[X*Y] for centered variables (already -1/+1)
    # corr[i,j] = mean(output_i * input_j)

    num_samples = len(inputs)
    input_bits = inputs.shape[1]

    # Vectorized correlation
    # outputs: (N, 256), inputs: (N, input_bits)
    # correlation: (256, input_bits)

    correlation = (outputs.T @ inputs) / num_samples

    return correlation


def analyze_correlations(corr_matrix: np.ndarray, input_bits: int) -> Dict:
    """Analyze the correlation structure."""

    results = {}

    # For each input bit, find strongest output correlations
    results['input_to_output'] = []
    for inp_bit in range(input_bits):
        col = corr_matrix[:, inp_bit]
        top_idx = np.argsort(np.abs(col))[::-1][:10]
        results['input_to_output'].append({
            'input_bit': inp_bit,
            'top_output_bits': top_idx.tolist(),
            'top_correlations': col[top_idx].tolist(),
            'max_abs_corr': float(np.max(np.abs(col))),
        })

    # For each output bit, find which input bits it correlates with
    results['output_to_input'] = []
    for out_bit in range(256):
        row = corr_matrix[out_bit, :]
        top_idx = np.argsort(np.abs(row))[::-1][:5]
        results['output_to_input'].append({
            'output_bit': out_bit,
            'top_input_bits': top_idx.tolist(),
            'top_correlations': row[top_idx].tolist(),
            'max_abs_corr': float(np.max(np.abs(row))),
        })

    # Overall statistics
    results['stats'] = {
        'mean_abs_corr': float(np.mean(np.abs(corr_matrix))),
        'max_abs_corr': float(np.max(np.abs(corr_matrix))),
        'std_corr': float(np.std(corr_matrix)),
    }

    # Find output bits with strongest overall correlations
    output_strength = np.max(np.abs(corr_matrix), axis=1)
    top_outputs = np.argsort(output_strength)[::-1][:20]
    results['strongest_output_bits'] = [
        {'bit': int(b), 'max_corr': float(output_strength[b])}
        for b in top_outputs
    ]

    return results


def build_transfer_features(corr_matrix: np.ndarray, threshold: float = 0.05) -> Dict:
    """
    Build a feature map for transfer learning.

    For each input bit, identify which output bits are predictive.
    This structure should transfer across input sizes.
    """
    input_bits = corr_matrix.shape[1]

    transfer_map = {}

    for inp_bit in range(input_bits):
        col = corr_matrix[:, inp_bit]

        # Output bits with correlation above threshold
        predictive_outputs = np.where(np.abs(col) > threshold)[0]

        transfer_map[inp_bit] = {
            'predictive_output_bits': predictive_outputs.tolist(),
            'correlations': col[predictive_outputs].tolist(),
            'num_predictive': len(predictive_outputs),
        }

    return transfer_map


def test_transfer(
    source_bits: int,
    target_bits: int,
    source_samples: int,
    target_samples: int,
) -> Dict:
    """
    Test if correlation structure transfers across input sizes.

    1. Learn correlations at source_bits
    2. Apply to target_bits
    3. Measure if it helps prediction
    """
    print(f"\n{'='*60}")
    print(f"TRANSFER TEST: {source_bits} bits → {target_bits} bits")
    print(f"{'='*60}")

    # Generate source data
    print(f"\nGenerating {source_samples:,} source samples ({source_bits} bits)...")
    source_inputs, source_outputs = generate_samples(source_samples, source_bits)

    # Compute correlations on source
    print("Computing correlation matrix...")
    corr_matrix = compute_correlation_matrix(source_inputs, source_outputs)

    # Analyze
    analysis = analyze_correlations(corr_matrix, source_bits)
    print(f"Mean abs correlation: {analysis['stats']['mean_abs_corr']:.4f}")
    print(f"Max abs correlation: {analysis['stats']['max_abs_corr']:.4f}")

    # Build transfer features
    transfer_map = build_transfer_features(corr_matrix, threshold=0.03)

    # Generate target data
    print(f"\nGenerating {target_samples:,} target samples ({target_bits} bits)...")
    target_inputs, target_outputs = generate_samples(target_samples, target_bits)

    # Test: use correlation-weighted output bits to predict input bits
    print("\nTesting transfer...")

    results = {'bit_accuracies': []}

    # For each of the first source_bits input bits (which exist in both)
    test_bits = min(source_bits, target_bits)

    for inp_bit in range(test_bits):
        # Get predictive output bits from source
        pred_outputs = transfer_map[inp_bit]['predictive_output_bits']
        pred_corrs = np.array(transfer_map[inp_bit]['correlations'])

        if len(pred_outputs) == 0:
            results['bit_accuracies'].append(0.5)
            continue

        # Weighted vote from predictive output bits
        target_out_subset = target_outputs[:, pred_outputs]  # (N, num_predictive)

        # Prediction: sign of correlation-weighted sum
        weighted_sum = target_out_subset @ pred_corrs
        predictions = (weighted_sum > 0).astype(np.int8) * 2 - 1  # back to -1/+1

        # Accuracy
        actual = target_inputs[:, inp_bit]
        accuracy = np.mean(predictions == actual)

        results['bit_accuracies'].append(float(accuracy))

    results['mean_accuracy'] = np.mean(results['bit_accuracies'])
    results['leak_strength'] = results['mean_accuracy'] - 0.5

    print(f"\nTransfer results:")
    for i, acc in enumerate(results['bit_accuracies']):
        print(f"  Bit {i}: {acc:.3f}")

    print(f"\nMean accuracy: {results['mean_accuracy']:.4f}")
    print(f"Leak strength: {results['leak_strength']:+.4f}")
    print(f"(Random baseline: 0.5)")

    return {
        'source_bits': source_bits,
        'target_bits': target_bits,
        'correlation_analysis': analysis,
        'transfer_map': transfer_map,
        'transfer_results': results,
    }


def run_correlation_analysis(input_bits: int = 16, num_samples: int = 100000):
    """Run full correlation analysis."""

    print("="*60)
    print(f"CORRELATION ANALYSIS - {input_bits} BITS")
    print("="*60)

    print(f"\nGenerating {num_samples:,} samples...")
    inputs, outputs = generate_samples(num_samples, input_bits)

    print("Computing correlation matrix...")
    corr_matrix = compute_correlation_matrix(inputs, outputs)

    print("Analyzing structure...")
    analysis = analyze_correlations(corr_matrix, input_bits)

    print(f"\n{'='*60}")
    print("RESULTS")
    print("="*60)

    print(f"\nOverall statistics:")
    print(f"  Mean |correlation|: {analysis['stats']['mean_abs_corr']:.4f}")
    print(f"  Max |correlation|: {analysis['stats']['max_abs_corr']:.4f}")
    print(f"  Std correlation: {analysis['stats']['std_corr']:.4f}")

    print(f"\nStrongest output bits (most predictive):")
    for item in analysis['strongest_output_bits'][:10]:
        print(f"  Output bit {item['bit']}: max |corr| = {item['max_corr']:.4f}")

    print(f"\nPer-input-bit analysis:")
    for item in analysis['input_to_output'][:8]:
        inp = item['input_bit']
        top_outs = item['top_output_bits'][:3]
        top_corrs = item['top_correlations'][:3]
        print(f"  Input bit {inp}: top outputs {top_outs}, corrs {[f'{c:.3f}' for c in top_corrs]}")

    return corr_matrix, analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bits', type=int, default=16)
    parser.add_argument('--samples', type=int, default=100000)
    parser.add_argument('--transfer', action='store_true', help='Run transfer test')
    parser.add_argument('--target-bits', type=int, default=32)
    args = parser.parse_args()

    if args.transfer:
        test_transfer(
            source_bits=args.bits,
            target_bits=args.target_bits,
            source_samples=args.samples,
            target_samples=args.samples // 2,
        )
    else:
        run_correlation_analysis(args.bits, args.samples)
