#!/usr/bin/env python3
"""
Differential Analysis - Study the "Shaking" Process

Key insight: When you flip one input bit, the output changes.
But HOW it changes follows SHA256's structure - it's not random.

Study: Δinput → Δoutput
- Flip input bit i
- XOR the outputs
- The pattern of which bits flip reveals propagation structure
- Learn these patterns - they might transfer across scales
"""

import numpy as np
import hashlib
import time
from typing import Tuple, Dict, List
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score


def sha256_hash(data: bytes) -> np.ndarray:
    """Get SHA256 output as bit array."""
    h = hashlib.sha256(data).digest()
    return np.unpackbits(np.frombuffer(h, dtype=np.uint8))


def generate_differential_samples(
    num_samples: int,
    input_bits: int,
    bit_to_flip: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate differential samples for one input bit.

    Returns:
        base_outputs: outputs of base inputs
        delta_outputs: XOR of (base output, flipped output)
        base_inputs: the base input bits
    """
    input_bytes = (input_bits + 7) // 8
    byte_idx = bit_to_flip // 8
    bit_mask = 1 << (7 - (bit_to_flip % 8))

    base_inputs = np.zeros((num_samples, input_bits), dtype=np.uint8)
    base_outputs = np.zeros((num_samples, 256), dtype=np.uint8)
    delta_outputs = np.zeros((num_samples, 256), dtype=np.uint8)

    for i in range(num_samples):
        # Generate base input
        data = bytearray(np.random.bytes(input_bytes))
        if input_bits % 8 != 0:
            mask = (1 << (input_bits % 8)) - 1
            data[0] &= mask

        base_data = bytes(data)
        base_hash = sha256_hash(base_data)

        # Flip the target bit
        data[byte_idx] ^= bit_mask
        flipped_data = bytes(data)
        flipped_hash = sha256_hash(flipped_data)

        # Store
        inp_bits = np.unpackbits(np.frombuffer(base_data, dtype=np.uint8))
        base_inputs[i, :min(len(inp_bits), input_bits)] = inp_bits[:input_bits]
        base_outputs[i] = base_hash
        delta_outputs[i] = base_hash ^ flipped_hash  # XOR = which bits changed

    return base_outputs, delta_outputs, base_inputs


def analyze_propagation_patterns(input_bits: int, samples_per_bit: int = 10000):
    """
    Analyze how each input bit's flip propagates to output.

    For each input bit, compute:
    - Which output bits are MOST likely to flip
    - Which output bits are LEAST likely to flip
    - The overall pattern
    """

    print("="*70)
    print(f"PROPAGATION ANALYSIS - {input_bits} BITS")
    print("="*70)

    # Propagation probability matrix: P(output_j flips | input_i flips)
    prop_matrix = np.zeros((input_bits, 256), dtype=np.float32)

    for bit_idx in range(input_bits):
        print(f"\nAnalyzing input bit {bit_idx}...")
        _, delta_outputs, _ = generate_differential_samples(samples_per_bit, input_bits, bit_idx)

        # Probability each output bit flips
        flip_probs = np.mean(delta_outputs, axis=0)
        prop_matrix[bit_idx] = flip_probs

        # Should be ~0.5 for ideal avalanche
        print(f"  Mean flip prob: {np.mean(flip_probs):.4f} (ideal: 0.5)")
        print(f"  Std flip prob: {np.std(flip_probs):.4f} (ideal: ~0)")
        print(f"  Min: {np.min(flip_probs):.4f}, Max: {np.max(flip_probs):.4f}")

    # Analyze the matrix
    print(f"\n{'='*70}")
    print("PROPAGATION STRUCTURE")
    print("="*70)

    # Overall statistics
    print(f"\nOverall flip probability: {np.mean(prop_matrix):.4f}")
    print(f"Deviation from 0.5: {np.abs(np.mean(prop_matrix) - 0.5):.4f}")

    # Which output bits have most VARIANCE across input bits?
    # High variance = different input bits affect this output differently = STRUCTURE
    output_variance = np.var(prop_matrix, axis=0)
    high_var_outputs = np.argsort(output_variance)[::-1][:20]

    print(f"\nOutput bits with HIGHEST variance (most structure):")
    for out_bit in high_var_outputs[:10]:
        print(f"  Output bit {out_bit}: variance = {output_variance[out_bit]:.6f}")

    # Which input bits have most UNIQUE patterns?
    # Compare each input's pattern to average
    avg_pattern = np.mean(prop_matrix, axis=0)
    input_uniqueness = np.mean(np.abs(prop_matrix - avg_pattern), axis=1)

    print(f"\nInput bits with MOST UNIQUE propagation:")
    unique_inputs = np.argsort(input_uniqueness)[::-1][:10]
    for inp_bit in unique_inputs:
        print(f"  Input bit {inp_bit}: uniqueness = {input_uniqueness[inp_bit]:.6f}")

    return prop_matrix, output_variance, input_uniqueness


def learn_from_differentials(
    input_bits: int = 16,
    num_samples: int = 50000,
):
    """
    Use differential information to predict input bits.

    Instead of: output → input bit
    Try: (output, delta_output for each bit) → which bit was flipped

    This uses the PROPAGATION PATTERN as a fingerprint.
    """

    print("="*70)
    print("DIFFERENTIAL LEARNING")
    print("="*70)

    # Generate base data
    input_bytes = (input_bits + 7) // 8

    print(f"\nGenerating {num_samples} base samples...")

    base_inputs = []
    base_outputs = []

    for i in range(num_samples):
        data = bytearray(np.random.bytes(input_bytes))
        if input_bits % 8 != 0:
            mask = (1 << (input_bits % 8)) - 1
            data[0] &= mask

        base_data = bytes(data)
        base_hash = sha256_hash(base_data)

        inp_bits = np.unpackbits(np.frombuffer(base_data, dtype=np.uint8))
        base_inputs.append(inp_bits[:input_bits])
        base_outputs.append(base_hash)

        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{num_samples}")

    base_inputs = np.array(base_inputs, dtype=np.uint8)
    base_outputs = np.array(base_outputs, dtype=np.uint8)

    # For each sample, compute differential fingerprint
    # Flip each input bit and record which output bits change
    print(f"\nComputing differential fingerprints...")

    diff_features = np.zeros((num_samples, input_bits * 256), dtype=np.uint8)

    for i in range(num_samples):
        data = bytearray(np.random.bytes(input_bytes))
        # Reconstruct the input
        for b in range(input_bits):
            byte_idx = b // 8
            bit_idx = 7 - (b % 8)
            if base_inputs[i, b] == 1:
                data[byte_idx] |= (1 << bit_idx)
            else:
                data[byte_idx] &= ~(1 << bit_idx)

        base_data = bytes(data)

        for bit_to_flip in range(input_bits):
            flipped_data = bytearray(base_data)
            byte_idx = bit_to_flip // 8
            bit_mask = 1 << (7 - (bit_to_flip % 8))
            flipped_data[byte_idx] ^= bit_mask

            flipped_hash = sha256_hash(bytes(flipped_data))
            delta = base_outputs[i] ^ flipped_hash

            # Store in feature matrix
            start_idx = bit_to_flip * 256
            diff_features[i, start_idx:start_idx+256] = delta

        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{num_samples}")

    # Now try to predict input bits using both output AND differential fingerprint
    print(f"\nTraining with differential features...")
    print(f"Feature size: {diff_features.shape[1]} (diff) + 256 (output) = {diff_features.shape[1] + 256}")

    # Combine features
    combined_features = np.hstack([base_outputs, diff_features])

    # Split
    split = int(0.8 * num_samples)
    X_train = combined_features[:split]
    y_train = base_inputs[:split]
    X_test = combined_features[split:]
    y_test = base_inputs[split:]

    # Also test output-only baseline
    X_train_base = base_outputs[:split]
    X_test_base = base_outputs[split:]

    results = {
        'output_only': [],
        'with_differential': [],
    }

    start = time.time()
    for bit_idx in range(input_bits):
        # Output only
        model_base = HistGradientBoostingClassifier(max_iter=100, max_depth=8, random_state=42)
        model_base.fit(X_train_base, y_train[:, bit_idx])
        pred_base = model_base.predict(X_test_base)
        acc_base = accuracy_score(y_test[:, bit_idx], pred_base)

        # With differential
        model_diff = HistGradientBoostingClassifier(max_iter=100, max_depth=8, random_state=42)
        model_diff.fit(X_train, y_train[:, bit_idx])
        pred_diff = model_diff.predict(X_test)
        acc_diff = accuracy_score(y_test[:, bit_idx], pred_diff)

        results['output_only'].append(acc_base)
        results['with_differential'].append(acc_diff)

        improvement = acc_diff - acc_base
        print(f"  Bit {bit_idx}: base={acc_base:.3f}, diff={acc_diff:.3f}, Δ={improvement:+.3f} ({time.time()-start:.0f}s)")

    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    print(f"\nOutput-only mean: {np.mean(results['output_only']):.4f}")
    print(f"With differential mean: {np.mean(results['with_differential']):.4f}")
    print(f"Improvement: {np.mean(results['with_differential']) - np.mean(results['output_only']):+.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bits', type=int, default=16)
    parser.add_argument('--samples', type=int, default=20000)
    parser.add_argument('--analyze', action='store_true', help='Run propagation analysis')
    args = parser.parse_args()

    if args.analyze:
        analyze_propagation_patterns(args.bits, samples_per_bit=5000)
    else:
        learn_from_differentials(args.bits, args.samples)
