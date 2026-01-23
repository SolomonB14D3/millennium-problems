#!/usr/bin/env python3
"""
Probing-Based Transfer Learning

Key insight: Direct model transfer failed (0% efficiency).
But we can USE the model to PROBE the system - ask it questions
and use the RESPONSE PATTERNS to learn about larger scales.

The probe reveals STRUCTURAL information that transfers,
even if the exact model weights don't.

Approach:
1. Train model on small scale (16 bits)
2. Use it to PROBE: for each target input, ask "what does the
   small-scale model think about different input hypotheses?"
3. The pattern of responses IS the transferable structure
"""

import numpy as np
import hashlib
import time
import multiprocessing as mp
from typing import Tuple, Dict, List
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

NUM_CORES = mp.cpu_count()


def sha256_bits(data: bytes) -> np.ndarray:
    """Get SHA256 output as bit array."""
    h = hashlib.sha256(data).digest()
    return np.unpackbits(np.frombuffer(h, dtype=np.uint8))


def generate_samples(num_samples: int, input_bits: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate input/output pairs."""
    if seed is not None:
        np.random.seed(seed)

    input_bytes = (input_bits + 7) // 8

    inputs = np.zeros((num_samples, input_bits), dtype=np.uint8)
    outputs = np.zeros((num_samples, 256), dtype=np.uint8)

    for i in range(num_samples):
        data = bytearray(np.random.bytes(input_bytes))
        if input_bits % 8 != 0:
            mask = (1 << (input_bits % 8)) - 1
            data[0] &= mask
        data = bytes(data)

        h = hashlib.sha256(data).digest()

        inp_bits = np.unpackbits(np.frombuffer(bytes(data[:input_bytes]), dtype=np.uint8))
        inputs[i, :min(len(inp_bits), input_bits)] = inp_bits[:input_bits]
        outputs[i] = np.unpackbits(np.frombuffer(h, dtype=np.uint8))

    return inputs, outputs


def train_probe_models(source_bits: int, num_samples: int) -> List:
    """Train models on source scale to use as probes."""
    print(f"\n[1] Training probe models on {source_bits} bits...")

    inputs, outputs = generate_samples(num_samples, source_bits)

    split = int(0.8 * num_samples)
    X_train = outputs[:split]
    y_train = inputs[:split]
    X_val = outputs[split:]
    y_val = inputs[split:]

    models = []
    accuracies = []

    start = time.time()
    for bit_idx in range(source_bits):
        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X_train, y_train[:, bit_idx])

        pred = model.predict(X_val)
        acc = accuracy_score(y_val[:, bit_idx], pred)
        accuracies.append(acc)
        models.append(model)

        if (bit_idx + 1) % 4 == 0:
            print(f"  Trained {bit_idx + 1}/{source_bits} models ({time.time()-start:.0f}s)")

    print(f"  Source accuracy: {np.mean(accuracies):.3f}")
    return models, np.mean(accuracies)


def probe_system(models: List, target_output: np.ndarray) -> np.ndarray:
    """
    Use probe models to query the system.

    For each model, get:
    - Predicted class (0 or 1)
    - Prediction probability (confidence)

    This creates a "probe response" fingerprint.
    """
    num_models = len(models)

    # Get predictions and probabilities
    predictions = np.zeros(num_models, dtype=np.float32)

    for i, model in enumerate(models):
        # predict_proba gives [P(0), P(1)]
        proba = model.predict_proba(target_output.reshape(1, -1))[0]
        predictions[i] = proba[1]  # P(bit=1)

    return predictions


def build_probe_features(
    models: List,
    target_outputs: np.ndarray,
) -> np.ndarray:
    """
    Build probe-based features for target data.

    For each target sample, the feature is the response pattern
    from all probe models.
    """
    num_samples = len(target_outputs)
    num_probes = len(models)

    features = np.zeros((num_samples, num_probes), dtype=np.float32)

    print(f"  Probing {num_samples:,} samples...")
    start = time.time()

    for i in range(num_samples):
        features[i] = probe_system(models, target_outputs[i])

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"    {i+1:,}/{num_samples:,} ({rate:.0f}/s)")

    return features


def probing_transfer_experiment(
    source_bits: int = 16,
    target_bits: int = 32,
    source_samples: int = 100000,
    target_samples: int = 30000,
):
    """
    Full probing transfer experiment.

    1. Train probes on source scale
    2. Generate target data
    3. Build probe features for target
    4. Train new model using probe features + raw output
    5. Compare to baseline (output only)
    """

    print("="*70)
    print(f"PROBING TRANSFER: {source_bits} bits → {target_bits} bits")
    print("="*70)

    # Step 1: Train probe models
    probe_models, source_acc = train_probe_models(source_bits, source_samples)

    # Step 2: Generate target data
    print(f"\n[2] Generating {target_samples:,} target samples ({target_bits} bits)...")
    target_inputs, target_outputs = generate_samples(target_samples, target_bits)

    # Step 3: Build probe features
    print(f"\n[3] Building probe features...")
    probe_features = build_probe_features(probe_models, target_outputs)

    # Step 4: Combine features
    # Option A: probe features only
    # Option B: probe features + raw output
    # Option C: raw output only (baseline)

    combined_features = np.hstack([target_outputs, probe_features])

    print(f"\n  Feature dimensions:")
    print(f"    Raw output: {target_outputs.shape[1]}")
    print(f"    Probe features: {probe_features.shape[1]}")
    print(f"    Combined: {combined_features.shape[1]}")

    # Split
    split = int(0.8 * target_samples)
    X_train_combined = combined_features[:split]
    X_train_output = target_outputs[:split]
    X_train_probe = probe_features[:split]
    y_train = target_inputs[:split]

    X_test_combined = combined_features[split:]
    X_test_output = target_outputs[split:]
    X_test_probe = probe_features[split:]
    y_test = target_inputs[split:]

    # Step 5: Train and evaluate
    print(f"\n[4] Training on target scale...")

    results = {
        'output_only': [],
        'probe_only': [],
        'combined': [],
    }

    start = time.time()

    # Test on first source_bits (which we have probes for)
    test_bits = min(source_bits, target_bits)

    for bit_idx in range(test_bits):
        y_tr = y_train[:, bit_idx]
        y_te = y_test[:, bit_idx]

        # Baseline: output only
        model_out = HistGradientBoostingClassifier(max_iter=100, max_depth=8, random_state=42)
        model_out.fit(X_train_output, y_tr)
        acc_out = accuracy_score(y_te, model_out.predict(X_test_output))

        # Probe only
        model_probe = HistGradientBoostingClassifier(max_iter=100, max_depth=8, random_state=42)
        model_probe.fit(X_train_probe, y_tr)
        acc_probe = accuracy_score(y_te, model_probe.predict(X_test_probe))

        # Combined
        model_comb = HistGradientBoostingClassifier(max_iter=100, max_depth=8, random_state=42)
        model_comb.fit(X_train_combined, y_tr)
        acc_comb = accuracy_score(y_te, model_comb.predict(X_test_combined))

        results['output_only'].append(acc_out)
        results['probe_only'].append(acc_probe)
        results['combined'].append(acc_comb)

        improvement = acc_comb - acc_out
        print(f"  Bit {bit_idx}: output={acc_out:.3f}, probe={acc_probe:.3f}, combined={acc_comb:.3f} (Δ={improvement:+.3f})")

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)

    mean_out = np.mean(results['output_only'])
    mean_probe = np.mean(results['probe_only'])
    mean_comb = np.mean(results['combined'])

    print(f"\nSource ({source_bits} bits) accuracy: {source_acc:.4f}")
    print(f"\nTarget ({target_bits} bits):")
    print(f"  Output only:     {mean_out:.4f} (leak: {mean_out - 0.5:+.4f})")
    print(f"  Probe only:      {mean_probe:.4f} (leak: {mean_probe - 0.5:+.4f})")
    print(f"  Combined:        {mean_comb:.4f} (leak: {mean_comb - 0.5:+.4f})")

    improvement = mean_comb - mean_out
    print(f"\nProbe contribution: {improvement:+.4f}")

    if improvement > 0.01:
        print("→ PROBING HELPS! Structure transfers through interaction.")
    elif improvement > 0:
        print("→ Small improvement. Some structure may transfer.")
    else:
        print("→ No improvement. Need different probing strategy.")

    # Interesting: does probe_only work at all?
    probe_leak = mean_probe - 0.5
    if probe_leak > 0.01:
        print(f"\n★ PROBE-ONLY has {probe_leak:.3f} leak!")
        print("  This means source-scale structure IS visible at target scale!")

    return results


def differential_probing(
    source_bits: int = 16,
    target_bits: int = 32,
    source_samples: int = 100000,
    target_samples: int = 20000,
):
    """
    Differential probing: probe how the system responds to bit flips.

    For target input X:
    1. Hash X → output_0
    2. For each bit i, flip bit i: X_i → output_i
    3. Probe all outputs with source models
    4. The PATTERN of how probes change = structural fingerprint
    """

    print("="*70)
    print(f"DIFFERENTIAL PROBING: {source_bits} → {target_bits} bits")
    print("="*70)

    # Train probes
    probe_models, source_acc = train_probe_models(source_bits, source_samples)

    # Generate target data
    print(f"\n[2] Generating {target_samples:,} target samples...")
    target_inputs, target_outputs = generate_samples(target_samples, target_bits)

    # Build differential features
    print(f"\n[3] Building differential probe features...")
    print(f"  For each sample: probe base + probe {source_bits} flipped variants")

    # Feature: base probe + delta for each of first source_bits flips
    num_features = source_bits + source_bits * source_bits  # base + (num_flips × num_probes)
    diff_features = np.zeros((target_samples, num_features), dtype=np.float32)

    input_bytes = (target_bits + 7) // 8

    start = time.time()
    for i in range(target_samples):
        # Base probe
        base_probe = probe_system(probe_models, target_outputs[i])
        diff_features[i, :source_bits] = base_probe

        # Reconstruct input bytes
        data = bytearray(input_bytes)
        for b in range(target_bits):
            byte_idx = b // 8
            bit_idx = 7 - (b % 8)
            if target_inputs[i, b] == 1:
                data[byte_idx] |= (1 << bit_idx)

        base_data = bytes(data)

        # Flip each of first source_bits and probe
        for flip_bit in range(source_bits):
            flipped = bytearray(base_data)
            byte_idx = flip_bit // 8
            bit_mask = 1 << (7 - (flip_bit % 8))
            flipped[byte_idx] ^= bit_mask

            flipped_output = sha256_bits(bytes(flipped))
            flipped_probe = probe_system(probe_models, flipped_output)

            # Store delta (how probe changed)
            delta = flipped_probe - base_probe
            start_idx = source_bits + flip_bit * source_bits
            diff_features[i, start_idx:start_idx + source_bits] = delta

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - start
            print(f"    {i+1:,}/{target_samples:,} ({elapsed:.0f}s)")

    # Train with differential features
    print(f"\n[4] Training with differential probe features...")

    split = int(0.8 * target_samples)
    X_train = diff_features[:split]
    y_train = target_inputs[:split]
    X_test = diff_features[split:]
    y_test = target_inputs[split:]

    # Also test output-only baseline
    X_train_out = target_outputs[:split]
    X_test_out = target_outputs[split:]

    results = {'output_only': [], 'diff_probe': []}

    test_bits = min(source_bits, target_bits)

    for bit_idx in range(test_bits):
        # Baseline
        model_out = HistGradientBoostingClassifier(max_iter=100, max_depth=8, random_state=42)
        model_out.fit(X_train_out, y_train[:, bit_idx])
        acc_out = accuracy_score(y_test[:, bit_idx], model_out.predict(X_test_out))

        # Differential probe
        model_diff = HistGradientBoostingClassifier(max_iter=100, max_depth=8, random_state=42)
        model_diff.fit(X_train, y_train[:, bit_idx])
        acc_diff = accuracy_score(y_test[:, bit_idx], model_diff.predict(X_test))

        results['output_only'].append(acc_out)
        results['diff_probe'].append(acc_diff)

        print(f"  Bit {bit_idx}: output={acc_out:.3f}, diff_probe={acc_diff:.3f} (Δ={acc_diff-acc_out:+.3f})")

    print(f"\n{'='*70}")
    print("DIFFERENTIAL PROBING RESULTS")
    print("="*70)

    mean_out = np.mean(results['output_only'])
    mean_diff = np.mean(results['diff_probe'])

    print(f"\nOutput only: {mean_out:.4f}")
    print(f"Diff probe:  {mean_diff:.4f}")
    print(f"Improvement: {mean_diff - mean_out:+.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=int, default=16)
    parser.add_argument('--target', type=int, default=32)
    parser.add_argument('--samples', type=int, default=100000)
    parser.add_argument('--diff', action='store_true', help='Use differential probing')
    args = parser.parse_args()

    if args.diff:
        differential_probing(
            source_bits=args.source,
            target_bits=args.target,
            source_samples=args.samples,
            target_samples=args.samples // 4,
        )
    else:
        probing_transfer_experiment(
            source_bits=args.source,
            target_bits=args.target,
            source_samples=args.samples,
            target_samples=args.samples // 3,
        )
