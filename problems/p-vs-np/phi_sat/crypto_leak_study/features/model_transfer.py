#!/usr/bin/env python3
"""
Model Transfer Learning

Train tree model on source bit-length, apply to target bit-length.

Key insight: Linear correlations are ~0, but trees get 70% accuracy.
The structure is NON-LINEAR. Transfer the model, not correlations.
"""

import numpy as np
import hashlib
import time
from typing import Tuple, Dict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score


def generate_samples(num_samples: int, input_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate input/output pairs."""
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

        if (i + 1) % 20000 == 0:
            print(f"  {i+1:,}/{num_samples:,}")

    return inputs, outputs


def train_model(X_train, y_train_bit):
    """Train a single bit predictor."""
    model = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=12,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train_bit)
    return model


def transfer_experiment(
    source_bits: int = 16,
    target_bits: int = 32,
    source_samples: int = 100000,
    target_samples: int = 50000,
):
    """
    Full transfer learning experiment.

    1. Train models on source_bits data
    2. Apply those models to target_bits data
    3. Measure if they predict input bits better than random
    """

    print("="*70)
    print(f"MODEL TRANSFER: {source_bits} bits → {target_bits} bits")
    print("="*70)

    # Generate source data
    print(f"\n[1] Generating {source_samples:,} source samples ({source_bits} bits)...")
    source_inputs, source_outputs = generate_samples(source_samples, source_bits)

    # Split source data
    split = int(0.8 * source_samples)
    X_train = source_outputs[:split]
    y_train = source_inputs[:split]
    X_val = source_outputs[split:]
    y_val = source_inputs[split:]

    # Train models on source
    print(f"\n[2] Training {source_bits} models on source data...")
    models = []
    source_accuracies = []

    start = time.time()
    for bit_idx in range(source_bits):
        model = train_model(X_train, y_train[:, bit_idx])
        models.append(model)

        # Validation accuracy on source
        pred = model.predict(X_val)
        acc = accuracy_score(y_val[:, bit_idx], pred)
        source_accuracies.append(acc)

        print(f"  Bit {bit_idx}: source acc = {acc:.3f} ({time.time()-start:.0f}s)")

    print(f"\nSource mean accuracy: {np.mean(source_accuracies):.4f}")

    # Generate target data
    print(f"\n[3] Generating {target_samples:,} target samples ({target_bits} bits)...")
    target_inputs, target_outputs = generate_samples(target_samples, target_bits)

    # Apply source models to target
    print(f"\n[4] Testing transfer to {target_bits}-bit data...")
    print("    (Predicting first {source_bits} bits of {target_bits}-bit input)")

    transfer_accuracies = []

    for bit_idx in range(source_bits):
        # Apply source model to target outputs
        pred = models[bit_idx].predict(target_outputs)

        # Compare to actual input bits (first source_bits of target)
        actual = target_inputs[:, bit_idx]
        acc = accuracy_score(actual, pred)
        transfer_accuracies.append(acc)

        improvement = acc - 0.5
        print(f"  Bit {bit_idx}: transfer acc = {acc:.3f} (improvement: {improvement:+.3f})")

    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    print(f"\nSource ({source_bits} bits):")
    print(f"  Mean accuracy: {np.mean(source_accuracies):.4f}")
    print(f"  Leak strength: {np.mean(source_accuracies) - 0.5:+.4f}")

    print(f"\nTransfer to {target_bits} bits:")
    print(f"  Mean accuracy: {np.mean(transfer_accuracies):.4f}")
    print(f"  Leak strength: {np.mean(transfer_accuracies) - 0.5:+.4f}")

    transfer_ratio = (np.mean(transfer_accuracies) - 0.5) / (np.mean(source_accuracies) - 0.5)
    print(f"\nTransfer efficiency: {transfer_ratio*100:.1f}%")
    print("(100% = full transfer, 0% = no transfer)")

    return {
        'source_bits': source_bits,
        'target_bits': target_bits,
        'source_accuracies': source_accuracies,
        'transfer_accuracies': transfer_accuracies,
        'source_mean': np.mean(source_accuracies),
        'transfer_mean': np.mean(transfer_accuracies),
        'transfer_efficiency': transfer_ratio,
    }


def multi_scale_transfer():
    """Test transfer across multiple scales."""

    print("="*70)
    print("MULTI-SCALE TRANSFER TEST")
    print("="*70)

    results = []

    # Train on 16 bits
    source_bits = 16
    source_samples = 100000

    print(f"\n[1] Training on {source_bits} bits...")
    source_inputs, source_outputs = generate_samples(source_samples, source_bits)

    split = int(0.8 * source_samples)
    X_train = source_outputs[:split]
    y_train = source_inputs[:split]

    models = []
    for bit_idx in range(source_bits):
        model = train_model(X_train, y_train[:, bit_idx])
        models.append(model)
        if (bit_idx + 1) % 4 == 0:
            print(f"  Trained {bit_idx + 1}/{source_bits} models")

    # Test on multiple target sizes
    for target_bits in [16, 24, 32, 48, 64]:
        print(f"\n[2] Testing transfer to {target_bits} bits...")

        target_samples = 30000
        target_inputs, target_outputs = generate_samples(target_samples, target_bits)

        accuracies = []
        for bit_idx in range(source_bits):
            pred = models[bit_idx].predict(target_outputs)
            actual = target_inputs[:, bit_idx]
            acc = accuracy_score(actual, pred)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        leak = mean_acc - 0.5

        results.append({
            'target_bits': target_bits,
            'mean_accuracy': mean_acc,
            'leak': leak,
        })

        print(f"  {target_bits} bits: accuracy={mean_acc:.4f}, leak={leak:+.4f}")

    print(f"\n{'='*70}")
    print("MULTI-SCALE SUMMARY")
    print("="*70)
    print(f"\nSource: {source_bits} bits (trained on 100% coverage)")
    print(f"\n{'Target':>8} {'Accuracy':>10} {'Leak':>10}")
    print("-"*30)
    for r in results:
        print(f"{r['target_bits']:>8} {r['mean_accuracy']:>10.4f} {r['leak']:>+10.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=int, default=16)
    parser.add_argument('--target', type=int, default=32)
    parser.add_argument('--samples', type=int, default=100000)
    parser.add_argument('--multi', action='store_true', help='Multi-scale test')
    args = parser.parse_args()

    if args.multi:
        multi_scale_transfer()
    else:
        transfer_experiment(
            source_bits=args.source,
            target_bits=args.target,
            source_samples=args.samples,
            target_samples=args.samples // 2,
        )
