#!/usr/bin/env python3
"""
Output-Only Learning v2 - Simple but fast

Use vectorized numpy for generation, HistGradientBoosting for speed.
Sequential training (still fast with HistGB).
"""

import numpy as np
import hashlib
from typing import Dict, Tuple
import time


def generate_samples(num_samples: int, input_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate samples with vectorized operations where possible."""
    input_bytes = (input_bits + 7) // 8

    inputs = np.zeros((num_samples, input_bits), dtype=np.uint8)
    outputs = np.zeros((num_samples, 256), dtype=np.uint8)

    # Generate all random data at once
    all_data = np.random.randint(0, 256, size=(num_samples, input_bytes), dtype=np.uint8)

    # Mask first byte if needed
    if input_bits % 8 != 0:
        mask = (1 << (input_bits % 8)) - 1
        all_data[:, 0] &= mask

    start = time.time()
    for i in range(num_samples):
        data = bytes(all_data[i])
        h = hashlib.sha256(data).digest()

        inputs[i] = np.unpackbits(all_data[i])[:input_bits]
        outputs[i] = np.unpackbits(np.frombuffer(h, dtype=np.uint8))

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"  {i+1:,}/{num_samples:,} ({rate:.0f}/s)")

    return inputs, outputs


def train(inputs: np.ndarray, outputs: np.ndarray) -> Dict:
    """Train with HistGradientBoosting (fast)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    num_samples, num_input_bits = inputs.shape

    X_train, X_test, y_train, y_test = train_test_split(
        outputs, inputs, test_size=0.2, random_state=42
    )

    results = {'bit_accuracies': []}

    start = time.time()
    for bit_idx in range(num_input_bits):
        y_train_bit = y_train[:, bit_idx]
        y_test_bit = y_test[:, bit_idx]

        if len(np.unique(y_train_bit)) < 2:
            results['bit_accuracies'].append(0.5)
            continue

        model = HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
        )

        model.fit(X_train, y_train_bit)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test_bit, pred)

        results['bit_accuracies'].append(acc)

        elapsed = time.time() - start
        print(f"  Bit {bit_idx}: {acc:.3f} ({elapsed:.1f}s)")

    results['mean_accuracy'] = np.mean(results['bit_accuracies'])
    results['leak_strength'] = results['mean_accuracy'] - 0.5

    return results


def run_test(input_bits: int, num_samples: int):
    """Run test."""
    print("="*70)
    print(f"OUTPUT-ONLY v2 - {input_bits} BITS, {num_samples:,} SAMPLES")
    print("="*70)

    input_space = 2 ** input_bits
    coverage = num_samples / input_space
    print(f"Input space: {input_space:,}")
    print(f"Coverage: {coverage*100:.1f}%" if coverage < 100 else f"Coverage: {coverage:.1f}x")

    print(f"\nGenerating...")
    inputs, outputs = generate_samples(num_samples, input_bits)

    print(f"\nTraining...")
    results = train(inputs, outputs)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Mean accuracy: {results['mean_accuracy']:.4f}")
    print(f"Leak strength: {results['leak_strength']:+.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bits', type=int, default=16)
    parser.add_argument('--samples', type=int, default=60000)
    args = parser.parse_args()

    run_test(args.bits, args.samples)
