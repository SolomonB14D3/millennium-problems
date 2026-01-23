#!/usr/bin/env python3
"""
Output-Only Learning - Fast/Parallel Version

Use all available cores, vectorized operations, efficient models.
"""

import numpy as np
import hashlib
from typing import Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_CORES = mp.cpu_count()
print(f"Using {NUM_CORES} cores")


def generate_batch(args):
    """Generate a batch of samples (for parallel execution)."""
    batch_size, input_bits, seed = args
    np.random.seed(seed)

    input_bytes = (input_bits + 7) // 8

    inputs = np.zeros((batch_size, input_bits), dtype=np.uint8)
    outputs = np.zeros((batch_size, 256), dtype=np.uint8)

    for i in range(batch_size):
        data = bytearray(np.random.bytes(input_bytes))
        if input_bits % 8 != 0:
            mask = (1 << (input_bits % 8)) - 1
            data[0] &= mask
        data = bytes(data)

        h = hashlib.sha256(data).digest()
        output = np.unpackbits(np.frombuffer(h, dtype=np.uint8))

        input_bits_arr = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        inputs[i, :min(len(input_bits_arr), input_bits)] = input_bits_arr[:input_bits]
        outputs[i] = output

    return inputs, outputs


def generate_parallel(num_samples: int, input_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate samples in parallel across all cores."""
    batch_size = max(1000, num_samples // NUM_CORES)
    num_batches = (num_samples + batch_size - 1) // batch_size

    args_list = [(batch_size, input_bits, seed) for seed in range(num_batches)]

    all_inputs = []
    all_outputs = []

    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = [executor.submit(generate_batch, args) for args in args_list]

        for i, future in enumerate(as_completed(futures)):
            inputs, outputs = future.result()
            all_inputs.append(inputs)
            all_outputs.append(outputs)

            if (i + 1) % 10 == 0:
                print(f"  Generated {(i+1) * batch_size}/{num_samples}")

    inputs = np.vstack(all_inputs)[:num_samples]
    outputs = np.vstack(all_outputs)[:num_samples]

    return inputs, outputs


def train_fast(inputs: np.ndarray, outputs: np.ndarray) -> Dict:
    """Train with optimized settings."""
    from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    num_samples, num_input_bits = inputs.shape

    X_train, X_test, y_train, y_test = train_test_split(
        outputs, inputs, test_size=0.2, random_state=42
    )

    results = {
        'num_samples': num_samples,
        'num_input_bits': num_input_bits,
        'bit_accuracies': [],
    }

    # Use HistGradientBoosting - much faster for large datasets
    for bit_idx in range(num_input_bits):
        y_train_bit = y_train[:, bit_idx]
        y_test_bit = y_test[:, bit_idx]

        if len(np.unique(y_train_bit)) < 2:
            results['bit_accuracies'].append(0.5)
            continue

        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
        )

        model.fit(X_train, y_train_bit)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test_bit, pred)

        results['bit_accuracies'].append(acc)
        print(f"  Bit {bit_idx}: {acc:.3f}")

    results['mean_accuracy'] = np.mean(results['bit_accuracies'])
    results['leak_strength'] = results['mean_accuracy'] - 0.5

    return results


def train_all_bits_parallel(inputs: np.ndarray, outputs: np.ndarray) -> Dict:
    """Train all bits in parallel using joblib."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from joblib import Parallel, delayed

    num_samples, num_input_bits = inputs.shape

    X_train, X_test, y_train, y_test = train_test_split(
        outputs, inputs, test_size=0.2, random_state=42
    )

    def train_single_bit(bit_idx, X_tr, X_te, y_tr, y_te):
        y_train_bit = y_tr[:, bit_idx]
        y_test_bit = y_te[:, bit_idx]

        if len(np.unique(y_train_bit)) < 2:
            return bit_idx, 0.5

        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
        )

        model.fit(X_tr, y_train_bit)
        pred = model.predict(X_te)
        acc = accuracy_score(y_test_bit, pred)

        return bit_idx, acc

    # Use joblib for parallel execution
    results_list = Parallel(n_jobs=-1, verbose=10)(
        delayed(train_single_bit)(i, X_train, X_test, y_train, y_test)
        for i in range(num_input_bits)
    )

    results = {'bit_accuracies': [0.0] * num_input_bits}
    for bit_idx, acc in results_list:
        results['bit_accuracies'][bit_idx] = acc
        print(f"  Bit {bit_idx}: {acc:.3f}")

    results['mean_accuracy'] = np.mean(results['bit_accuracies'])
    results['leak_strength'] = results['mean_accuracy'] - 0.5

    return results


def run_test(input_bits: int, num_samples: int):
    """Run full test with parallel generation and training."""

    print("="*70)
    print(f"OUTPUT-ONLY LEARNING (FAST) - {input_bits} BITS, {num_samples:,} SAMPLES")
    print("="*70)

    input_space = 2 ** input_bits
    coverage = num_samples / input_space
    print(f"Input space: {input_space:,}")
    print(f"Coverage: {coverage*100:.1f}%")

    print(f"\nGenerating samples (parallel)...")
    inputs, outputs = generate_parallel(num_samples, input_bits)
    print(f"Generated {len(inputs):,} samples")

    print(f"\nTraining (parallel)...")
    results = train_all_bits_parallel(inputs, outputs)

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
    parser.add_argument('--samples', type=int, default=50000)
    args = parser.parse_args()

    run_test(args.bits, args.samples)
