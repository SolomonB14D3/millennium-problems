#!/usr/bin/env python3
"""
Output-Only Learning - Full Power

Use parallel data generation + HistGradientBoosting (which found the pattern).
"""

import numpy as np
import hashlib
import time
import multiprocessing as mp
from typing import Tuple, Dict

NUM_CORES = mp.cpu_count()


def generate_chunk(args):
    """Generate a chunk of samples."""
    chunk_size, input_bits, seed = args
    np.random.seed(seed)
    input_bytes = (input_bits + 7) // 8

    inputs = np.zeros((chunk_size, input_bits), dtype=np.uint8)
    outputs = np.zeros((chunk_size, 256), dtype=np.uint8)

    for i in range(chunk_size):
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


def generate_parallel(num_samples: int, input_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate samples in parallel."""
    chunk_size = max(10000, num_samples // NUM_CORES)
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    args_list = [(chunk_size, input_bits, seed) for seed in range(num_chunks)]

    print(f"Generating {num_samples:,} samples ({NUM_CORES} cores)...")
    start = time.time()

    with mp.Pool(NUM_CORES) as pool:
        results = pool.map(generate_chunk, args_list)

    all_inputs = np.vstack([r[0] for r in results])[:num_samples]
    all_outputs = np.vstack([r[1] for r in results])[:num_samples]

    elapsed = time.time() - start
    print(f"Generated in {elapsed:.1f}s ({num_samples/elapsed:,.0f}/s)")

    return all_inputs, all_outputs


def train_histgb(X_train, y_train, X_test, y_test, input_bits):
    """Train HistGradientBoosting for all bits."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score

    results = {'bit_accuracies': []}
    start = time.time()

    for bit_idx in range(input_bits):
        y_tr = y_train[:, bit_idx]
        y_te = y_test[:, bit_idx]

        if len(np.unique(y_tr)) < 2:
            results['bit_accuracies'].append(0.5)
            continue

        model = HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=12,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42,
        )

        model.fit(X_train, y_tr)
        pred = model.predict(X_test)
        acc = accuracy_score(y_te, pred)
        results['bit_accuracies'].append(acc)

        elapsed = time.time() - start
        print(f"  Bit {bit_idx}: {acc:.3f} ({elapsed:.1f}s)")

    results['mean_accuracy'] = np.mean(results['bit_accuracies'])
    results['leak_strength'] = results['mean_accuracy'] - 0.5

    return results


def run_scaling_test():
    """Test how accuracy scales with coverage."""
    print("="*70)
    print("SCALING TEST: How does accuracy scale with coverage?")
    print("="*70)

    input_bits = 16
    input_space = 2 ** input_bits

    results = []

    for coverage_pct in [50, 100, 200, 400]:
        num_samples = int(input_space * coverage_pct / 100)

        print(f"\n{'='*50}")
        print(f"Coverage: {coverage_pct}% ({num_samples:,} samples)")
        print(f"{'='*50}")

        inputs, outputs = generate_parallel(num_samples, input_bits)

        # Split
        split = int(0.8 * num_samples)
        X_train, X_test = outputs[:split], outputs[split:]
        y_train, y_test = inputs[:split], inputs[split:]

        print(f"Training...")
        r = train_histgb(X_train, y_train, X_test, y_test, input_bits)

        results.append({
            'coverage': coverage_pct,
            'samples': num_samples,
            'accuracy': r['mean_accuracy'],
            'leak': r['leak_strength']
        })

        print(f"\n{coverage_pct}% coverage: accuracy={r['mean_accuracy']:.3f}, leak={r['leak_strength']:+.3f}")

    print("\n" + "="*70)
    print("SCALING SUMMARY (16 bits)")
    print("="*70)
    print(f"\n{'Coverage':>10} {'Samples':>12} {'Accuracy':>10} {'Leak':>10}")
    print("-"*45)
    for r in results:
        print(f"{r['coverage']:>9}% {r['samples']:>12,} {r['accuracy']:>10.3f} {r['leak']:>+10.3f}")


def run_single_test(input_bits: int, num_samples: int):
    """Run single test."""
    print("="*70)
    print(f"OUTPUT-ONLY POWER - {input_bits} BITS, {num_samples:,} SAMPLES")
    print("="*70)

    input_space = 2 ** input_bits
    coverage = num_samples / input_space * 100
    print(f"Input space: {input_space:,}, Coverage: {coverage:.0f}%")

    inputs, outputs = generate_parallel(num_samples, input_bits)

    split = int(0.8 * num_samples)
    X_train, X_test = outputs[:split], outputs[split:]
    y_train, y_test = inputs[:split], inputs[split:]

    print(f"\nTraining (HistGradientBoosting)...")
    results = train_histgb(X_train, y_train, X_test, y_test, input_bits)

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
    parser.add_argument('--samples', type=int, default=100000)
    parser.add_argument('--scaling', action='store_true')
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    if args.scaling:
        run_scaling_test()
    else:
        run_single_test(args.bits, args.samples)
