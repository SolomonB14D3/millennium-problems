#!/usr/bin/env python3
"""
Output-Only Learning

The output IS the information. 256 bits, deterministic function of input.
Can we learn to read it?

No intermediate states. No timing. Just input → output.
"""

import numpy as np
from typing import Dict, Tuple
import hashlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sha256_simple(data: bytes) -> np.ndarray:
    """Just the hash, as bits."""
    h = hashlib.sha256(data).digest()
    return np.unpackbits(np.frombuffer(h, dtype=np.uint8))


def generate_output_only_dataset(
    num_samples: int,
    input_bits: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset: input bits → output bits only.

    No intermediate states. No timing. Just the transformation.
    """
    input_bytes = (input_bits + 7) // 8

    inputs = np.zeros((num_samples, input_bits), dtype=np.uint8)
    outputs = np.zeros((num_samples, 256), dtype=np.uint8)

    for i in range(num_samples):
        # Generate input
        data = bytearray(np.random.bytes(input_bytes))
        if input_bits % 8 != 0:
            mask = (1 << (input_bits % 8)) - 1
            data[0] &= mask
        data = bytes(data)

        # Get output
        output = sha256_simple(data)

        # Store
        input_bits_arr = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        inputs[i, :min(len(input_bits_arr), input_bits)] = input_bits_arr[:input_bits]
        outputs[i] = output

        if (i + 1) % max(1, num_samples // 10) == 0:
            print(f"  {i+1}/{num_samples}")

    return inputs, outputs


def train_output_only(
    inputs: np.ndarray,
    outputs: np.ndarray,
    model_type: str = 'gb'
) -> Dict:
    """
    Train to predict input bits from output bits ONLY.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    num_samples, num_input_bits = inputs.shape

    X_train, X_test, y_train, y_test = train_test_split(
        outputs, inputs, test_size=0.2, random_state=42
    )

    results = {
        'num_samples': num_samples,
        'num_input_bits': num_input_bits,
        'num_features': 256,  # output bits only
        'bit_accuracies': [],
        'model_type': model_type,
    }

    for bit_idx in range(num_input_bits):
        y_train_bit = y_train[:, bit_idx]
        y_test_bit = y_test[:, bit_idx]

        if len(np.unique(y_train_bit)) < 2:
            results['bit_accuracies'].append(0.5)
            continue

        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gb':
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=42
            )
        elif model_type == 'mlp':
            model = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                max_iter=500,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train_bit)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test_bit, pred)

        results['bit_accuracies'].append(acc)
        print(f"  Bit {bit_idx}: {acc:.3f}")

    results['mean_accuracy'] = np.mean(results['bit_accuracies'])
    results['min_accuracy'] = np.min(results['bit_accuracies'])
    results['max_accuracy'] = np.max(results['bit_accuracies'])
    results['leak_strength'] = results['mean_accuracy'] - 0.5

    return results


def run_output_only_test(input_bits: int = 8, num_samples: int = 5000):
    """Test learning from output only."""

    print("="*70)
    print(f"OUTPUT-ONLY LEARNING - {input_bits} BITS")
    print("="*70)
    print("\nThe question: Can we learn to read the input from output alone?")
    print("Features: 256 output bits. Nothing else.")

    print(f"\nGenerating {num_samples} samples...")
    inputs, outputs = generate_output_only_dataset(num_samples, input_bits)

    print(f"\nTraining (Gradient Boosting)...")
    results = train_output_only(inputs, outputs, model_type='gb')

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Mean accuracy: {results['mean_accuracy']:.4f}")
    print(f"Min accuracy:  {results['min_accuracy']:.4f}")
    print(f"Max accuracy:  {results['max_accuracy']:.4f}")
    print(f"Leak strength: {results['leak_strength']:+.4f}")
    print(f"Random baseline: 0.5000")

    if results['leak_strength'] > 0.01:
        print("\n*** LEAK DETECTED FROM OUTPUT ALONE ***")
    else:
        print("\nNo significant leak from output alone.")

    return results


def run_scaling_test(max_bits: int = 32, samples: int = 5000):
    """Test how output-only accuracy scales with input bits."""

    print("="*70)
    print("OUTPUT-ONLY SCALING TEST")
    print("="*70)

    results = {}

    for bits in [4, 8, 16, 24, 32]:
        if bits > max_bits:
            break

        print(f"\n{'='*50}")
        print(f"Testing {bits} bits...")
        print(f"{'='*50}")

        inputs, outputs = generate_output_only_dataset(samples, bits)
        r = train_output_only(inputs, outputs, model_type='gb')
        results[bits] = r

        print(f"\n{bits} bits: accuracy={r['mean_accuracy']:.3f}, leak={r['leak_strength']:+.3f}")

    print("\n" + "="*70)
    print("SCALING SUMMARY")
    print("="*70)
    print(f"\n{'Bits':>6} {'Accuracy':>10} {'Leak':>10}")
    print("-"*30)
    for bits, r in sorted(results.items()):
        print(f"{bits:>6} {r['mean_accuracy']:>10.3f} {r['leak_strength']:>+10.3f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--scaling', action='store_true', help='Run scaling test')
    args = parser.parse_args()

    if args.scaling:
        run_scaling_test(max_bits=args.bits, samples=args.samples)
    else:
        run_output_only_test(args.bits, args.samples)
