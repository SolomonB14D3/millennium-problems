#!/usr/bin/env python3
"""
Output-Only Learning - MLX Version (Apple Silicon GPU)

Leverage M3 Ultra:
- 28 cores for data generation
- GPU (MLX) for neural network training
- 96GB RAM for large datasets
"""

import numpy as np
import hashlib
import time
import multiprocessing as mp
from typing import Tuple, Dict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

NUM_CORES = mp.cpu_count()
print(f"Cores: {NUM_CORES}, MLX GPU: {mx.default_device()}")


class LeakDetector(nn.Module):
    """Neural network to detect input bits from output."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, output_dim: int = 1):
        super().__init__()
        self.layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, output_dim),
        ]

    def __call__(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = nn.relu(layer(x))
        return self.layers[-1](x)


def generate_chunk(args):
    """Generate a chunk of samples."""
    chunk_size, input_bits, seed = args
    np.random.seed(seed)

    input_bytes = (input_bits + 7) // 8

    inputs = np.zeros((chunk_size, input_bits), dtype=np.float32)
    outputs = np.zeros((chunk_size, 256), dtype=np.float32)

    for i in range(chunk_size):
        data = bytearray(np.random.bytes(input_bytes))
        if input_bits % 8 != 0:
            mask = (1 << (input_bits % 8)) - 1
            data[0] &= mask
        data = bytes(data)

        h = hashlib.sha256(data).digest()

        inp_bits = np.unpackbits(np.frombuffer(bytes(data[:input_bytes]), dtype=np.uint8))
        inputs[i, :min(len(inp_bits), input_bits)] = inp_bits[:input_bits].astype(np.float32)
        outputs[i] = np.unpackbits(np.frombuffer(h, dtype=np.uint8)).astype(np.float32)

    return inputs, outputs


def generate_parallel(num_samples: int, input_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate samples in parallel across all cores."""
    chunk_size = max(10000, num_samples // NUM_CORES)
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    args_list = [(chunk_size, input_bits, seed) for seed in range(num_chunks)]

    print(f"Generating {num_samples:,} samples across {NUM_CORES} cores...")
    start = time.time()

    with mp.Pool(NUM_CORES) as pool:
        results = pool.map(generate_chunk, args_list)

    all_inputs = np.vstack([r[0] for r in results])[:num_samples]
    all_outputs = np.vstack([r[1] for r in results])[:num_samples]

    elapsed = time.time() - start
    print(f"Generated {num_samples:,} samples in {elapsed:.1f}s ({num_samples/elapsed:.0f}/s)")

    return all_inputs, all_outputs


def train_mlx(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    bit_idx: int,
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 0.001,
) -> float:
    """Train neural network for single bit using MLX."""

    # Convert to MLX arrays
    X_tr = mx.array(X_train)
    y_tr = mx.array(y_train[:, bit_idx:bit_idx+1])
    X_te = mx.array(X_test)
    y_te = mx.array(y_test[:, bit_idx:bit_idx+1])

    # Model
    model = LeakDetector(input_dim=256, hidden_dim=512, output_dim=1)
    mx.eval(model.parameters())

    # Optimizer
    optimizer = optim.Adam(learning_rate=lr)

    # Loss function
    def loss_fn(model, X, y):
        logits = model(X)
        return nn.losses.binary_cross_entropy(mx.sigmoid(logits), y, reduction='mean')

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Training loop
    num_batches = len(X_tr) // batch_size

    for epoch in range(epochs):
        # Shuffle
        perm = mx.array(np.random.permutation(len(X_tr)))
        X_tr_shuffled = X_tr[perm]
        y_tr_shuffled = y_tr[perm]

        epoch_loss = 0.0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            X_batch = X_tr_shuffled[start_idx:end_idx]
            y_batch = y_tr_shuffled[start_idx:end_idx]

            loss, grads = loss_and_grad_fn(model, X_batch, y_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += loss.item()

        epoch_loss /= num_batches

    # Evaluate
    logits = model(X_te)
    preds = (mx.sigmoid(logits) > 0.5).astype(mx.float32)
    accuracy = (preds == y_te).astype(mx.float32).mean().item()

    return accuracy


def run_test(input_bits: int, num_samples: int, epochs: int = 50):
    """Run full test with MLX."""

    print("="*70)
    print(f"OUTPUT-ONLY MLX - {input_bits} BITS, {num_samples:,} SAMPLES")
    print("="*70)

    input_space = 2 ** input_bits
    coverage = num_samples / input_space
    print(f"Input space: {input_space:,}")
    print(f"Coverage: {coverage*100:.1f}%" if coverage < 100 else f"Coverage: {coverage:.1f}x")

    # Generate data
    inputs, outputs = generate_parallel(num_samples, input_bits)

    # Split
    split = int(0.8 * num_samples)
    X_train, X_test = outputs[:split], outputs[split:]
    y_train, y_test = inputs[:split], inputs[split:]

    print(f"\nTraining {input_bits} bit predictors with MLX (GPU)...")
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    results = {'bit_accuracies': []}
    start = time.time()

    for bit_idx in range(input_bits):
        acc = train_mlx(X_train, y_train, X_test, y_test, bit_idx, epochs=epochs)
        results['bit_accuracies'].append(acc)

        elapsed = time.time() - start
        print(f"  Bit {bit_idx}: {acc:.3f} ({elapsed:.1f}s total)")

    results['mean_accuracy'] = np.mean(results['bit_accuracies'])
    results['leak_strength'] = results['mean_accuracy'] - 0.5

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
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    # Protect multiprocessing
    mp.set_start_method('spawn', force=True)
    run_test(args.bits, args.samples, args.epochs)
