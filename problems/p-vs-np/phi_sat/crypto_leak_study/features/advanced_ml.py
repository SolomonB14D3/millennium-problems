#!/usr/bin/env python3
"""
Advanced ML Pattern Hunting

1. SHAP feature importance - which output bits predict input?
2. Autoencoder - learn to reconstruct high bits from low bits
3. Control experiment - simulated hardened HD keys
"""

import numpy as np
import hashlib
import hmac
import time
from typing import Tuple, List, Dict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Puzzle keys
PUZZLE_KEYS = {
    1: 0x1, 2: 0x3, 3: 0x7, 4: 0x8, 5: 0x15, 6: 0x31, 7: 0x4c, 8: 0xe0,
    9: 0x1d3, 10: 0x202, 11: 0x483, 12: 0xa7b, 13: 0x1460, 14: 0x2930,
    15: 0x68f3, 16: 0xc936, 17: 0x1764f, 18: 0x3080d, 19: 0x5749f,
    20: 0xd2c55, 21: 0x1ba534, 22: 0x2de40f, 23: 0x556e52, 24: 0xdc2a04,
    25: 0x1fa5ee5, 26: 0x340326e, 27: 0x6ac3875, 28: 0xd916ce8,
    29: 0x17e2551e, 30: 0x3d94cd64, 31: 0x7d4fe747, 32: 0xb862a62e,
    33: 0x1a96ca8d8, 34: 0x34a65911d, 35: 0x4aed21170, 36: 0x9de820a7c,
    37: 0x1757756a93, 38: 0x22382facd0, 39: 0x4b5f8303e9, 40: 0xe9ae4933d6,
}

SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def sha256_bits(data: bytes) -> np.ndarray:
    """Get SHA256 hash as bit array."""
    h = hashlib.sha256(data).digest()
    return np.unpackbits(np.frombuffer(h, dtype=np.uint8))


def generate_samples(num_samples: int, input_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate SHA256 input/output pairs."""
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


def shap_feature_importance(input_bits: int = 16, num_samples: int = 50000):
    """
    Find which output bits are most important for predicting each input bit.
    Uses tree-based feature importance (faster than SHAP).
    """
    print("="*70)
    print(f"FEATURE IMPORTANCE ANALYSIS - {input_bits} bits")
    print("="*70)

    print(f"\nGenerating {num_samples:,} samples...")
    inputs, outputs = generate_samples(num_samples, input_bits)

    split = int(0.8 * num_samples)
    X_train, X_test = outputs[:split], outputs[split:]
    y_train, y_test = inputs[:split], inputs[split:]

    # Track importance across all input bits
    all_importances = np.zeros((input_bits, 256))
    accuracies = []

    print(f"\nTraining {input_bits} models and extracting feature importance...")
    start = time.time()

    for bit_idx in range(input_bits):
        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X_train, y_train[:, bit_idx])

        # Get feature importance (not SHAP, but built-in importance)
        # For HistGradientBoosting, we need to compute permutation importance
        # or use a proxy. Let's use prediction sensitivity.

        pred = model.predict(X_test)
        acc = accuracy_score(y_test[:, bit_idx], pred)
        accuracies.append(acc)

        # Compute importance via prediction change when masking each output bit
        base_proba = model.predict_proba(X_test)[:, 1]

        importances = np.zeros(256)
        for out_bit in range(256):
            # Flip this output bit
            X_test_flipped = X_test.copy()
            X_test_flipped[:, out_bit] = 1 - X_test_flipped[:, out_bit]

            new_proba = model.predict_proba(X_test_flipped)[:, 1]
            importance = np.mean(np.abs(new_proba - base_proba))
            importances[out_bit] = importance

        all_importances[bit_idx] = importances

        if (bit_idx + 1) % 4 == 0:
            print(f"  Bit {bit_idx + 1}/{input_bits}: acc={acc:.3f} ({time.time()-start:.0f}s)")

    print(f"\nMean accuracy: {np.mean(accuracies):.4f}")

    # Analyze importance patterns
    print(f"\n{'='*70}")
    print("IMPORTANCE ANALYSIS")
    print("="*70)

    # Which output bits are most important OVERALL?
    overall_importance = np.mean(all_importances, axis=0)
    top_outputs = np.argsort(overall_importance)[::-1][:20]

    print(f"\nTop 20 most important output bits (across all input bits):")
    for i, out_bit in enumerate(top_outputs):
        print(f"  {i+1}. Output bit {out_bit}: importance = {overall_importance[out_bit]:.4f}")

    # Which input bits have UNIQUE importance patterns?
    avg_pattern = np.mean(all_importances, axis=0)
    uniqueness = np.mean(np.abs(all_importances - avg_pattern), axis=1)

    print(f"\nInput bits with most UNIQUE importance patterns:")
    unique_inputs = np.argsort(uniqueness)[::-1][:10]
    for inp_bit in unique_inputs:
        print(f"  Input bit {inp_bit}: uniqueness = {uniqueness[inp_bit]:.4f}, acc = {accuracies[inp_bit]:.3f}")

    # Are certain output bits specifically important for certain input bits?
    print(f"\nInput-specific important outputs:")
    for inp_bit in range(min(8, input_bits)):
        top_for_this = np.argsort(all_importances[inp_bit])[::-1][:5]
        print(f"  Input {inp_bit}: top outputs = {top_for_this.tolist()}")

    return all_importances, accuracies


def autoencoder_bit_reconstruction(input_bits: int = 16, num_samples: int = 50000):
    """
    Train autoencoder: hash output → reconstruct input bits.

    If high bits can be reconstructed from low bits, there's structure.
    """
    print("\n" + "="*70)
    print(f"AUTOENCODER BIT RECONSTRUCTION - {input_bits} bits")
    print("="*70)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("PyTorch not available, skipping autoencoder")
        return None

    print(f"\nGenerating {num_samples:,} samples...")
    inputs, outputs = generate_samples(num_samples, input_bits)

    # Convert to tensors
    X = torch.FloatTensor(outputs)
    y = torch.FloatTensor(inputs)

    split = int(0.8 * num_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Simple autoencoder: 256 (output) -> 64 -> input_bits
    class BitReconstructor(nn.Module):
        def __init__(self, output_size=256, hidden_size=128, input_size=16):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(output_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_size),
                nn.Sigmoid(),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    model = BitReconstructor(256, 128, input_bits)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print(f"\nTraining autoencoder...")
    start = time.time()

    for epoch in range(20):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: loss = {total_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred = model(X_test)
        pred_bits = (pred > 0.5).float()

        # Per-bit accuracy
        bit_accuracies = []
        for bit_idx in range(input_bits):
            acc = (pred_bits[:, bit_idx] == y_test[:, bit_idx]).float().mean().item()
            bit_accuracies.append(acc)

    print(f"\nAutoencoder results ({time.time()-start:.0f}s):")
    print(f"  Mean bit accuracy: {np.mean(bit_accuracies):.4f}")
    print(f"  Random baseline: 0.5")
    print(f"  Leak strength: {np.mean(bit_accuracies) - 0.5:+.4f}")

    print(f"\n  Per-bit accuracies:")
    for bit_idx in range(input_bits):
        print(f"    Bit {bit_idx}: {bit_accuracies[bit_idx]:.3f}")

    return bit_accuracies


def control_hardened_hd():
    """
    Control experiment: Generate keys using hardened HD derivation.

    If our analysis finds NO leak in these, and we DO find leak in real puzzles,
    that's evidence the real puzzles have exploitable structure.
    """
    print("\n" + "="*70)
    print("CONTROL: HARDENED HD WALLET SIMULATION")
    print("="*70)

    # Simulate hardened HD derivation
    # child_key = HMAC-SHA512(chaincode, 0x00 || parent_key || index)[:32]

    np.random.seed(42)
    master_key = np.random.bytes(32)
    chaincode = np.random.bytes(32)

    def derive_hardened(parent_key: bytes, chaincode: bytes, index: int) -> Tuple[bytes, bytes]:
        """BIP32-like hardened derivation."""
        # Hardened: use 0x00 || key || index
        data = b'\x00' + parent_key + index.to_bytes(4, 'big')
        I = hmac.new(chaincode, data, hashlib.sha512).digest()
        child_key = I[:32]
        child_chaincode = I[32:]
        return child_key, child_chaincode

    # Generate 40 consecutive hardened keys
    hd_keys = {}
    current_key = master_key
    current_cc = chaincode

    for i in range(1, 41):
        current_key, current_cc = derive_hardened(current_key, current_cc, i)
        # Convert to int and mask to i bits
        full_key = int.from_bytes(current_key, 'big') % SECP256K1_ORDER

        # Mask like puzzle: low (i-1) bits + set high bit
        low_bits = full_key % (1 << (i - 1))
        masked = low_bits | (1 << (i - 1))
        hd_keys[i] = masked

    print(f"\nGenerated 40 hardened HD keys (masked to puzzle bit lengths)")

    # Run same analyses as on real puzzles
    print(f"\n--- Ratio Analysis (like real puzzles) ---")
    ratios = []
    for i in range(1, 40):
        if hd_keys[i] > 0:
            ratio = hd_keys[i + 1] / hd_keys[i]
            ratios.append(ratio)

    print(f"HD wallet ratio stats: mean={np.mean(ratios):.3f}, std={np.std(ratios):.3f}")

    # Compare to real puzzle stats
    real_ratios = []
    for i in range(14, 40):
        if PUZZLE_KEYS[i] > 0:
            ratio = PUZZLE_KEYS[i + 1] / PUZZLE_KEYS[i]
            real_ratios.append(ratio)

    print(f"Real puzzle (14+) stats: mean={np.mean(real_ratios):.3f}, std={np.std(real_ratios):.3f}")

    # CRT consistency check
    print(f"\n--- CRT Consistency (like deep analysis) ---")

    inconsistent_count = 0
    for i in range(35, 40):
        k_i = hd_keys[i]
        k_next = hd_keys[i + 1]

        low_i = k_i & ((1 << (i - 1)) - 1)
        low_next = k_next & ((1 << i) - 1)
        low_next_trunc = low_next & ((1 << (i - 1)) - 1)

        delta1 = (low_next_trunc - low_i) % (1 << (i - 1))

        # Check against next pair
        if i < 39:
            k_next2 = hd_keys[i + 2]
            low_next2 = k_next2 & ((1 << (i + 1)) - 1)
            low_next2_trunc = low_next2 & ((1 << i) - 1)
            delta2 = (low_next2_trunc - low_next) % (1 << i)

            # Check consistency at common mod
            common_bits = i - 1
            if (delta1 % (1 << common_bits)) != (delta2 % (1 << common_bits)):
                inconsistent_count += 1

    print(f"HD wallet CRT inconsistencies: {inconsistent_count}/4")
    print(f"(Hardened HD SHOULD be inconsistent, like real puzzles)")

    # ML leak test
    print(f"\n--- ML Leak Test ---")

    # Build training data from HD keys
    X = []  # Features from key structure
    y = []  # Target: next key's low bits

    for i in range(14, 39):
        # Feature: current key's bits
        key = hd_keys[i]
        features = [(key >> b) & 1 for b in range(i)]
        X.append(features[:16])  # Use first 16 bits as features

        # Target: next key's bit 0
        y.append(hd_keys[i + 1] & 1)

    # Pad features to same length
    X = np.array([f + [0] * (16 - len(f)) for f in X])
    y = np.array(y)

    if len(X) > 10:
        from sklearn.model_selection import cross_val_score
        model = HistGradientBoostingClassifier(max_iter=50, random_state=42)
        scores = cross_val_score(model, X, y, cv=5)
        print(f"HD wallet predictability: {np.mean(scores):.3f}")
        print(f"(Random baseline: 0.5)")

    return hd_keys


def seed_recovery_extension():
    """
    We know seeds for puzzles 1-13. Try to extend or find patterns.

    Known seeds:
    - Puzzles 1-8: seed 34378104
    - Puzzles 9-11: seed 78372297
    - Puzzles 12-13: seed 2408880
    """
    print("\n" + "="*70)
    print("SEED RECOVERY EXTENSION")
    print("="*70)

    import random

    # Verify known seeds
    seeds = {
        (1, 8): 34378104,
        (9, 11): 78372297,
        (12, 13): 2408880,
    }

    print("\nVerifying known seeds:")
    for (start, end), seed in seeds.items():
        random.seed(seed)
        matches = 0
        for p in range(start, end + 1):
            max_val = (1 << p) - 1
            min_val = 1 << (p - 1)
            generated = random.randint(min_val, max_val)
            actual = PUZZLE_KEYS[p]
            if generated == actual:
                matches += 1
        print(f"  Seed {seed} for puzzles {start}-{end}: {matches}/{end-start+1} matches")

    # Try to find seed for puzzle 14+
    print("\nSearching for seed that produces puzzle 14...")

    found_seeds = []
    for seed in range(10000000):  # Search first 10M seeds
        random.seed(seed)

        # Generate what puzzle 14 would be
        max_val = (1 << 14) - 1
        min_val = 1 << 13
        generated = random.randint(min_val, max_val)

        if generated == PUZZLE_KEYS[14]:
            # Found a candidate! Check puzzle 15
            gen_15 = random.randint(1 << 14, (1 << 15) - 1)
            if gen_15 == PUZZLE_KEYS[15]:
                print(f"  STRONG MATCH: seed {seed}")
                found_seeds.append(seed)
            else:
                # Still record single match
                if len(found_seeds) < 5:
                    print(f"  Partial match (14 only): seed {seed}")

        if seed % 1000000 == 0 and seed > 0:
            print(f"  Searched {seed:,}...")

    if not found_seeds:
        print("  No matching seed found for puzzles 14+")
        print("  This suggests different generation method (possibly HD wallet)")

    # Analyze the known seed values
    print("\nKnown seed analysis:")
    seed_values = [34378104, 78372297, 2408880]
    print(f"  Seeds: {seed_values}")
    print(f"  Differences: {seed_values[1] - seed_values[0]}, {seed_values[2] - seed_values[1]}")

    # Check if seeds have pattern
    print(f"  Seed 1 bits: {bin(seed_values[0])}")
    print(f"  Seed 2 bits: {bin(seed_values[1])}")
    print(f"  Seed 3 bits: {bin(seed_values[2])}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--importance', action='store_true')
    parser.add_argument('--autoencoder', action='store_true')
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--seeds', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--bits', type=int, default=16)
    parser.add_argument('--samples', type=int, default=30000)
    args = parser.parse_args()

    if args.all or args.importance:
        shap_feature_importance(args.bits, args.samples)

    if args.all or args.autoencoder:
        autoencoder_bit_reconstruction(args.bits, args.samples)

    if args.all or args.control:
        control_hardened_hd()

    if args.all or args.seeds:
        seed_recovery_extension()

    if not any([args.importance, args.autoencoder, args.control, args.seeds, args.all]):
        # Default: run importance and control
        shap_feature_importance(args.bits, args.samples)
        control_hardened_hd()
