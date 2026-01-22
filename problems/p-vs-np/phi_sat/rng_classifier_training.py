#!/usr/bin/env python3
"""
RNG Classifier Training

Instead of brute-forcing seeds, train ML to recognize RNG signatures.

Training data:
- Generate sequences from known RNGs (MT19937, LCG, SHA, etc.)
- Extract fingerprint features
- Train classifier to identify RNG type

Then:
- Extract fingerprint from actual puzzle keys
- Classify which RNG family produced them
- This narrows the search dramatically

"Include all of that in the training and extinct things that are hurting"
"""

import random
import hashlib
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
import struct

# =============================================================================
# RNG IMPLEMENTATIONS (from different eras/sources)
# =============================================================================

def mt19937_sequence(seed: int, n_puzzles: int = 70) -> List[int]:
    """Python's Mersenne Twister (random.randint)."""
    random.seed(seed)
    keys = []
    for n in range(1, n_puzzles + 1):
        min_val = 2**(n-1) if n > 1 else 1
        max_val = 2**n - 1
        keys.append(random.randint(min_val, max_val))
    return keys


def lcg_glibc_sequence(seed: int, n_puzzles: int = 70) -> List[int]:
    """glibc LCG (common in C programs)."""
    a, c, m = 1103515245, 12345, 2**31
    state = seed

    def next_val():
        nonlocal state
        state = (a * state + c) % m
        return state

    keys = []
    for n in range(1, n_puzzles + 1):
        min_val = 2**(n-1) if n > 1 else 1
        max_val = 2**n - 1
        range_size = max_val - min_val + 1

        # Need multiple calls for large ranges
        bits_needed = n
        raw = 0
        for _ in range((bits_needed + 30) // 31):
            raw = (raw << 31) | next_val()

        keys.append(min_val + (raw % range_size))
    return keys


def sha256_counter_sequence(seed: int, n_puzzles: int = 70) -> List[int]:
    """SHA256(seed || counter) DRBG pattern."""
    seed_bytes = seed.to_bytes(32, 'big')
    counter = 0

    def next_bytes(n: int) -> bytes:
        nonlocal counter
        result = b''
        while len(result) < n:
            data = seed_bytes + counter.to_bytes(8, 'big')
            result += hashlib.sha256(data).digest()
            counter += 1
        return result[:n]

    keys = []
    for n in range(1, n_puzzles + 1):
        min_val = 2**(n-1) if n > 1 else 1
        max_val = 2**n - 1
        range_size = max_val - min_val + 1

        num_bytes = (n + 7) // 8 + 1
        raw = int.from_bytes(next_bytes(num_bytes), 'big')
        keys.append(min_val + (raw % range_size))
    return keys


def xorshift128_sequence(seed: int, n_puzzles: int = 70) -> List[int]:
    """Xorshift128 - popular fast RNG."""
    state = [
        (seed) & 0xFFFFFFFF,
        (seed >> 32) & 0xFFFFFFFF,
        (seed * 0x5DEECE66D) & 0xFFFFFFFF,
        (seed * 0xB) & 0xFFFFFFFF,
    ]
    if all(s == 0 for s in state):
        state[0] = 1

    def next_val():
        t = state[3]
        s = state[0]
        state[3] = state[2]
        state[2] = state[1]
        state[1] = s
        t ^= (t << 11) & 0xFFFFFFFF
        t ^= (t >> 8) & 0xFFFFFFFF
        state[0] = t ^ s ^ ((s >> 19) & 0xFFFFFFFF)
        return state[0]

    keys = []
    for n in range(1, n_puzzles + 1):
        min_val = 2**(n-1) if n > 1 else 1
        max_val = 2**n - 1
        range_size = max_val - min_val + 1

        bits_needed = n
        raw = 0
        for _ in range((bits_needed + 31) // 32):
            raw = (raw << 32) | next_val()

        keys.append(min_val + (raw % range_size))
    return keys


def manual_random_sequence(seed: int, n_puzzles: int = 70) -> List[int]:
    """
    Simulate 'manual' selection with some randomness.
    Human-chosen numbers tend to avoid extremes and have patterns.
    """
    random.seed(seed)
    keys = []
    for n in range(1, n_puzzles + 1):
        min_val = 2**(n-1) if n > 1 else 1
        max_val = 2**n - 1

        # Humans tend to pick values not at extremes
        # Use beta distribution centered around 0.5
        position = random.betavariate(2, 2)  # Peaked at center
        key = int(min_val + position * (max_val - min_val))
        keys.append(key)
    return keys


def mixed_rng_sequence(seed: int, n_puzzles: int = 70) -> List[int]:
    """
    Mix of RNGs - maybe creator used different methods.
    First few puzzles one way, rest another way.
    """
    random.seed(seed)

    keys = []
    for n in range(1, n_puzzles + 1):
        min_val = 2**(n-1) if n > 1 else 1
        max_val = 2**n - 1

        if n <= 10:
            # Small puzzles: might be more manual/deliberate
            position = random.betavariate(2, 2)
            key = int(min_val + position * (max_val - min_val))
        else:
            # Larger puzzles: pure random
            key = random.randint(min_val, max_val)

        keys.append(key)
    return keys


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_rng_features(keys: List[int]) -> np.ndarray:
    """
    Extract features that distinguish RNG types.

    Features designed to capture:
    1. Bit-level patterns (RNG internal state leakage)
    2. Sequential correlations (LCG weakness)
    3. Distribution properties (human bias detection)
    4. Specific pattern frequencies (fingerprints)
    """
    n_puzzles = len(keys)
    features = []

    # === BIT FREQUENCY FEATURES ===
    # Count 1s at each bit position (first 40 bits)
    bit_freqs = []
    for bit_pos in range(40):
        ones = sum(1 for i, k in enumerate(keys) if (i + 1) > bit_pos and (k >> bit_pos) & 1)
        total = sum(1 for i in range(len(keys)) if (i + 1) > bit_pos)
        bit_freqs.append(ones / total if total > 0 else 0.5)
    features.extend(bit_freqs)

    # === POSITION FEATURES ===
    # Where each key falls in its valid range
    positions = []
    for n, key in enumerate(keys, 1):
        min_val = 2**(n-1) if n > 1 else 1
        max_val = 2**n - 1
        pos = (key - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        positions.append(pos)

    features.append(np.mean(positions))  # Mean position
    features.append(np.std(positions))   # Std of positions
    features.append(np.median(positions))  # Median

    # Position distribution (quartiles)
    q1 = sum(1 for p in positions if p < 0.25) / len(positions)
    q2 = sum(1 for p in positions if 0.25 <= p < 0.5) / len(positions)
    q3 = sum(1 for p in positions if 0.5 <= p < 0.75) / len(positions)
    q4 = sum(1 for p in positions if p >= 0.75) / len(positions)
    features.extend([q1, q2, q3, q4])

    # === CORRELATION FEATURES ===
    # Serial correlation (LCG detector)
    mean_pos = np.mean(positions)
    var_pos = np.var(positions)
    if var_pos > 0:
        serial_corr = np.mean([(positions[i] - mean_pos) * (positions[i+1] - mean_pos)
                               for i in range(len(positions)-1)]) / var_pos
    else:
        serial_corr = 0
    features.append(serial_corr)

    # Lag-2 correlation
    if var_pos > 0 and len(positions) > 2:
        lag2_corr = np.mean([(positions[i] - mean_pos) * (positions[i+2] - mean_pos)
                             for i in range(len(positions)-2)]) / var_pos
    else:
        lag2_corr = 0
    features.append(lag2_corr)

    # === XOR FEATURES ===
    # XOR between adjacent keys (normalized)
    xor_ratios = []
    for i in range(len(keys) - 1):
        max_bits = max(keys[i].bit_length(), keys[i+1].bit_length())
        if max_bits > 0:
            xor = keys[i] ^ keys[i+1]
            xor_ratios.append(bin(xor).count('1') / max_bits)
    features.append(np.mean(xor_ratios))
    features.append(np.std(xor_ratios))

    # === PATTERN FEATURES ===
    # Convert to bit stream and count 8-bit patterns
    all_bits = []
    for n, key in enumerate(keys, 1):
        for bit_pos in range(n):
            all_bits.append((key >> bit_pos) & 1)

    bit_str = ''.join(str(b) for b in all_bits)
    pattern_counts = Counter()
    for i in range(len(bit_str) - 7):
        pattern_counts[bit_str[i:i+8]] += 1

    # Top pattern frequencies (normalized)
    total_patterns = sum(pattern_counts.values())
    top_freqs = [count / total_patterns for _, count in pattern_counts.most_common(10)]
    while len(top_freqs) < 10:
        top_freqs.append(0)
    features.extend(top_freqs)

    # Specific patterns we identified as significant
    features.append(pattern_counts.get('11010110', 0) / total_patterns)  # 0xD6
    features.append(pattern_counts.get('10111010', 0) / total_patterns)  # 0xBA

    # === RUNS FEATURES ===
    # Count runs in bit stream
    runs = 1
    for i in range(1, len(all_bits)):
        if all_bits[i] != all_bits[i-1]:
            runs += 1
    expected_runs = len(all_bits) / 2
    features.append(runs / expected_runs)  # Runs ratio

    # === BYTE-LEVEL FEATURES ===
    # Byte value distribution
    byte_counts = Counter()
    for key in keys:
        temp = key
        while temp > 0:
            byte_counts[temp & 0xFF] += 1
            temp >>= 8

    total_bytes = sum(byte_counts.values())
    # Entropy of byte distribution
    probs = [count / total_bytes for count in byte_counts.values()]
    byte_entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    features.append(byte_entropy)

    # Specific byte frequencies
    features.append(byte_counts.get(0x01, 0) / total_bytes)
    features.append(byte_counts.get(0xFF, 0) / total_bytes)

    return np.array(features)


# =============================================================================
# TRAINING DATA GENERATION
# =============================================================================

def generate_training_data(n_samples_per_rng: int = 1000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate labeled training data from different RNGs."""

    rng_generators = {
        'mersenne_twister': mt19937_sequence,
        'lcg_glibc': lcg_glibc_sequence,
        'sha256_counter': sha256_counter_sequence,
        'xorshift128': xorshift128_sequence,
        'manual_random': manual_random_sequence,
        'mixed': mixed_rng_sequence,
    }

    X = []
    y = []
    labels = list(rng_generators.keys())

    print(f"Generating {n_samples_per_rng} samples per RNG type...")

    for label_idx, (name, generator) in enumerate(rng_generators.items()):
        print(f"  Generating {name}...")
        for seed in range(n_samples_per_rng):
            try:
                keys = generator(seed)
                features = extract_rng_features(keys)
                X.append(features)
                y.append(label_idx)
            except Exception as e:
                continue

    return np.array(X), np.array(y), labels


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_rng_classifier(X: np.ndarray, y: np.ndarray, labels: List[str]):
    """Train classifier to identify RNG type."""
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError:
        print("scikit-learn required")
        return None, None

    print("\n" + "=" * 70)
    print("TRAINING RNG CLASSIFIER")
    print("=" * 70)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = rf.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    # Cross-validation
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5)
    print(f"\nCross-validation accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")

    # Feature importance
    print("\nTop 20 Feature Importances:")
    feature_names = (
        [f'bit_{i}_freq' for i in range(40)] +
        ['pos_mean', 'pos_std', 'pos_median', 'q1', 'q2', 'q3', 'q4'] +
        ['serial_corr', 'lag2_corr'] +
        ['xor_mean', 'xor_std'] +
        [f'pattern_{i}' for i in range(10)] +
        ['pattern_0xD6', 'pattern_0xBA'] +
        ['runs_ratio', 'byte_entropy', 'byte_0x01', 'byte_0xFF']
    )

    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx[:20]:
        if i < len(feature_names):
            print(f"  {feature_names[i]:20s}: {importances[i]:.4f}")

    return rf, scaler


# =============================================================================
# CLASSIFY ACTUAL PUZZLE KEYS
# =============================================================================

def classify_puzzle_keys(model, scaler, labels: List[str]):
    """Classify the actual Bitcoin puzzle keys."""
    print("\n" + "=" * 70)
    print("CLASSIFYING ACTUAL PUZZLE KEYS")
    print("=" * 70)

    # Actual puzzle keys
    PUZZLE_KEYS = [
        0x1, 0x3, 0x7, 0x8, 0x15, 0x31, 0x4c, 0xe0,
        0x1d3, 0x202, 0x483, 0xa7b, 0x1460, 0x2930,
        0x68f3, 0xc936, 0x1764f, 0x3080d, 0x5749f,
        0xd2c55, 0x1ba534, 0x2de40f, 0x556e52, 0xdc2a04,
        0x1fa5ee5, 0x340326e, 0x6ac3875, 0xd916ce8,
        0x17e2551e, 0x3d94cd64, 0x7d4fe747, 0xb862a62e,
        0x1a96ca8d8, 0x34a65911d, 0x4aed21170, 0x9de820a7c,
        0x1757756a93, 0x22382facd0, 0x4b5f8303e9, 0xe9ae4933d6,
        0x153869acc5b, 0x2a221c58d8f, 0x6bd3b27c591, 0xe02b35a358f,
        0x122fca143c05, 0x2ec18388d544, 0x6cd610b53cba, 0xade6d7ce3b9b,
        0x174176b015f4d, 0x22bd43c2e9354, 0x75070a1a009d4, 0xefae164cb9e3c,
        0x180788e47e326c, 0x236fb6d5ad1f43, 0x6abe1f9b67e114, 0x9d18b63ac4ffdf,
        0x1eb25c90795d61c, 0x2c675b852189a21, 0x7496cbb87cab44f, 0xfc07a1825367bbe,
        0x13c96a3742f64906, 0x363d541eb611abee, 0x7cce5efdaccf6808, 0xf7051f27b09112d4,
        0x1a838b13505b26867, 0x2832ed74f2b5e35ee, 0x730fc235c1942c1ae,
        0xbebb3940cd0fc1491, 0x101d83275fb2bc7e0c, 0x349b84b6431a6c4ef1,
    ]

    # Extract features
    features = extract_rng_features(PUZZLE_KEYS)
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    print(f"\nPredicted RNG type: {labels[prediction]}")
    print("\nProbabilities:")
    for label, prob in sorted(zip(labels, probabilities), key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"  {label:20s}: {prob:.1%} {bar}")

    return labels[prediction], dict(zip(labels, probabilities))


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("RNG CLASSIFIER TRAINING")
    print("Learn to identify RNG type from output patterns")
    print("=" * 70)

    # Generate training data
    X, y, labels = generate_training_data(n_samples_per_rng=500)
    print(f"\nTraining data: {len(y)} samples, {len(labels)} classes")

    # Train classifier
    model, scaler = train_rng_classifier(X, y, labels)

    if model is not None:
        # Classify actual puzzle keys
        predicted, probs = classify_puzzle_keys(model, scaler, labels)

        print("\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        print(f"""
The ML classifier predicts the puzzle keys were generated by:
  {predicted}

This narrows our search from "any RNG" to a specific family.

If it's Mersenne Twister (Python random):
  - Focus on finding the seed (2^32 possibilities)
  - Check common seeds, timestamps, Bitcoin-related values

If it's SHA-based:
  - Look for the base string/seed used
  - Check common prefixes like "bitcoin", "puzzle", etc.

If it's LCG:
  - Very easy to crack once identified
  - Just need to find the initial state

If it's 'manual' or 'mixed':
  - Some human intervention
  - Harder to fully predict
  - But patterns in human choices are also exploitable
""")


if __name__ == "__main__":
    main()
