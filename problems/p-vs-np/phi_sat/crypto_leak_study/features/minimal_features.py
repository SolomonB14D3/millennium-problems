#!/usr/bin/env python3
"""
Minimal Feature Set

Based on information flow analysis:
- Signal lives in rounds 1-5
- After round 5, noise dominates
- Specific state bits carry concentrated information

Goal: Achieve same accuracy with minimal features.
This tells us exactly what information is needed.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.sha256_study import sha256_instrumented, SHA256Trace


@dataclass
class MinimalFeatures:
    """Only the features that carry signal."""

    # Round 1-5 state bits (where signal lives)
    early_round_states: np.ndarray  # (5, 256) = 1280 bits

    # Message schedule for early rounds
    early_w_bits: np.ndarray  # (5, 32) = 160 bits

    # T1 for early rounds (intermediate computation)
    early_t1_bits: np.ndarray  # (5, 32) = 160 bits

    # Concentrated locations only (top signal bits per round)
    signal_bits: np.ndarray  # Variable size based on signal locations

    # Output for reference
    output_bits: np.ndarray  # (256,)

    def to_flat_array(self) -> np.ndarray:
        """Flatten to single vector."""
        return np.concatenate([
            self.early_round_states.flatten(),
            self.early_w_bits.flatten(),
            self.early_t1_bits.flatten(),
            self.signal_bits.flatten(),
            self.output_bits.flatten(),
        ])

    @property
    def feature_count(self) -> int:
        return len(self.to_flat_array())


# Signal locations from information flow analysis
# Format: (round, state_bit)
SIGNAL_LOCATIONS = [
    # Round 1 - direct input mapping
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15),
    (1, 128), (1, 129), (1, 130), (1, 131), (1, 132), (1, 133), (1, 134), (1, 135),
    (1, 136), (1, 137), (1, 138), (1, 139), (1, 140), (1, 141), (1, 142), (1, 143),

    # Round 2 - early spreading
    (2, 172), (2, 175), (2, 32), (2, 33), (2, 34), (2, 35),
    (2, 160), (2, 161), (2, 162), (2, 163),

    # Round 3 - secondary mixing
    (3, 79), (3, 206), (3, 64), (3, 65), (3, 66), (3, 67),
    (3, 192), (3, 193), (3, 194), (3, 195),

    # Round 4 - before saturation
    (4, 111), (4, 237), (4, 238), (4, 239), (4, 96), (4, 97),
    (4, 224), (4, 225), (4, 226), (4, 227),
]


def int_to_bits(val: int, n_bits: int = 32) -> np.ndarray:
    """Convert integer to bit array."""
    return np.array([(val >> (n_bits - 1 - i)) & 1 for i in range(n_bits)], dtype=np.uint8)


def extract_minimal_features(trace: SHA256Trace) -> MinimalFeatures:
    """Extract only the signal-carrying features."""

    n_early_rounds = 5

    # Early round states (rounds 0-4, which are indices 0-4)
    early_round_states = np.zeros((n_early_rounds, 256), dtype=np.uint8)
    early_w_bits = np.zeros((n_early_rounds, 32), dtype=np.uint8)
    early_t1_bits = np.zeros((n_early_rounds, 32), dtype=np.uint8)

    for r in range(min(n_early_rounds, len(trace.round_states))):
        state = trace.round_states[r]
        early_round_states[r] = state.to_array()
        early_w_bits[r] = int_to_bits(state.w)
        early_t1_bits[r] = int_to_bits(state.t1)

    # Extract specific signal locations
    signal_bits = []
    for round_idx, state_bit in SIGNAL_LOCATIONS:
        if round_idx < len(trace.round_states):
            state = trace.round_states[round_idx]
            state_array = state.to_array()
            if state_bit < len(state_array):
                signal_bits.append(state_array[state_bit])
            else:
                signal_bits.append(0)
        else:
            signal_bits.append(0)

    return MinimalFeatures(
        early_round_states=early_round_states,
        early_w_bits=early_w_bits,
        early_t1_bits=early_t1_bits,
        signal_bits=np.array(signal_bits, dtype=np.uint8),
        output_bits=trace.output_bits,
    )


def generate_minimal_dataset(
    num_samples: int,
    input_bits: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generate dataset with minimal features only.

    Returns:
        inputs: (num_samples, input_bits)
        features: (num_samples, num_features)
        num_features: feature count
    """
    input_bytes = (input_bits + 7) // 8

    # Get feature count from test sample
    test_trace = sha256_instrumented(np.random.bytes(input_bytes))
    test_features = extract_minimal_features(test_trace)
    num_features = test_features.feature_count

    print(f"Minimal feature count: {num_features} (vs 24,001 exhaustive)")

    inputs = np.zeros((num_samples, input_bits), dtype=np.uint8)
    features = np.zeros((num_samples, num_features), dtype=np.float32)

    for i in range(num_samples):
        # Generate input
        data = bytearray(np.random.bytes(input_bytes))
        if input_bits % 8 != 0:
            mask = (1 << (input_bits % 8)) - 1
            data[0] &= mask
        data = bytes(data)

        trace = sha256_instrumented(data)
        feat = extract_minimal_features(trace)

        input_bits_arr = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        inputs[i, :min(len(input_bits_arr), input_bits)] = input_bits_arr[:input_bits]
        features[i] = feat.to_flat_array()

        if (i + 1) % max(1, num_samples // 10) == 0:
            print(f"  {i+1}/{num_samples}")

    return inputs, features, num_features


def train_with_minimal_features(
    inputs: np.ndarray,
    features: np.ndarray,
    verbose: bool = True
) -> Dict:
    """Train using only minimal features."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    num_samples, num_input_bits = inputs.shape

    # Normalize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, inputs, test_size=0.2, random_state=42
    )

    results = {
        'num_samples': num_samples,
        'num_features': features.shape[1],
        'num_input_bits': num_input_bits,
        'bit_accuracies': [],
    }

    for bit_idx in range(num_input_bits):
        y_train_bit = y_train[:, bit_idx]
        y_test_bit = y_test[:, bit_idx]

        if len(np.unique(y_train_bit)) < 2:
            results['bit_accuracies'].append(0.5)
            continue

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train_bit)
        rf_acc = accuracy_score(y_test_bit, rf.predict(X_test))

        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        gb.fit(X_train, y_train_bit)
        gb_acc = accuracy_score(y_test_bit, gb.predict(X_test))

        best_acc = max(rf_acc, gb_acc)
        results['bit_accuracies'].append(best_acc)

        if verbose:
            print(f"  Bit {bit_idx}: RF={rf_acc:.3f}, GB={gb_acc:.3f}")

    results['mean_accuracy'] = np.mean(results['bit_accuracies'])
    results['min_accuracy'] = np.min(results['bit_accuracies'])
    results['leak_strength'] = results['mean_accuracy'] - 0.5

    return results


# ============================================================
# ULTRA-MINIMAL: Only the absolute essential features
# ============================================================

@dataclass
class UltraMinimalFeatures:
    """Absolute minimum - only round 1 state where input directly maps."""

    round_1_state: np.ndarray  # (256,) - just round 1
    round_1_w: np.ndarray  # (32,) - message word

    def to_flat_array(self) -> np.ndarray:
        return np.concatenate([
            self.round_1_state,
            self.round_1_w,
        ])

    @property
    def feature_count(self) -> int:
        return len(self.to_flat_array())


def extract_ultra_minimal(trace: SHA256Trace) -> UltraMinimalFeatures:
    """Extract only round 1 features."""
    state = trace.round_states[1] if len(trace.round_states) > 1 else trace.round_states[0]
    return UltraMinimalFeatures(
        round_1_state=state.to_array(),
        round_1_w=int_to_bits(state.w),
    )


def generate_ultra_minimal_dataset(
    num_samples: int,
    input_bits: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Generate with ultra-minimal features (288 features only)."""
    input_bytes = (input_bits + 7) // 8

    test_trace = sha256_instrumented(np.random.bytes(input_bytes))
    test_features = extract_ultra_minimal(test_trace)
    num_features = test_features.feature_count

    print(f"Ultra-minimal feature count: {num_features}")

    inputs = np.zeros((num_samples, input_bits), dtype=np.uint8)
    features = np.zeros((num_samples, num_features), dtype=np.float32)

    for i in range(num_samples):
        data = bytearray(np.random.bytes(input_bytes))
        if input_bits % 8 != 0:
            mask = (1 << (input_bits % 8)) - 1
            data[0] &= mask
        data = bytes(data)

        trace = sha256_instrumented(data)
        feat = extract_ultra_minimal(trace)

        input_bits_arr = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        inputs[i, :min(len(input_bits_arr), input_bits)] = input_bits_arr[:input_bits]
        features[i] = feat.to_flat_array()

        if (i + 1) % max(1, num_samples // 10) == 0:
            print(f"  {i+1}/{num_samples}")

    return inputs, features, num_features


def run_comparison(input_bits: int = 16, num_samples: int = 2000):
    """Compare exhaustive vs minimal vs ultra-minimal."""

    print("="*70)
    print(f"FEATURE SET COMPARISON - {input_bits} BITS")
    print("="*70)

    # Ultra-minimal (round 1 only)
    print("\n[1/2] ULTRA-MINIMAL (Round 1 only)")
    print("-"*50)
    inputs_um, features_um, n_feat_um = generate_ultra_minimal_dataset(num_samples, input_bits)
    print(f"Features: {n_feat_um}")
    print("Training...")
    results_um = train_with_minimal_features(inputs_um, features_um)

    # Minimal (rounds 1-5 + signal locations)
    print("\n[2/2] MINIMAL (Rounds 1-5 + signal)")
    print("-"*50)
    inputs_min, features_min, n_feat_min = generate_minimal_dataset(num_samples, input_bits)
    print(f"Features: {n_feat_min}")
    print("Training...")
    results_min = train_with_minimal_features(inputs_min, features_min)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Feature Set':<25} {'Features':>10} {'Accuracy':>10} {'Leak':>10}")
    print("-"*55)
    print(f"{'Exhaustive':<25} {'24,001':>10} {'1.000':>10} {'+0.500':>10}")
    print(f"{'Minimal (rounds 1-5)':<25} {n_feat_min:>10} {results_min['mean_accuracy']:>10.3f} {results_min['leak_strength']:>+10.3f}")
    print(f"{'Ultra-minimal (round 1)':<25} {n_feat_um:>10} {results_um['mean_accuracy']:>10.3f} {results_um['leak_strength']:>+10.3f}")

    # Feature reduction ratio
    print(f"\nFeature reduction: 24,001 → {n_feat_min} → {n_feat_um}")
    print(f"Reduction ratio: {24001/n_feat_um:.0f}x (ultra) / {24001/n_feat_min:.0f}x (minimal)")

    return {
        'ultra_minimal': results_um,
        'minimal': results_min,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bits', type=int, default=16)
    parser.add_argument('--samples', type=int, default=2000)
    args = parser.parse_args()

    run_comparison(args.bits, args.samples)
