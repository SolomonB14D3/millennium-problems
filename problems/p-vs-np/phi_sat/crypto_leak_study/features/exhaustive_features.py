#!/usr/bin/env python3
"""
Exhaustive Feature Extraction

DAT Philosophy: Capture EVERYTHING. Then let ML tell us what matters.

For SHA256, at each of 64 rounds we have:
- 8 state words (a,b,c,d,e,f,g,h) = 256 bits
- Message schedule word (w) = 32 bits
- T1, T2 temporaries = 64 bits
- Previous state (for transitions) = 256 bits

Additional derived features:
- Hamming weight of each word
- Hamming distance from previous round
- Bit transitions (0→1 and 1→0 counts)
- XOR patterns between words
- Carry propagation signatures from additions

Total potential features per sample: ~50,000+
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.sha256_study import sha256_instrumented, SHA256Trace


@dataclass
class ExhaustiveFeatures:
    """All extractable features from one SHA256 execution."""

    # Raw bits
    output_bits: np.ndarray           # (256,)
    all_round_states: np.ndarray      # (64, 256) - all intermediate states as bits
    message_schedule_bits: np.ndarray # (64, 32) - w words as bits
    t1_bits: np.ndarray               # (64, 32) - T1 values as bits
    t2_bits: np.ndarray               # (64, 32) - T2 values as bits

    # Hamming features
    state_hamming: np.ndarray         # (64,) - hamming weight per round
    word_hamming: np.ndarray          # (64, 8) - hamming weight per word per round

    # Transition features
    bit_flips_per_round: np.ndarray   # (64,) - bits changed from previous round
    zero_to_one: np.ndarray           # (64,) - 0→1 transitions per round
    one_to_zero: np.ndarray           # (64,) - 1→0 transitions per round

    # Cross-word features
    xor_adjacent: np.ndarray          # (64, 7) - XOR hamming between adjacent words

    # Timing
    timing_ns: int

    def to_flat_array(self) -> np.ndarray:
        """Flatten all features into single vector."""
        features = [
            self.output_bits.flatten(),
            self.all_round_states.flatten(),
            self.message_schedule_bits.flatten(),
            self.t1_bits.flatten(),
            self.t2_bits.flatten(),
            self.state_hamming.flatten(),
            self.word_hamming.flatten(),
            self.bit_flips_per_round.flatten(),
            self.zero_to_one.flatten(),
            self.one_to_zero.flatten(),
            self.xor_adjacent.flatten(),
            np.array([self.timing_ns / 1e6]),  # normalized timing in ms
        ]
        return np.concatenate(features)

    @property
    def feature_count(self) -> int:
        return len(self.to_flat_array())


def int_to_bits(val: int, n_bits: int = 32) -> np.ndarray:
    """Convert integer to bit array (MSB first)."""
    return np.array([(val >> (n_bits - 1 - i)) & 1 for i in range(n_bits)], dtype=np.uint8)


def hamming_weight(bits: np.ndarray) -> int:
    """Count 1s in bit array."""
    return int(np.sum(bits))


def extract_exhaustive_features(trace: SHA256Trace) -> ExhaustiveFeatures:
    """Extract all possible features from a SHA256 trace."""

    n_rounds = len(trace.round_states)

    # Initialize arrays
    all_round_states = np.zeros((n_rounds, 256), dtype=np.uint8)
    message_schedule_bits = np.zeros((n_rounds, 32), dtype=np.uint8)
    t1_bits = np.zeros((n_rounds, 32), dtype=np.uint8)
    t2_bits = np.zeros((n_rounds, 32), dtype=np.uint8)
    state_hamming = np.zeros(n_rounds, dtype=np.int32)
    word_hamming = np.zeros((n_rounds, 8), dtype=np.int32)
    bit_flips_per_round = np.zeros(n_rounds, dtype=np.int32)
    zero_to_one = np.zeros(n_rounds, dtype=np.int32)
    one_to_zero = np.zeros(n_rounds, dtype=np.int32)
    xor_adjacent = np.zeros((n_rounds, 7), dtype=np.int32)

    prev_state_bits = None

    for i, state in enumerate(trace.round_states):
        # State bits
        state_bits = state.to_array()
        all_round_states[i] = state_bits

        # Message schedule
        message_schedule_bits[i] = int_to_bits(state.w)

        # T1, T2
        t1_bits[i] = int_to_bits(state.t1)
        t2_bits[i] = int_to_bits(state.t2)

        # Hamming weights
        state_hamming[i] = hamming_weight(state_bits)

        words = [state.a, state.b, state.c, state.d, state.e, state.f, state.g, state.h]
        for j, w in enumerate(words):
            word_hamming[i, j] = bin(w).count('1')

        # Transitions from previous round
        if prev_state_bits is not None:
            diff = state_bits != prev_state_bits
            bit_flips_per_round[i] = np.sum(diff)
            zero_to_one[i] = np.sum((prev_state_bits == 0) & (state_bits == 1))
            one_to_zero[i] = np.sum((prev_state_bits == 1) & (state_bits == 0))

        # XOR between adjacent words
        for j in range(7):
            xor_adjacent[i, j] = bin(words[j] ^ words[j+1]).count('1')

        prev_state_bits = state_bits.copy()

    return ExhaustiveFeatures(
        output_bits=trace.output_bits,
        all_round_states=all_round_states,
        message_schedule_bits=message_schedule_bits,
        t1_bits=t1_bits,
        t2_bits=t2_bits,
        state_hamming=state_hamming,
        word_hamming=word_hamming,
        bit_flips_per_round=bit_flips_per_round,
        zero_to_one=zero_to_one,
        one_to_zero=one_to_zero,
        xor_adjacent=xor_adjacent,
        timing_ns=trace.timing_ns,
    )


def generate_exhaustive_dataset(
    num_samples: int,
    input_bits: int
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate dataset with exhaustive features.

    Returns:
        inputs: (num_samples, input_bits)
        features: (num_samples, num_features)
        feature_names: list of feature names
    """
    input_bytes = (input_bits + 7) // 8

    # First pass to get feature count
    test_data = np.random.bytes(input_bytes)
    test_trace = sha256_instrumented(test_data)
    test_features = extract_exhaustive_features(test_trace)
    num_features = test_features.feature_count

    print(f"Extracting {num_features} features per sample...")

    inputs = np.zeros((num_samples, input_bits), dtype=np.uint8)
    features = np.zeros((num_samples, num_features), dtype=np.float32)

    for i in range(num_samples):
        # Generate input
        data = bytearray(np.random.bytes(input_bytes))
        # Mask to exact bit length
        if input_bits % 8 != 0:
            mask = (1 << (input_bits % 8)) - 1
            data[0] &= mask
        data = bytes(data)

        # Run SHA256 with instrumentation
        trace = sha256_instrumented(data)

        # Extract features
        feat = extract_exhaustive_features(trace)

        # Store
        input_bits_arr = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        inputs[i, :min(len(input_bits_arr), input_bits)] = input_bits_arr[:input_bits]
        features[i] = feat.to_flat_array()

        if (i + 1) % max(1, num_samples // 10) == 0:
            print(f"  {i+1}/{num_samples}")

    # Generate feature names
    feature_names = _generate_feature_names(test_features)

    return inputs, features, feature_names


def _generate_feature_names(feat: ExhaustiveFeatures) -> List[str]:
    """Generate descriptive names for each feature."""
    names = []

    # Output bits
    for i in range(256):
        names.append(f"output_bit_{i}")

    # Round states
    for r in range(64):
        for b in range(256):
            names.append(f"round_{r}_state_bit_{b}")

    # Message schedule
    for r in range(64):
        for b in range(32):
            names.append(f"round_{r}_w_bit_{b}")

    # T1
    for r in range(64):
        for b in range(32):
            names.append(f"round_{r}_t1_bit_{b}")

    # T2
    for r in range(64):
        for b in range(32):
            names.append(f"round_{r}_t2_bit_{b}")

    # Hamming
    for r in range(64):
        names.append(f"round_{r}_hamming")

    for r in range(64):
        for w in range(8):
            names.append(f"round_{r}_word_{w}_hamming")

    # Transitions
    for r in range(64):
        names.append(f"round_{r}_bit_flips")
    for r in range(64):
        names.append(f"round_{r}_zero_to_one")
    for r in range(64):
        names.append(f"round_{r}_one_to_zero")

    # XOR
    for r in range(64):
        for j in range(7):
            names.append(f"round_{r}_xor_word_{j}_{j+1}")

    # Timing
    names.append("timing_ms")

    return names


class ExhaustiveTrainer:
    """Train with exhaustive features and identify what matters."""

    def __init__(self):
        self.feature_importance = {}

    def train_and_analyze(
        self,
        inputs: np.ndarray,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Train models and identify most important features.
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import StandardScaler

        num_samples, num_input_bits = inputs.shape

        # Normalize features
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
            'bit_results': [],
            'feature_importance': {},
        }

        # Train per-bit with feature importance tracking
        all_importances = np.zeros(features.shape[1])

        for bit_idx in range(num_input_bits):
            y_train_bit = y_train[:, bit_idx]
            y_test_bit = y_test[:, bit_idx]

            if len(np.unique(y_train_bit)) < 2:
                results['bit_results'].append({
                    'bit': bit_idx,
                    'accuracy': 0.5,
                    'model': 'skip'
                })
                continue

            # Random Forest (gives feature importance)
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train_bit)
            rf_pred = rf.predict(X_test)
            rf_acc = accuracy_score(y_test_bit, rf_pred)

            # Track feature importance
            all_importances += rf.feature_importances_

            # Gradient Boosting for comparison
            gb = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            gb.fit(X_train, y_train_bit)
            gb_pred = gb.predict(X_test)
            gb_acc = accuracy_score(y_test_bit, gb_pred)

            best_acc = max(rf_acc, gb_acc)
            best_model = 'rf' if rf_acc >= gb_acc else 'gb'

            results['bit_results'].append({
                'bit': bit_idx,
                'accuracy': best_acc,
                'rf_accuracy': rf_acc,
                'gb_accuracy': gb_acc,
                'model': best_model
            })

            print(f"  Bit {bit_idx}: RF={rf_acc:.3f}, GB={gb_acc:.3f}")

        # Normalize and rank feature importance
        all_importances /= num_input_bits
        top_indices = np.argsort(all_importances)[::-1][:50]

        results['top_features'] = [
            (feature_names[i], float(all_importances[i]))
            for i in top_indices
        ]

        # Overall accuracy
        accuracies = [r['accuracy'] for r in results['bit_results']]
        results['mean_accuracy'] = np.mean(accuracies)
        results['min_accuracy'] = np.min(accuracies)
        results['max_accuracy'] = np.max(accuracies)
        results['leak_strength'] = results['mean_accuracy'] - 0.5

        return results


def run_exhaustive_analysis(input_bits: int = 8, num_samples: int = 2000):
    """Run full exhaustive analysis at given bit length."""

    print("="*70)
    print(f"EXHAUSTIVE FEATURE ANALYSIS - {input_bits} BITS")
    print("="*70)

    # Generate data
    print("\nGenerating dataset with exhaustive features...")
    inputs, features, feature_names = generate_exhaustive_dataset(num_samples, input_bits)

    print(f"\nDataset shape: {inputs.shape[0]} samples")
    print(f"Input bits: {inputs.shape[1]}")
    print(f"Features per sample: {features.shape[1]}")

    # Train and analyze
    print("\nTraining models...")
    trainer = ExhaustiveTrainer()
    results = trainer.train_and_analyze(inputs, features, feature_names)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Mean accuracy: {results['mean_accuracy']:.4f}")
    print(f"Min accuracy:  {results['min_accuracy']:.4f}")
    print(f"Max accuracy:  {results['max_accuracy']:.4f}")
    print(f"Leak strength: {results['leak_strength']:+.4f}")

    print("\nTop 20 most important features:")
    for i, (name, importance) in enumerate(results['top_features'][:20]):
        print(f"  {i+1:2d}. {name}: {importance:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--samples', type=int, default=2000)
    args = parser.parse_args()

    run_exhaustive_analysis(args.bits, args.samples)
