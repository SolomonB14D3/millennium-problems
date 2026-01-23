#!/usr/bin/env python3
"""
ML-Based Puzzle Pattern Finding

The CRT analysis proved keys aren't simple HD wallet.
But there MIGHT be a pattern ML can find that humans can't.

Approach:
1. Use features from known puzzles (1-40)
2. Train to predict key bits from puzzle number
3. See if patterns emerge that could help with 66+
"""

import numpy as np
from typing import List, Tuple
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Known puzzle keys
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


def build_features(puzzle_num: int, key: int) -> np.ndarray:
    """Build feature vector for a puzzle."""
    features = []

    # Basic features
    features.append(puzzle_num)
    features.append(puzzle_num ** 2)
    features.append(np.log2(puzzle_num))

    # Key-derived features (normalized)
    expected_bits = puzzle_num
    actual_bits = key.bit_length()
    features.append(actual_bits / expected_bits)

    # Position of highest unforced bit (after bit puzzle_num-1)
    low_bits = key & ((1 << (puzzle_num - 1)) - 1)
    features.append(low_bits.bit_length() / (puzzle_num - 1) if puzzle_num > 1 else 0)

    # Low bit patterns
    for i in range(8):
        features.append((key >> i) & 1)

    # Bit density (fraction of 1s)
    density = bin(key).count('1') / puzzle_num
    features.append(density)

    # Ratio to max possible value
    max_val = (1 << puzzle_num) - 1
    min_val = 1 << (puzzle_num - 1)
    features.append((key - min_val) / (max_val - min_val) if max_val > min_val else 0)

    return np.array(features)


def build_target_features(puzzle_num: int, key: int) -> np.ndarray:
    """Build target (what we want to predict) for a puzzle."""
    # Target: the "position" within the N-bit space
    # Normalized to [0, 1]
    min_val = 1 << (puzzle_num - 1)
    max_val = (1 << puzzle_num) - 1
    position = (key - min_val) / (max_val - min_val) if max_val > min_val else 0
    return position


def train_position_predictor():
    """Train model to predict key position from puzzle number."""
    print("="*70)
    print("POSITION PREDICTOR")
    print("Predicting where in [2^(N-1), 2^N) the key falls")
    print("="*70)

    X = []
    y = []

    for p, key in sorted(PUZZLE_KEYS.items()):
        feat = [
            p,
            p ** 2,
            np.log2(p),
            np.sin(p),
            np.cos(p),
            p % 7,
            p % 13,
        ]
        X.append(feat)

        pos = build_target_features(p, key)
        y.append(pos)

    X = np.array(X)
    y = np.array(y)

    print(f"\nTraining data: {len(X)} puzzles")
    print(f"Features: {X.shape[1]}")

    # Leave-one-out cross validation
    model = HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=42)

    loo = LeaveOneOut()
    predictions = []
    actuals = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        predictions.append(pred)
        actuals.append(y_test[0])

    # Evaluate
    mse = mean_squared_error(actuals, predictions)
    print(f"\nLeave-one-out MSE: {mse:.4f}")
    print(f"Random baseline MSE: ~0.083 (uniform distribution)")

    # Show predictions vs actual
    print(f"\n{'Puzzle':<8} {'Actual Pos':<12} {'Predicted':<12} {'Error':<12}")
    print("-"*45)
    for i, (p, key) in enumerate(sorted(PUZZLE_KEYS.items())):
        print(f"{p:<8} {actuals[i]:<12.4f} {predictions[i]:<12.4f} {abs(actuals[i]-predictions[i]):<12.4f}")

    # Train on all data and predict puzzle 66
    print("\n" + "="*70)
    print("PREDICTION FOR PUZZLE 66")
    print("="*70)

    model.fit(X, y)

    feat_66 = [66, 66**2, np.log2(66), np.sin(66), np.cos(66), 66 % 7, 66 % 13]
    pred_pos = model.predict([feat_66])[0]

    print(f"\nPredicted position in [2^65, 2^66): {pred_pos:.4f}")

    min_val = 1 << 65
    max_val = (1 << 66) - 1
    pred_key = int(min_val + pred_pos * (max_val - min_val))

    print(f"Predicted key (approximate center): {hex(pred_key)}")
    print(f"Search range: {hex(int(pred_key * 0.9))} to {hex(int(pred_key * 1.1))}")

    return model, mse


def train_bit_predictors():
    """Train model to predict individual bits from puzzle number."""
    print("\n" + "="*70)
    print("BIT PREDICTORS")
    print("Predicting individual bit values from puzzle number")
    print("="*70)

    # Focus on bits that are common across puzzles
    # Bit 0 (LSB) exists in all puzzles
    # Bit 7 exists in puzzles 8+

    for bit_pos in [0, 1, 2, 3, 4, 5, 6, 7]:
        X = []
        y = []

        min_puzzle = bit_pos + 2  # Need puzzle to have this bit

        for p, key in sorted(PUZZLE_KEYS.items()):
            if p < min_puzzle:
                continue

            feat = [
                p,
                p ** 2,
                np.log2(p),
                p % 7,
                p % 13,
            ]
            X.append(feat)

            bit_val = (key >> bit_pos) & 1
            y.append(bit_val)

        if len(X) < 10:
            continue

        X = np.array(X)
        y = np.array(y)

        # Cross validation
        model = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=42)

        try:
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            mean_acc = np.mean(scores)

            print(f"Bit {bit_pos}: accuracy = {mean_acc:.2%} (random = 50%)")

            if mean_acc > 0.6:
                print(f"  → ABOVE CHANCE! Bit {bit_pos} has pattern")

                # Predict for puzzle 66
                model.fit(X, y)
                feat_66 = [66, 66**2, np.log2(66), 66 % 7, 66 % 13]
                pred = model.predict([feat_66])[0]
                proba = model.predict_proba([feat_66])[0]
                print(f"  → Puzzle 66 bit {bit_pos} prediction: {pred} (confidence: {max(proba):.2%})")

        except Exception as e:
            print(f"Bit {bit_pos}: error - {e}")


def analyze_segment_transition():
    """Analyze the transition from RNG (1-13) to unknown (14+)."""
    print("\n" + "="*70)
    print("SEGMENT TRANSITION ANALYSIS")
    print("="*70)

    # RNG segment: confirmed Python MT (1-13)
    # Unknown segment: possibly HD wallet (14+)

    # Feature: normalized position within bit space
    rng_positions = []
    unk_positions = []

    for p, key in PUZZLE_KEYS.items():
        min_val = 1 << (p - 1)
        max_val = (1 << p) - 1
        pos = (key - min_val) / (max_val - min_val)

        if p <= 13:
            rng_positions.append(pos)
        else:
            unk_positions.append(pos)

    print(f"\nRNG segment (1-13) positions:")
    print(f"  Mean: {np.mean(rng_positions):.3f}")
    print(f"  Std: {np.std(rng_positions):.3f}")
    print(f"  Min: {np.min(rng_positions):.3f}, Max: {np.max(rng_positions):.3f}")

    print(f"\nUnknown segment (14+) positions:")
    print(f"  Mean: {np.mean(unk_positions):.3f}")
    print(f"  Std: {np.std(unk_positions):.3f}")
    print(f"  Min: {np.min(unk_positions):.3f}, Max: {np.max(unk_positions):.3f}")

    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(rng_positions, unk_positions)
    print(f"\nT-test: t={t_stat:.3f}, p={p_value:.3f}")

    if p_value < 0.05:
        print("→ Segments are SIGNIFICANTLY different!")
    else:
        print("→ Segments not significantly different")


def predict_puzzle_66():
    """Combine all insights to make predictions for puzzle 66."""
    print("\n" + "="*70)
    print("COMBINED PREDICTION FOR PUZZLE 66")
    print("="*70)

    # Known: puzzle 66 key is in [2^65, 2^66 - 1]
    # That's about 3.6 × 10^19 possible values

    min_val = 1 << 65
    max_val = (1 << 66) - 1
    range_size = max_val - min_val

    print(f"\nPuzzle 66 search space: {range_size:,} values")
    print(f"That's approximately 3.7 × 10^19 values")

    # Based on our analysis:
    # 1. Unknown segment has std ~0.26 vs RNG std ~0.27 (similar)
    # 2. Mean position ~0.48 (slightly below center)
    # 3. No strong bit patterns found

    # Conservative estimate: key is somewhere in the range
    # Without more structure, we can't narrow it down significantly

    print("\nBased on analysis:")
    print("1. Position distribution suggests key is near center of range")
    print("2. No exploitable patterns found in solved puzzles")
    print("3. CRT analysis proves keys aren't simple HD wallet")
    print("4. Structure exists but may not be exploitable")

    # What would help:
    print("\nWhat would help:")
    print("1. More solved puzzles (especially 41-65)")
    print("2. Knowledge of the deterministic wallet scheme used")
    print("3. Side-channel information (timing, etc.)")


if __name__ == "__main__":
    train_position_predictor()
    train_bit_predictors()
    analyze_segment_transition()
    predict_puzzle_66()
