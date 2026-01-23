#!/usr/bin/env python3
"""
Transformer-based Puzzle Analysis

Use attention mechanism to find which bits "attend to" which.
This reveals the non-linear relationships trees are finding.

Also: train on solved puzzles, predict higher ones.
"""

import numpy as np
from typing import List, Tuple, Dict
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


def key_to_bits(key: int, num_bits: int) -> np.ndarray:
    """Convert key to bit array (LSB first)."""
    bits = np.zeros(num_bits, dtype=np.float32)
    for i in range(min(num_bits, key.bit_length())):
        bits[i] = (key >> i) & 1
    return bits


def build_puzzle_dataset():
    """Build dataset from solved puzzles."""
    X = []  # Features
    y = []  # Targets

    max_bits = 40  # Common bit length

    for puzzle_num, key in sorted(PUZZLE_KEYS.items()):
        # Feature: puzzle number encoded + previous key bits (if available)
        feat = np.zeros(max_bits + 10, dtype=np.float32)

        # Puzzle number features
        feat[0] = puzzle_num / 40.0  # Normalized
        feat[1] = np.sin(puzzle_num * np.pi / 20)
        feat[2] = np.cos(puzzle_num * np.pi / 20)
        feat[3] = (puzzle_num % 7) / 7.0
        feat[4] = (puzzle_num % 13) / 13.0

        # Previous puzzle bits (if available)
        if puzzle_num > 1 and (puzzle_num - 1) in PUZZLE_KEYS:
            prev_key = PUZZLE_KEYS[puzzle_num - 1]
            prev_bits = key_to_bits(prev_key, max_bits)
            feat[10:10+max_bits] = prev_bits

        X.append(feat)

        # Target: current key's normalized position
        min_val = 1 << (puzzle_num - 1)
        max_val = (1 << puzzle_num) - 1
        pos = (key - min_val) / (max_val - min_val) if max_val > min_val else 0
        y.append(pos)

    return np.array(X), np.array(y)


def train_position_transformer():
    """Train a small transformer to predict key positions."""
    print("="*70)
    print("TRANSFORMER-BASED POSITION PREDICTION")
    print("="*70)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("PyTorch not available")
        return None

    X, y = build_puzzle_dataset()
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    print(f"\nDataset: {len(X)} puzzles, {X.shape[1]} features")

    # Simple transformer encoder
    class PositionPredictor(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.output = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            # Add sequence dimension (treat each feature as a token)
            x = self.input_proj(x)
            x = x.unsqueeze(1)  # (batch, 1, hidden)
            x = self.transformer(x)
            x = x.squeeze(1)
            return self.output(x).squeeze(-1)

    model = PositionPredictor(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Train with leave-one-out
    print("\nTraining with leave-one-out validation...")

    loo_predictions = []

    for i in range(len(X)):
        # Leave one out
        mask = np.ones(len(X), dtype=bool)
        mask[i] = False

        X_train = X[mask]
        y_train = y[mask]
        X_test = X[i:i+1]

        # Reset model
        model = PositionPredictor(X.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Train
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            pred = model(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()

        # Predict
        model.eval()
        with torch.no_grad():
            pred = model(X_test).item()

        loo_predictions.append(pred)

    # Evaluate
    loo_predictions = np.array(loo_predictions)
    y_np = y.numpy()

    mse = np.mean((loo_predictions - y_np) ** 2)
    print(f"\nLeave-one-out MSE: {mse:.4f}")
    print(f"Random baseline MSE: ~0.083")

    # Show predictions
    print(f"\n{'Puzzle':<8} {'Actual':<10} {'Predicted':<10} {'Error':<10}")
    print("-"*40)
    for i, (p, key) in enumerate(sorted(PUZZLE_KEYS.items())):
        print(f"{p:<8} {y_np[i]:<10.4f} {loo_predictions[i]:<10.4f} {abs(y_np[i]-loo_predictions[i]):<10.4f}")

    # Train on all and predict puzzle 66
    print("\n" + "="*70)
    print("PREDICTING PUZZLE 66")
    print("="*70)

    model = PositionPredictor(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

    # Create feature for puzzle 66
    feat_66 = np.zeros(X.shape[1], dtype=np.float32)
    feat_66[0] = 66 / 40.0
    feat_66[1] = np.sin(66 * np.pi / 20)
    feat_66[2] = np.cos(66 * np.pi / 20)
    feat_66[3] = (66 % 7) / 7.0
    feat_66[4] = (66 % 13) / 13.0

    # Use puzzle 40's bits as "previous" (approximation)
    prev_bits = key_to_bits(PUZZLE_KEYS[40], 40)
    feat_66[10:10+40] = prev_bits

    model.eval()
    with torch.no_grad():
        pred_66 = model(torch.FloatTensor(feat_66).unsqueeze(0)).item()

    print(f"\nPredicted position for puzzle 66: {pred_66:.4f}")

    min_val = 1 << 65
    max_val = (1 << 66) - 1
    pred_key = int(min_val + pred_66 * (max_val - min_val))

    print(f"Predicted key: {hex(pred_key)}")

    return model


def analyze_bit_attention():
    """Analyze which bits influence predictions most."""
    print("\n" + "="*70)
    print("BIT INFLUENCE ANALYSIS")
    print("="*70)

    from sklearn.ensemble import HistGradientBoostingRegressor

    X, y = build_puzzle_dataset()

    # Train gradient boosting
    model = HistGradientBoostingRegressor(max_iter=100, max_depth=4)
    model.fit(X, y)

    # Analyze feature importance via permutation
    base_pred = model.predict(X)
    base_mse = np.mean((base_pred - y) ** 2)

    importances = []
    for i in range(X.shape[1]):
        X_perm = X.copy()
        np.random.shuffle(X_perm[:, i])
        perm_pred = model.predict(X_perm)
        perm_mse = np.mean((perm_pred - y) ** 2)
        importance = perm_mse - base_mse
        importances.append(importance)

    importances = np.array(importances)

    print("\nFeature importance (permutation-based):")
    feature_names = ['puzzle_num', 'sin', 'cos', 'mod7', 'mod13'] + [f'padding_{i}' for i in range(5)] + [f'prev_bit_{i}' for i in range(40)]

    top_features = np.argsort(importances)[::-1][:15]
    for idx in top_features:
        if importances[idx] > 0.001:
            name = feature_names[idx] if idx < len(feature_names) else f'feat_{idx}'
            print(f"  {name}: {importances[idx]:.4f}")


def cross_puzzle_correlation():
    """Find correlations between bits across different puzzles."""
    print("\n" + "="*70)
    print("CROSS-PUZZLE BIT CORRELATIONS")
    print("="*70)

    # Build bit matrix: each row is a puzzle, each column is a bit position
    max_bits = 40
    bit_matrix = np.zeros((40, max_bits))

    for p in range(1, 41):
        key = PUZZLE_KEYS[p]
        for b in range(min(p, max_bits)):
            bit_matrix[p-1, b] = (key >> b) & 1

    # Convert to +1/-1 for correlation
    bit_matrix = 2 * bit_matrix - 1

    # Compute correlation matrix for bit positions
    # Only consider bits that exist in multiple puzzles
    print("\nBit-to-bit correlations (across puzzles that have both bits):")

    for b1 in range(8):
        for b2 in range(b1+1, 8):
            # Get puzzles that have both bits
            valid_puzzles = [p for p in range(1, 41) if p > max(b1, b2)]

            if len(valid_puzzles) < 10:
                continue

            vec1 = [bit_matrix[p-1, b1] for p in valid_puzzles]
            vec2 = [bit_matrix[p-1, b2] for p in valid_puzzles]

            corr = np.corrcoef(vec1, vec2)[0, 1]

            if abs(corr) > 0.3:
                print(f"  Bit {b1} vs Bit {b2}: corr = {corr:.3f}")


def train_on_solved_predict_higher():
    """
    Train models on solved puzzles, test generalization.

    Split: train on puzzles 1-30, test on 31-40.
    """
    print("\n" + "="*70)
    print("GENERALIZATION TEST: Train on 1-30, Test on 31-40")
    print("="*70)

    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

    # Build features for each puzzle
    def build_puzzle_features(puzzle_num: int, prev_key: int = None) -> np.ndarray:
        feat = [
            puzzle_num,
            puzzle_num ** 2,
            np.log2(puzzle_num),
            np.sin(puzzle_num),
            np.cos(puzzle_num),
            puzzle_num % 7,
            puzzle_num % 13,
        ]
        if prev_key is not None:
            for b in range(16):
                feat.append((prev_key >> b) & 1)
        else:
            feat.extend([0] * 16)
        return np.array(feat)

    # Train data: puzzles 1-30
    X_train = []
    y_train_pos = []  # Normalized position
    y_train_bits = []  # Low 8 bits

    for p in range(1, 31):
        key = PUZZLE_KEYS[p]
        prev_key = PUZZLE_KEYS.get(p-1, 0)
        X_train.append(build_puzzle_features(p, prev_key))

        min_val = 1 << (p - 1)
        max_val = (1 << p) - 1
        pos = (key - min_val) / (max_val - min_val) if max_val > min_val else 0
        y_train_pos.append(pos)

        y_train_bits.append([(key >> b) & 1 for b in range(8)])

    X_train = np.array(X_train)
    y_train_pos = np.array(y_train_pos)
    y_train_bits = np.array(y_train_bits)

    # Test data: puzzles 31-40
    X_test = []
    y_test_pos = []
    y_test_bits = []

    for p in range(31, 41):
        key = PUZZLE_KEYS[p]
        prev_key = PUZZLE_KEYS.get(p-1, 0)
        X_test.append(build_puzzle_features(p, prev_key))

        min_val = 1 << (p - 1)
        max_val = (1 << p) - 1
        pos = (key - min_val) / (max_val - min_val) if max_val > min_val else 0
        y_test_pos.append(pos)

        y_test_bits.append([(key >> b) & 1 for b in range(8)])

    X_test = np.array(X_test)
    y_test_pos = np.array(y_test_pos)
    y_test_bits = np.array(y_test_bits)

    # Train position predictor
    print("\nPosition prediction:")
    model_pos = HistGradientBoostingRegressor(max_iter=100, max_depth=4)
    model_pos.fit(X_train, y_train_pos)

    train_pred = model_pos.predict(X_train)
    test_pred = model_pos.predict(X_test)

    train_mse = np.mean((train_pred - y_train_pos) ** 2)
    test_mse = np.mean((test_pred - y_test_pos) ** 2)

    print(f"  Train MSE: {train_mse:.4f}")
    print(f"  Test MSE: {test_mse:.4f}")
    print(f"  Random baseline: ~0.083")

    print(f"\n  Test predictions vs actual:")
    for i, p in enumerate(range(31, 41)):
        print(f"    Puzzle {p}: actual={y_test_pos[i]:.3f}, pred={test_pred[i]:.3f}")

    # Train bit predictors
    print("\nBit prediction:")
    for bit_idx in range(8):
        model_bit = HistGradientBoostingClassifier(max_iter=50, max_depth=3)
        model_bit.fit(X_train, y_train_bits[:, bit_idx])

        train_acc = np.mean(model_bit.predict(X_train) == y_train_bits[:, bit_idx])
        test_acc = np.mean(model_bit.predict(X_test) == y_test_bits[:, bit_idx])

        print(f"  Bit {bit_idx}: train_acc={train_acc:.2f}, test_acc={test_acc:.2f}")


if __name__ == "__main__":
    train_position_transformer()
    analyze_bit_attention()
    cross_puzzle_correlation()
    train_on_solved_predict_higher()
