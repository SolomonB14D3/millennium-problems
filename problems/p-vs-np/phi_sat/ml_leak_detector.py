#!/usr/bin/env python3
"""
ML-based Information Leak Detector for Random 3-SAT

Learns to detect structural fingerprints that break "randomness".

Key insight: If we can predict SAT/UNSAT from structural features,
then the instance isn't truly random - there's an information leak.

Features extracted:
- Polarity statistics (what we already know works)
- Variable occurrence patterns
- Clause interaction graph properties
- Local structure metrics
- Phase transition distance

The model learns which combinations of features leak information.
"""

import random
import math
import subprocess
import tempfile
import os
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import pickle

# Check for sklearn
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")


# =============================================================================
# Feature Extraction
# =============================================================================

@dataclass
class SATFeatures:
    """All extracted features from a SAT instance."""

    # Basic structure
    n_vars: int
    n_clauses: int
    alpha: float                    # m/n ratio
    alpha_distance: float           # Distance from predicted α_c

    # Polarity features (what we know works)
    avg_symmetry: float             # Average polarity balance
    symmetry_std: float             # Std dev of symmetry
    symmetry_skew: float            # Skewness of symmetry distribution
    backbone_potential: float       # Fraction of strongly polarized vars

    # Variable occurrence features
    var_occurrence_mean: float      # Mean occurrences per variable
    var_occurrence_std: float       # Std dev of occurrences
    var_occurrence_max: float       # Max occurrences (normalized)
    var_occurrence_min: float       # Min occurrences (normalized)
    occurrence_entropy: float       # Entropy of occurrence distribution

    # Clause interaction features
    clause_overlap_mean: float      # Mean variable overlap between clauses
    clause_overlap_max: float       # Max overlap
    var_clause_ratio: float         # Variables appearing in multiple clauses

    # Graph features (variable-clause incidence)
    avg_var_degree: float           # Average variable degree
    max_var_degree: float           # Max variable degree (normalized)
    var_degree_std: float           # Std dev of variable degrees

    # Local structure
    pure_literal_fraction: float    # Variables appearing in only one polarity
    unit_propagation_depth: float   # How much unit prop simplifies
    binary_clause_fraction: float   # Fraction of effectively binary clauses

    # Interaction patterns
    positive_clause_fraction: float # Clauses with more positive literals
    mixed_polarity_fraction: float  # Clauses with both polarities

    def to_vector(self) -> List[float]:
        """Convert to feature vector for ML."""
        return [
            self.alpha,
            self.alpha_distance,
            self.avg_symmetry,
            self.symmetry_std,
            self.symmetry_skew,
            self.backbone_potential,
            self.var_occurrence_mean,
            self.var_occurrence_std,
            self.var_occurrence_max,
            self.var_occurrence_min,
            self.occurrence_entropy,
            self.clause_overlap_mean,
            self.clause_overlap_max,
            self.var_clause_ratio,
            self.avg_var_degree,
            self.max_var_degree,
            self.var_degree_std,
            self.pure_literal_fraction,
            self.unit_propagation_depth,
            self.binary_clause_fraction,
            self.positive_clause_fraction,
            self.mixed_polarity_fraction,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        """Names for each feature (for interpretability)."""
        return [
            'alpha',
            'alpha_distance',
            'avg_symmetry',
            'symmetry_std',
            'symmetry_skew',
            'backbone_potential',
            'var_occurrence_mean',
            'var_occurrence_std',
            'var_occurrence_max',
            'var_occurrence_min',
            'occurrence_entropy',
            'clause_overlap_mean',
            'clause_overlap_max',
            'var_clause_ratio',
            'avg_var_degree',
            'max_var_degree',
            'var_degree_std',
            'pure_literal_fraction',
            'unit_propagation_depth',
            'binary_clause_fraction',
            'positive_clause_fraction',
            'mixed_polarity_fraction',
        ]


def predict_alpha_c(n: int) -> float:
    """Predict critical α for given n (from phase transition theory)."""
    # Empirical fit - transition is ~4.2-4.267 for all practical n
    # Finite-size effects cause slight shift but not as dramatic as old models suggested
    if n <= 20: return 3.8
    if n <= 50: return 4.0
    if n <= 100: return 4.15
    if n <= 200: return 4.2
    return 4.267  # Asymptotic limit


def extract_features(clauses: List[List[int]], n_vars: int) -> SATFeatures:
    """Extract all features from a CNF formula."""

    n_clauses = len(clauses)
    alpha = n_clauses / n_vars if n_vars > 0 else 0
    alpha_c = predict_alpha_c(n_vars)
    alpha_distance = (alpha - alpha_c) / alpha_c if alpha_c > 0 else 0

    # Count occurrences
    pos_count = Counter()
    neg_count = Counter()
    var_occurrence = Counter()

    for clause in clauses:
        for lit in clause:
            var = abs(lit)
            var_occurrence[var] += 1
            if lit > 0:
                pos_count[var] += 1
            else:
                neg_count[var] += 1

    # Symmetry features
    symmetry_scores = []
    backbone_count = 0
    pure_count = 0

    for v in range(1, n_vars + 1):
        p = pos_count.get(v, 0)
        n = neg_count.get(v, 0)
        total = p + n

        if total > 0:
            symmetry = 1 - abs(p - n) / total
            symmetry_scores.append(symmetry)

            if abs(p - n) / total > 0.6:
                backbone_count += 1

            # Pure literal: appears in only one polarity
            if p == 0 or n == 0:
                pure_count += 1

    if symmetry_scores:
        avg_symmetry = sum(symmetry_scores) / len(symmetry_scores)
        symmetry_std = math.sqrt(sum((s - avg_symmetry)**2 for s in symmetry_scores) / len(symmetry_scores))
        # Skewness
        if symmetry_std > 0:
            symmetry_skew = sum((s - avg_symmetry)**3 for s in symmetry_scores) / (len(symmetry_scores) * symmetry_std**3)
        else:
            symmetry_skew = 0
    else:
        avg_symmetry = 0.5
        symmetry_std = 0
        symmetry_skew = 0

    backbone_potential = backbone_count / n_vars if n_vars > 0 else 0
    pure_literal_fraction = pure_count / n_vars if n_vars > 0 else 0

    # Variable occurrence features
    occurrences = [var_occurrence.get(v, 0) for v in range(1, n_vars + 1)]
    if occurrences:
        var_occurrence_mean = sum(occurrences) / len(occurrences)
        var_occurrence_std = math.sqrt(sum((o - var_occurrence_mean)**2 for o in occurrences) / len(occurrences))
        var_occurrence_max = max(occurrences) / var_occurrence_mean if var_occurrence_mean > 0 else 0
        var_occurrence_min = min(occurrences) / var_occurrence_mean if var_occurrence_mean > 0 else 0

        # Entropy
        total_occ = sum(occurrences)
        if total_occ > 0:
            probs = [o / total_occ for o in occurrences if o > 0]
            occurrence_entropy = -sum(p * math.log2(p) for p in probs) / math.log2(n_vars) if n_vars > 1 else 0
        else:
            occurrence_entropy = 0
    else:
        var_occurrence_mean = 0
        var_occurrence_std = 0
        var_occurrence_max = 0
        var_occurrence_min = 0
        occurrence_entropy = 0

    # Clause interaction features
    clause_sets = [frozenset(abs(lit) for lit in clause) for clause in clauses]

    overlaps = []
    for i in range(min(len(clause_sets), 1000)):  # Sample for large instances
        for j in range(i + 1, min(len(clause_sets), 1000)):
            overlap = len(clause_sets[i] & clause_sets[j])
            overlaps.append(overlap)

    if overlaps:
        clause_overlap_mean = sum(overlaps) / len(overlaps)
        clause_overlap_max = max(overlaps)
    else:
        clause_overlap_mean = 0
        clause_overlap_max = 0

    # Variables appearing in multiple clauses
    multi_clause_vars = sum(1 for v in range(1, n_vars + 1) if var_occurrence.get(v, 0) > 1)
    var_clause_ratio = multi_clause_vars / n_vars if n_vars > 0 else 0

    # Graph features (variable degree = number of clauses containing it)
    var_degrees = [var_occurrence.get(v, 0) for v in range(1, n_vars + 1)]
    if var_degrees:
        avg_var_degree = sum(var_degrees) / len(var_degrees)
        max_var_degree = max(var_degrees) / avg_var_degree if avg_var_degree > 0 else 0
        var_degree_std = math.sqrt(sum((d - avg_var_degree)**2 for d in var_degrees) / len(var_degrees))
    else:
        avg_var_degree = 0
        max_var_degree = 0
        var_degree_std = 0

    # Unit propagation depth (simplified estimate)
    # Count variables that would be forced by pure literal elimination
    unit_propagation_depth = pure_literal_fraction  # Simplified

    # Binary clause fraction (clauses that become binary after one assignment)
    binary_clause_fraction = 0  # Would need simulation

    # Clause polarity features
    positive_clauses = sum(1 for clause in clauses if sum(1 for lit in clause if lit > 0) > len(clause) / 2)
    positive_clause_fraction = positive_clauses / n_clauses if n_clauses > 0 else 0.5

    mixed_clauses = sum(1 for clause in clauses
                        if any(lit > 0 for lit in clause) and any(lit < 0 for lit in clause))
    mixed_polarity_fraction = mixed_clauses / n_clauses if n_clauses > 0 else 0

    return SATFeatures(
        n_vars=n_vars,
        n_clauses=n_clauses,
        alpha=alpha,
        alpha_distance=alpha_distance,
        avg_symmetry=avg_symmetry,
        symmetry_std=symmetry_std,
        symmetry_skew=symmetry_skew,
        backbone_potential=backbone_potential,
        var_occurrence_mean=var_occurrence_mean,
        var_occurrence_std=var_occurrence_std,
        var_occurrence_max=var_occurrence_max,
        var_occurrence_min=var_occurrence_min,
        occurrence_entropy=occurrence_entropy,
        clause_overlap_mean=clause_overlap_mean,
        clause_overlap_max=clause_overlap_max,
        var_clause_ratio=var_clause_ratio,
        avg_var_degree=avg_var_degree,
        max_var_degree=max_var_degree,
        var_degree_std=var_degree_std,
        pure_literal_fraction=pure_literal_fraction,
        unit_propagation_depth=unit_propagation_depth,
        binary_clause_fraction=binary_clause_fraction,
        positive_clause_fraction=positive_clause_fraction,
        mixed_polarity_fraction=mixed_polarity_fraction,
    )


# =============================================================================
# Training Data Generation
# =============================================================================

def generate_random_3sat(n: int, m: int, seed: int) -> List[List[int]]:
    """Generate a random 3-SAT instance."""
    random.seed(seed)
    clauses = []
    for _ in range(m):
        # Pick 3 distinct variables
        vars = random.sample(range(1, n + 1), 3)
        # Random polarities
        clause = [v if random.random() > 0.5 else -v for v in vars]
        clauses.append(clause)
    return clauses


def solve_with_kissat(clauses: List[List[int]], n_vars: int, timeout: int = 30) -> Optional[bool]:
    """Solve a SAT instance using Kissat."""
    # Write to temp file
    lines = [f"p cnf {n_vars} {len(clauses)}"]
    for clause in clauses:
        lines.append(" ".join(map(str, clause)) + " 0")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
        f.write("\n".join(lines))
        path = f.name

    try:
        result = subprocess.run(
            ['kissat', '--quiet', path],
            capture_output=True,
            timeout=timeout
        )
        if result.returncode == 10:
            return True  # SAT
        elif result.returncode == 20:
            return False  # UNSAT
        return None  # Unknown
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    finally:
        os.remove(path)


def generate_training_data(
    n_instances: int = 1000,
    n_vars: int = 100,
    alpha_range: Tuple[float, float] = (3.0, 5.0),
    seed_start: int = 0,
    verbose: bool = True
) -> Tuple[List[List[float]], List[int]]:
    """
    Generate training data by creating random instances and solving them.

    Returns:
        X: Feature vectors
        y: Labels (1 = SAT, 0 = UNSAT)
    """
    X = []
    y = []

    alpha_c = predict_alpha_c(n_vars)

    for i in range(n_instances):
        seed = seed_start + i

        # Sample α from range, with more samples near α_c
        if random.random() < 0.5:
            # Near transition
            alpha = alpha_c + random.gauss(0, 0.3)
            alpha = max(alpha_range[0], min(alpha_range[1], alpha))
        else:
            # Uniform across range
            alpha = random.uniform(*alpha_range)

        m = int(n_vars * alpha)

        # Generate and solve
        clauses = generate_random_3sat(n_vars, m, seed)
        result = solve_with_kissat(clauses, n_vars)

        if result is None:
            continue  # Skip timeouts

        # Extract features
        features = extract_features(clauses, n_vars)
        X.append(features.to_vector())
        y.append(1 if result else 0)

        if verbose and (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{n_instances} instances...")

    return X, y


# =============================================================================
# ML Model
# =============================================================================

class LeakDetector:
    """
    ML model that learns to detect information leaks in random SAT.

    Uses ensemble methods (Random Forest, Gradient Boosting) to find
    which feature combinations predict SAT/UNSAT.
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_names = SATFeatures.feature_names()
        self.trained = False

    def train(self, X: List[List[float]], y: List[int], model_type: str = 'rf'):
        """
        Train the leak detector.

        Args:
            X: Feature vectors
            y: Labels (1=SAT, 0=UNSAT)
            model_type: 'rf' for Random Forest, 'gb' for Gradient Boosting
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required for training")

        import numpy as np
        X = np.array(X)
        y = np.array(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )

        self.model.fit(X_scaled, y)
        self.trained = True

        # Report feature importance
        importances = self.model.feature_importances_
        sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

        print("\nFeature Importances (leak indicators):")
        print("-" * 50)
        for i in sorted_idx[:10]:
            print(f"  {self.feature_names[i]:25s}: {importances[i]:.4f}")

    def predict(self, features: SATFeatures) -> Tuple[bool, float]:
        """
        Predict SAT/UNSAT from features.

        Returns:
            (prediction, confidence)
        """
        if not self.trained:
            raise RuntimeError("Model not trained")

        import numpy as np
        X = np.array([features.to_vector()])
        X_scaled = self.scaler.transform(X)

        pred = self.model.predict(X_scaled)[0]
        prob = self.model.predict_proba(X_scaled)[0]
        confidence = max(prob)

        return bool(pred), confidence

    def predict_clauses(self, clauses: List[List[int]], n_vars: int) -> Tuple[bool, float]:
        """Predict from raw clauses."""
        features = extract_features(clauses, n_vars)
        return self.predict(features)

    def evaluate(self, X: List[List[float]], y: List[int]) -> Dict:
        """Evaluate model performance."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required")

        import numpy as np
        X = np.array(X)
        y = np.array(y)
        X_scaled = self.scaler.transform(X)

        y_pred = self.model.predict(X_scaled)

        return {
            'accuracy': accuracy_score(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'report': classification_report(y, y_pred, target_names=['UNSAT', 'SAT'])
        }

    def cross_validate(self, X: List[List[float]], y: List[int], cv: int = 5) -> float:
        """Cross-validation score."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required")

        import numpy as np
        X = np.array(X)
        y = np.array(y)
        X_scaled = self.scaler.fit_transform(X)

        scores = cross_val_score(self.model, X_scaled, y, cv=cv)
        return scores.mean()

    def save(self, path: str):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'trained': self.trained
            }, f)

    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.trained = data['trained']


# =============================================================================
# Analysis Tools
# =============================================================================

def analyze_leak_strength(detector: LeakDetector, n_vars: int = 100, n_samples: int = 200):
    """
    Analyze how much information leaks at different α values.

    A high accuracy at α_c means strong leak (randomness is broken).
    """
    alpha_c = predict_alpha_c(n_vars)

    print(f"\nLeak Strength Analysis (n={n_vars}, α_c≈{alpha_c:.2f})")
    print("=" * 60)

    # Test at different α values - focus on the transition region
    alphas = [alpha_c * mult for mult in [0.90, 0.95, 0.98, 1.0, 1.02, 1.05, 1.10]]

    for alpha in alphas:
        m = int(n_vars * alpha)

        correct = 0
        total = 0
        sat_count = 0

        for seed in range(n_samples):
            clauses = generate_random_3sat(n_vars, m, seed + 10000)
            actual = solve_with_kissat(clauses, n_vars)

            if actual is None:
                continue

            pred, conf = detector.predict_clauses(clauses, n_vars)

            total += 1
            if actual:
                sat_count += 1
            if pred == actual:
                correct += 1

        if total > 0:
            accuracy = correct / total
            sat_rate = sat_count / total
            leak = abs(accuracy - 0.5) * 2  # 0 = no leak, 1 = perfect leak

            region = "SAT" if alpha < alpha_c * 0.9 else "UNSAT" if alpha > alpha_c * 1.1 else "HARD"

            print(f"  α={alpha:.2f} ({region:5s}): accuracy={accuracy:.1%}, "
                  f"SAT rate={sat_rate:.1%}, leak={leak:.1%}")


def find_hardest_instances(detector: LeakDetector, n_vars: int = 100, n_samples: int = 500):
    """
    Find instances that the ML model can't predict.

    These are candidates for "truly random" hard instances.
    """
    alpha_c = predict_alpha_c(n_vars)
    m = int(n_vars * alpha_c)

    print(f"\nSearching for truly hard instances (n={n_vars}, α={alpha_c:.2f})")
    print("-" * 60)

    hard_instances = []

    for seed in range(n_samples):
        clauses = generate_random_3sat(n_vars, m, seed + 20000)
        actual = solve_with_kissat(clauses, n_vars)

        if actual is None:
            continue

        pred, conf = detector.predict_clauses(clauses, n_vars)

        # Hard = low confidence AND wrong prediction
        if conf < 0.6:
            hard_instances.append({
                'seed': seed + 20000,
                'confidence': conf,
                'predicted': pred,
                'actual': actual,
                'correct': pred == actual
            })

    # Sort by confidence (lowest = hardest to predict)
    hard_instances.sort(key=lambda x: x['confidence'])

    print(f"Found {len(hard_instances)} low-confidence instances")
    print("\nHardest to predict:")
    for inst in hard_instances[:10]:
        status = "✓" if inst['correct'] else "✗"
        print(f"  seed={inst['seed']}: conf={inst['confidence']:.2f}, "
              f"pred={'SAT' if inst['predicted'] else 'UNSAT'}, "
              f"actual={'SAT' if inst['actual'] else 'UNSAT'} {status}")

    return hard_instances


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if not SKLEARN_AVAILABLE:
        print("Please install scikit-learn: pip install scikit-learn")
        exit(1)

    print("=" * 70)
    print("ML-based Information Leak Detector for Random 3-SAT")
    print("=" * 70)
    print()
    print("Training on random 3-SAT instances to learn structural fingerprints")
    print("that 'leak' whether an instance is SAT or UNSAT.")
    print()

    # Parameters
    N_VARS = 75  # Small enough for fast solving
    N_TRAIN = 800
    N_TEST = 300

    # Focus on the hard region around the phase transition
    ALPHA_LOW = 3.8
    ALPHA_HIGH = 4.8

    print(f"Configuration:")
    print(f"  Variables: {N_VARS}")
    print(f"  Training instances: {N_TRAIN}")
    print(f"  Test instances: {N_TEST}")
    print(f"  α range: {ALPHA_LOW} to {ALPHA_HIGH} (α_c ≈ {predict_alpha_c(N_VARS):.2f})")
    print()

    # Generate training data
    print("Generating training data...")
    X_train, y_train = generate_training_data(
        n_instances=N_TRAIN,
        n_vars=N_VARS,
        alpha_range=(ALPHA_LOW, ALPHA_HIGH),
        seed_start=0
    )

    print(f"\nTraining set: {len(y_train)} instances")
    print(f"  SAT: {sum(y_train)} ({sum(y_train)/len(y_train):.1%})")
    print(f"  UNSAT: {len(y_train) - sum(y_train)} ({(len(y_train)-sum(y_train))/len(y_train):.1%})")

    # Generate test data (different seeds)
    print("\nGenerating test data...")
    X_test, y_test = generate_training_data(
        n_instances=N_TEST,
        n_vars=N_VARS,
        alpha_range=(ALPHA_LOW, ALPHA_HIGH),
        seed_start=100000,
        verbose=False
    )

    # Train model
    print("\n" + "=" * 70)
    print("Training Random Forest classifier...")
    print("=" * 70)

    detector = LeakDetector()
    detector.train(X_train, y_train, model_type='rf')

    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    results = detector.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {results['accuracy']:.1%}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              UNSAT    SAT")
    print(f"  Actual UNSAT  {results['confusion_matrix'][0][0]:4d}   {results['confusion_matrix'][0][1]:4d}")
    print(f"  Actual SAT    {results['confusion_matrix'][1][0]:4d}   {results['confusion_matrix'][1][1]:4d}")

    print(f"\n{results['report']}")

    # Analyze leak strength at different α
    print("\n" + "=" * 70)
    print("Leak Strength Analysis")
    print("=" * 70)
    analyze_leak_strength(detector, n_vars=N_VARS, n_samples=100)

    # Find hardest instances
    print("\n" + "=" * 70)
    print("Searching for 'Truly Random' Instances")
    print("=" * 70)
    hard = find_hardest_instances(detector, n_vars=N_VARS, n_samples=200)

    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
Key finding: The ML model learns to detect information leaks in 'random' SAT.

What this means:
1. If accuracy >> 50%, the randomness is broken - there ARE detectable patterns
2. Feature importances show WHICH properties leak the most information
3. Low-confidence instances are candidates for 'truly random' hard instances

The meta-insight: By training on physics (phase transition behavior), we can
learn patterns that weren't explicitly programmed, potentially discovering
new structural fingerprints that break randomness.
""")

    # Save model
    model_path = os.path.join(os.path.dirname(__file__), 'leak_detector.pkl')
    detector.save(model_path)
    print(f"\nModel saved to: {model_path}")
