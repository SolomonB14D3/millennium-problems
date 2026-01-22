#!/usr/bin/env python3
"""
Unified Information Leak Detector for Random 3-SAT

Combines all discovered leak sources:
1. Phase transition distance (α vs α_c)
2. Polarity micro-structure (symmetry patterns)
3. Graph-theoretic features (variable interaction graph)
4. Information-theoretic measures (entropy patterns)

This represents the full arsenal for detecting that "random" isn't random.
"""

import os
import sys
import pickle
import random
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# ML imports
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# Feature Extraction
# =============================================================================

@dataclass
class UnifiedFeatures:
    """All features combined into one comprehensive set."""

    # Phase transition (known physics)
    alpha: float
    alpha_distance: float

    # Polarity micro-structure
    avg_symmetry: float
    symmetry_std: float
    symmetry_kurtosis: float
    backbone_potential: float

    # Variable occurrence patterns
    var_occurrence_std: float
    occurrence_entropy: float

    # Graph structure
    clustering_coeff: float
    degree_assortativity: float
    avg_neighbor_degree: float

    # Information-theoretic
    joint_entropy: float
    resolution_potential: float
    literal_balance: float

    # Local structure
    local_sat_potential: float
    pure_literal_fraction: float

    def to_vector(self) -> List[float]:
        return [
            self.alpha,
            self.alpha_distance,
            self.avg_symmetry,
            self.symmetry_std,
            self.symmetry_kurtosis,
            self.backbone_potential,
            self.var_occurrence_std,
            self.occurrence_entropy,
            self.clustering_coeff,
            self.degree_assortativity,
            self.avg_neighbor_degree,
            self.joint_entropy,
            self.resolution_potential,
            self.literal_balance,
            self.local_sat_potential,
            self.pure_literal_fraction,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        return [
            'alpha', 'alpha_distance',
            'avg_symmetry', 'symmetry_std', 'symmetry_kurtosis', 'backbone_potential',
            'var_occurrence_std', 'occurrence_entropy',
            'clustering_coeff', 'degree_assortativity', 'avg_neighbor_degree',
            'joint_entropy', 'resolution_potential', 'literal_balance',
            'local_sat_potential', 'pure_literal_fraction',
        ]


def predict_alpha_c(n: int) -> float:
    """Phase transition point for n variables."""
    if n <= 20: return 3.8
    if n <= 50: return 4.0
    if n <= 100: return 4.15
    if n <= 200: return 4.2
    return 4.267


def extract_unified_features(clauses: List[List[int]], n_vars: int) -> UnifiedFeatures:
    """Extract the unified feature set."""

    n_clauses = len(clauses)
    alpha = n_clauses / n_vars if n_vars > 0 else 0
    alpha_c = predict_alpha_c(n_vars)
    alpha_distance = (alpha - alpha_c) / alpha_c if alpha_c > 0 else 0

    # Polarity counts
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

            if p == 0 or n == 0:
                pure_count += 1

    if symmetry_scores:
        avg_symmetry = sum(symmetry_scores) / len(symmetry_scores)
        variance = sum((s - avg_symmetry)**2 for s in symmetry_scores) / len(symmetry_scores)
        symmetry_std = math.sqrt(variance) if variance > 0 else 0

        # Kurtosis
        if symmetry_std > 0:
            kurtosis = sum((s - avg_symmetry)**4 for s in symmetry_scores) / (len(symmetry_scores) * symmetry_std**4) - 3
        else:
            kurtosis = 0
    else:
        avg_symmetry, symmetry_std, kurtosis = 0.5, 0, 0

    backbone_potential = backbone_count / n_vars if n_vars > 0 else 0
    pure_literal_fraction = pure_count / n_vars if n_vars > 0 else 0

    # Variable occurrence features
    occurrences = [var_occurrence.get(v, 0) for v in range(1, n_vars + 1)]
    if occurrences:
        occ_mean = sum(occurrences) / len(occurrences)
        var_occurrence_std = math.sqrt(sum((o - occ_mean)**2 for o in occurrences) / len(occurrences))

        total_occ = sum(occurrences)
        if total_occ > 0:
            probs = [o / total_occ for o in occurrences if o > 0]
            occurrence_entropy = -sum(p * math.log2(p) for p in probs) / math.log2(n_vars) if n_vars > 1 else 0
        else:
            occurrence_entropy = 0
    else:
        var_occurrence_std, occurrence_entropy = 0, 0

    # Graph features
    graph = defaultdict(set)
    for clause in clauses:
        vars_in = [abs(lit) for lit in clause]
        for i, v1 in enumerate(vars_in):
            for v2 in vars_in[i+1:]:
                graph[v1].add(v2)
                graph[v2].add(v1)

    degrees = {v: len(graph[v]) for v in range(1, n_vars + 1)}

    # Clustering coefficient
    clustering_sum = 0
    count = 0
    for v in range(1, n_vars + 1):
        neighbors = graph[v]
        if len(neighbors) < 2:
            continue
        pairs = 0
        connected = 0
        neighbor_list = list(neighbors)
        for i, n1 in enumerate(neighbor_list):
            for n2 in neighbor_list[i+1:]:
                pairs += 1
                if n2 in graph[n1]:
                    connected += 1
        if pairs > 0:
            clustering_sum += connected / pairs
            count += 1

    clustering_coeff = clustering_sum / count if count > 0 else 0

    # Degree assortativity
    if degrees:
        degree_list = list(degrees.values())
        mean_degree = sum(degree_list) / len(degree_list)
        assort_sum = 0
        edge_count = 0
        for v in range(1, n_vars + 1):
            d1 = degrees[v]
            for n in graph[v]:
                d2 = degrees[n]
                assort_sum += (d1 - mean_degree) * (d2 - mean_degree)
                edge_count += 1
        degree_assortativity = assort_sum / edge_count if edge_count > 0 else 0
    else:
        degree_assortativity = 0

    # Average neighbor degree
    total_neighbor_deg = 0
    neighbor_count = 0
    for v in range(1, n_vars + 1):
        for n in graph[v]:
            total_neighbor_deg += degrees[n]
            neighbor_count += 1
    avg_neighbor_degree = total_neighbor_deg / neighbor_count if neighbor_count > 0 else 0

    # Joint entropy
    pair_counts = Counter()
    for clause in clauses:
        vars_in = [abs(lit) for lit in clause]
        for i, v1 in enumerate(vars_in):
            for v2 in vars_in[i+1:]:
                pair_counts[(min(v1, v2), max(v1, v2))] += 1

    total_pairs = sum(pair_counts.values())
    if total_pairs > 0:
        joint_entropy = -sum(
            (c / total_pairs) * math.log2(c / total_pairs)
            for c in pair_counts.values() if c > 0
        )
        max_pairs = n_vars * (n_vars - 1) // 2
        joint_entropy = joint_entropy / math.log2(max_pairs) if max_pairs > 1 else 0
    else:
        joint_entropy = 0

    # Resolution potential
    resolution_sum = sum(min(pos_count.get(v, 0), neg_count.get(v, 0)) for v in range(1, n_vars + 1))
    resolution_potential = resolution_sum / (n_vars * n_clauses) if n_vars > 0 and n_clauses > 0 else 0

    # Literal balance
    pos_total = sum(1 for clause in clauses for lit in clause if lit > 0)
    neg_total = sum(1 for clause in clauses for lit in clause if lit < 0)
    total_lits = pos_total + neg_total
    literal_balance = 1 - abs(pos_total - neg_total) / total_lits if total_lits > 0 else 0.5

    # Local SAT potential
    local_sat_sum = 0
    for v in range(1, n_vars + 1):
        p = pos_count.get(v, 0)
        n = neg_count.get(v, 0)
        total = p + n
        if total > 0:
            local_sat_sum += max(p, n) / total

    local_sat_potential = local_sat_sum / n_vars if n_vars > 0 else 0

    return UnifiedFeatures(
        alpha=alpha,
        alpha_distance=alpha_distance,
        avg_symmetry=avg_symmetry,
        symmetry_std=symmetry_std,
        symmetry_kurtosis=kurtosis,
        backbone_potential=backbone_potential,
        var_occurrence_std=var_occurrence_std,
        occurrence_entropy=occurrence_entropy,
        clustering_coeff=clustering_coeff,
        degree_assortativity=degree_assortativity,
        avg_neighbor_degree=avg_neighbor_degree,
        joint_entropy=joint_entropy,
        resolution_potential=resolution_potential,
        literal_balance=literal_balance,
        local_sat_potential=local_sat_potential,
        pure_literal_fraction=pure_literal_fraction,
    )


# =============================================================================
# Unified Detector
# =============================================================================

class UnifiedLeakDetector:
    """
    The complete information leak detector for "random" 3-SAT.

    Combines:
    - Physics: Phase transition theory
    - Statistics: Polarity micro-structure
    - Graphs: Variable interaction patterns
    - Information theory: Entropy measures
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.trained = False

    def train(self, X: List[List[float]], y: List[int]):
        """Train the unified detector."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required")

        X = np.array(X)
        y = np.array(y)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=5,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        self.trained = True

        # Feature importance
        importances = self.model.feature_importances_
        names = UnifiedFeatures.feature_names()
        sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

        print("\nLeak Sources (by importance):")
        print("-" * 50)
        for i in sorted_idx:
            bar = "█" * int(importances[i] * 50)
            print(f"  {names[i]:25s} {importances[i]:.3f} {bar}")

    def predict(self, clauses: List[List[int]], n_vars: int) -> Tuple[Optional[bool], float, Dict]:
        """
        Predict SAT/UNSAT and return detailed analysis.

        Returns:
            (prediction, confidence, leak_analysis)
        """
        features = extract_unified_features(clauses, n_vars)

        if not self.trained:
            # Fallback to rule-based prediction
            if abs(features.alpha_distance) > 0.15:
                prediction = features.alpha_distance < 0
                confidence = min(0.9, 0.6 + abs(features.alpha_distance))
            elif features.avg_symmetry < 0.76:
                prediction = True
                confidence = 0.65
            elif features.avg_symmetry > 0.78:
                prediction = False
                confidence = 0.65
            else:
                prediction = None
                confidence = 0.5

            return prediction, confidence, {'method': 'rule-based', 'features': features}

        X = np.array([features.to_vector()])
        X_scaled = self.scaler.transform(X)

        pred = self.model.predict(X_scaled)[0]
        prob = self.model.predict_proba(X_scaled)[0]
        confidence = max(prob)

        # Analyze which features contributed most
        leak_analysis = {
            'method': 'ml',
            'features': features,
            'primary_leaks': []
        }

        names = UnifiedFeatures.feature_names()
        importances = self.model.feature_importances_
        for i in sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)[:5]:
            leak_analysis['primary_leaks'].append({
                'feature': names[i],
                'value': features.to_vector()[i],
                'importance': importances[i]
            })

        return bool(pred), confidence, leak_analysis

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'trained': self.trained
            }, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.trained = data['trained']


# =============================================================================
# Training and Evaluation
# =============================================================================

def generate_random_3sat(n: int, m: int, seed: int) -> List[List[int]]:
    random.seed(seed)
    return [[v if random.random() > 0.5 else -v
             for v in random.sample(range(1, n + 1), 3)]
            for _ in range(m)]


def solve_kissat(clauses: List[List[int]], n_vars: int) -> Optional[bool]:
    import subprocess
    import tempfile

    lines = [f"p cnf {n_vars} {len(clauses)}"]
    for clause in clauses:
        lines.append(" ".join(map(str, clause)) + " 0")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
        f.write("\n".join(lines))
        path = f.name

    try:
        result = subprocess.run(['kissat', '--quiet', path], capture_output=True, timeout=30)
        if result.returncode == 10: return True
        if result.returncode == 20: return False
        return None
    except:
        return None
    finally:
        os.remove(path)


def main():
    print("=" * 70)
    print("UNIFIED INFORMATION LEAK DETECTOR FOR RANDOM 3-SAT")
    print("=" * 70)
    print()
    print("Combining all discovered leak sources:")
    print("  • Phase transition physics (α vs α_c)")
    print("  • Polarity micro-structure (symmetry patterns)")
    print("  • Graph theory (variable interaction network)")
    print("  • Information theory (entropy measures)")
    print()

    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn required. Install with: pip install scikit-learn")
        return

    N_VARS = 75
    N_TRAIN = 800
    alpha_c = predict_alpha_c(N_VARS)

    print(f"Training configuration:")
    print(f"  Variables: {N_VARS}")
    print(f"  Training instances: {N_TRAIN}")
    print(f"  Phase transition: α_c ≈ {alpha_c:.2f}")
    print()

    # Generate training data
    print("Generating training data...")
    X_train = []
    y_train = []

    for i in range(N_TRAIN):
        # Focus on transition region
        if random.random() < 0.6:
            alpha = alpha_c + random.gauss(0, 0.2)
        else:
            alpha = random.uniform(3.5, 4.8)
        alpha = max(3.5, min(4.8, alpha))
        m = int(N_VARS * alpha)

        clauses = generate_random_3sat(N_VARS, m, i)
        result = solve_kissat(clauses, N_VARS)

        if result is None:
            continue

        features = extract_unified_features(clauses, N_VARS)
        X_train.append(features.to_vector())
        y_train.append(1 if result else 0)

        if (i + 1) % 200 == 0:
            print(f"  Generated {i + 1}/{N_TRAIN}...")

    print(f"\nTraining set: {len(y_train)} instances")
    print(f"  SAT: {sum(y_train)} ({sum(y_train)/len(y_train):.1%})")
    print(f"  UNSAT: {len(y_train)-sum(y_train)} ({(len(y_train)-sum(y_train))/len(y_train):.1%})")

    # Train
    print("\n" + "=" * 70)
    print("Training Unified Detector")
    print("=" * 70)

    detector = UnifiedLeakDetector()
    detector.train(X_train, y_train)

    # Test
    print("\n" + "=" * 70)
    print("Testing")
    print("=" * 70)

    # Test at exact transition
    print(f"\nAt phase transition (α = {alpha_c:.2f}):")
    m = int(N_VARS * alpha_c)

    correct = 0
    total = 0
    confidences = []

    for seed in range(200):
        clauses = generate_random_3sat(N_VARS, m, seed + 50000)
        actual = solve_kissat(clauses, N_VARS)

        if actual is None:
            continue

        pred, conf, _ = detector.predict(clauses, N_VARS)
        total += 1
        confidences.append(conf)
        if pred == actual:
            correct += 1

    print(f"  Accuracy: {correct}/{total} = {correct/total:.1%}")
    print(f"  Avg confidence: {sum(confidences)/len(confidences):.2f}")

    # Test across range
    print("\nAcross phase transition:")
    for alpha_mult in [0.90, 0.95, 1.0, 1.05, 1.10]:
        alpha = alpha_c * alpha_mult
        m = int(N_VARS * alpha)

        correct = 0
        total = 0
        sat_count = 0

        for seed in range(100):
            clauses = generate_random_3sat(N_VARS, m, seed + 60000)
            actual = solve_kissat(clauses, N_VARS)

            if actual is None:
                continue

            pred, _, _ = detector.predict(clauses, N_VARS)
            total += 1
            if actual:
                sat_count += 1
            if pred == actual:
                correct += 1

        leak = abs(correct/total - 0.5) * 2 if total > 0 else 0
        region = "SAT" if alpha < alpha_c * 0.95 else "UNSAT" if alpha > alpha_c * 1.05 else "HARD"

        print(f"  α={alpha:.2f} ({region:5s}): acc={correct/total:.0%}, SAT={sat_count/total:.0%}, leak={leak:.0%}")

    # Save
    model_path = os.path.join(os.path.dirname(__file__), 'unified_detector.pkl')
    detector.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The unified detector combines multiple independent leak sources:

1. PHYSICS: Distance from phase transition (α - α_c)
   → Easy SAT/UNSAT far from transition

2. MICRO-STRUCTURE: Polarity symmetry patterns
   → SAT instances have lower symmetry (more polarized)

3. GRAPH THEORY: Variable interaction network
   → Hard instances have high degree assortativity

4. INFORMATION THEORY: Entropy of variable co-occurrences
   → Predictable instances have non-uniform entropy

Together, these expose structure in "random" instances that shouldn't
exist if they were truly random. The leak is real - randomness has
fingerprints.
""")


if __name__ == "__main__":
    main()
