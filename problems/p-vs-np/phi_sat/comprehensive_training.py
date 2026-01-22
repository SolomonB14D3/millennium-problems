#!/usr/bin/env python3
"""
Comprehensive Training Data Generation for Leak Detection

Generates diverse training samples to help ML discover more leak sources:

1. Scale variation: Different problem sizes (50-200 vars)
2. α variation: Full range around phase transition
3. Generation methods:
   - Uniform random (standard)
   - Planted solution (we know it's SAT)
   - Forced UNSAT (add contradicting clauses)
   - Community structure (variables in groups)
4. Large sample count for better statistics

The goal: expose the model to many "flavors" of SAT instances
so it can learn what features distinguish SAT from UNSAT.
"""

import os
import sys
import random
import math
import subprocess
import tempfile
import json
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

sys.path.insert(0, '/Users/bryan/millennium-problems/problems/p-vs-np/phi_sat')

from unified_leak_detector import extract_unified_features, UnifiedFeatures, predict_alpha_c


# =============================================================================
# Diverse Instance Generators
# =============================================================================

def generate_uniform_random(n_vars: int, n_clauses: int, seed: int) -> Tuple[List[List[int]], str]:
    """Standard uniform random 3-SAT."""
    random.seed(seed)
    clauses = []
    for _ in range(n_clauses):
        vars = random.sample(range(1, n_vars + 1), 3)
        clause = [v if random.random() > 0.5 else -v for v in vars]
        clauses.append(clause)
    return clauses, "uniform"


def generate_planted_sat(n_vars: int, n_clauses: int, seed: int) -> Tuple[List[List[int]], str]:
    """
    Generate instance with a planted (known) satisfying assignment.

    Method: First pick a random assignment, then generate clauses
    that are all satisfied by it. This guarantees SAT.
    """
    random.seed(seed)

    # Plant a solution
    solution = {v: random.choice([True, False]) for v in range(1, n_vars + 1)}

    clauses = []
    for _ in range(n_clauses):
        vars = random.sample(range(1, n_vars + 1), 3)

        # Generate clause satisfied by the planted solution
        clause = []
        for v in vars:
            if solution[v]:
                # Variable is True in solution
                if random.random() > 0.3:  # Usually keep positive
                    clause.append(v)
                else:
                    clause.append(-v)
            else:
                # Variable is False in solution
                if random.random() > 0.3:  # Usually keep negative
                    clause.append(-v)
                else:
                    clause.append(v)

        # Ensure at least one literal satisfies the clause
        satisfied = any(
            (lit > 0 and solution[abs(lit)]) or (lit < 0 and not solution[abs(lit)])
            for lit in clause
        )
        if not satisfied:
            # Fix by flipping one literal to match solution
            v = vars[0]
            clause[0] = v if solution[v] else -v

        clauses.append(clause)

    return clauses, "planted"


def generate_forced_unsat(n_vars: int, n_clauses: int, seed: int) -> Tuple[List[List[int]], str]:
    """
    Generate instance that's likely UNSAT by adding contradictions.

    Method: Start with random clauses, then add clauses that
    conflict with likely assignments.
    """
    random.seed(seed)

    # Start with random clauses
    clauses = []
    for _ in range(n_clauses - n_vars // 5):
        vars = random.sample(range(1, n_vars + 1), 3)
        clause = [v if random.random() > 0.5 else -v for v in vars]
        clauses.append(clause)

    # Add contradicting pairs for some variables
    # This increases constraint density and likelihood of UNSAT
    contradiction_vars = random.sample(range(1, n_vars + 1), min(n_vars // 5, n_clauses // 5))
    for v in contradiction_vars:
        # Add clauses that force both v and -v
        other_vars1 = random.sample([x for x in range(1, n_vars + 1) if x != v], 2)
        other_vars2 = random.sample([x for x in range(1, n_vars + 1) if x != v], 2)

        clauses.append([v] + [-x for x in other_vars1])  # Forces v to be true
        clauses.append([-v] + [-x for x in other_vars2])  # Forces v to be false

    return clauses[:n_clauses], "forced_unsat"


def generate_community_structure(n_vars: int, n_clauses: int, seed: int) -> Tuple[List[List[int]], str]:
    """
    Generate instance with community structure.

    Variables are divided into groups, and clauses preferentially
    use variables from the same group. This creates structure
    that might leak information.
    """
    random.seed(seed)

    # Divide variables into communities
    n_communities = max(3, n_vars // 20)
    communities = [[] for _ in range(n_communities)]
    for v in range(1, n_vars + 1):
        communities[v % n_communities].append(v)

    clauses = []
    for _ in range(n_clauses):
        if random.random() < 0.7:
            # Intra-community clause (70% of clauses)
            comm = random.choice(communities)
            if len(comm) >= 3:
                vars = random.sample(comm, 3)
            else:
                vars = random.sample(range(1, n_vars + 1), 3)
        else:
            # Inter-community clause (30% of clauses)
            vars = random.sample(range(1, n_vars + 1), 3)

        clause = [v if random.random() > 0.5 else -v for v in vars]
        clauses.append(clause)

    return clauses, "community"


def generate_biased_polarity(n_vars: int, n_clauses: int, seed: int) -> Tuple[List[List[int]], str]:
    """
    Generate instance with biased polarity distribution.

    Some variables appear mostly positive, others mostly negative.
    This is similar to what we see in real SAT instances.
    """
    random.seed(seed)

    # Assign bias to each variable
    bias = {v: random.gauss(0.5, 0.3) for v in range(1, n_vars + 1)}
    bias = {v: max(0.1, min(0.9, b)) for v, b in bias.items()}  # Clamp

    clauses = []
    for _ in range(n_clauses):
        vars = random.sample(range(1, n_vars + 1), 3)
        clause = []
        for v in vars:
            if random.random() < bias[v]:
                clause.append(v)
            else:
                clause.append(-v)
        clauses.append(clause)

    return clauses, "biased"


# =============================================================================
# Solving
# =============================================================================

def solve_instance(clauses: List[List[int]], n_vars: int, timeout: int = 30) -> Optional[bool]:
    """Solve with Kissat."""
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
            return True
        elif result.returncode == 20:
            return False
        return None
    except:
        return None
    finally:
        os.remove(path)


# =============================================================================
# Data Generation
# =============================================================================

def generate_single_instance(args) -> Optional[Dict]:
    """Generate and solve a single instance (for parallel processing)."""
    idx, n_vars, alpha, gen_type, seed = args

    n_clauses = int(n_vars * alpha)

    generators = {
        'uniform': generate_uniform_random,
        'planted': generate_planted_sat,
        'forced_unsat': generate_forced_unsat,
        'community': generate_community_structure,
        'biased': generate_biased_polarity,
    }

    clauses, _ = generators[gen_type](n_vars, n_clauses, seed)
    result = solve_instance(clauses, n_vars)

    if result is None:
        return None

    features = extract_unified_features(clauses, n_vars)

    return {
        'features': features.to_vector(),
        'label': 1 if result else 0,
        'n_vars': n_vars,
        'alpha': alpha,
        'gen_type': gen_type,
        'seed': seed,
    }


def generate_comprehensive_dataset(
    n_instances: int = 5000,
    var_sizes: List[int] = [50, 75, 100, 150],
    alpha_range: Tuple[float, float] = (3.5, 5.0),
    gen_types: List[str] = ['uniform', 'planted', 'forced_unsat', 'community', 'biased'],
    n_workers: int = 4,
    verbose: bool = True
) -> Tuple[List[List[float]], List[int], List[Dict]]:
    """
    Generate comprehensive training dataset.

    Returns:
        X: Feature vectors
        y: Labels (1=SAT, 0=UNSAT)
        metadata: Additional info about each instance
    """

    # Build task list
    tasks = []
    for idx in range(n_instances):
        n_vars = random.choice(var_sizes)
        alpha_c = predict_alpha_c(n_vars)

        # Sample α with focus on transition region
        if random.random() < 0.5:
            alpha = alpha_c + random.gauss(0, 0.3)
        else:
            alpha = random.uniform(*alpha_range)
        alpha = max(alpha_range[0], min(alpha_range[1], alpha))

        gen_type = random.choice(gen_types)
        seed = idx * 1000 + random.randint(0, 999)

        tasks.append((idx, n_vars, alpha, gen_type, seed))

    # Process in parallel
    X = []
    y = []
    metadata = []

    if verbose:
        print(f"Generating {n_instances} instances with {n_workers} workers...")

    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(generate_single_instance, task): task for task in tasks}

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                X.append(result['features'])
                y.append(result['label'])
                metadata.append({
                    'n_vars': result['n_vars'],
                    'alpha': result['alpha'],
                    'gen_type': result['gen_type'],
                })

            completed += 1
            if verbose and completed % 500 == 0:
                print(f"  Completed {completed}/{n_instances}...")

    return X, y, metadata


# =============================================================================
# Training
# =============================================================================

def train_comprehensive_model(X, y, metadata):
    """Train model on comprehensive dataset and analyze results."""

    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report
        import numpy as np
    except ImportError:
        print("scikit-learn required")
        return None

    X = np.array(X)
    y = np.array(y)

    print(f"\nDataset: {len(y)} instances")
    print(f"  SAT: {sum(y)} ({sum(y)/len(y):.1%})")
    print(f"  UNSAT: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y):.1%})")

    # Breakdown by generation type
    print("\nBy generation type:")
    for gen_type in set(m['gen_type'] for m in metadata):
        mask = [m['gen_type'] == gen_type for m in metadata]
        count = sum(mask)
        sat_count = sum(y[i] for i, m in enumerate(mask) if m)
        print(f"  {gen_type:15s}: {count:5d} instances, {sat_count/count:.1%} SAT")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Gradient Boosting (handles feature interactions well)
    print("\n" + "=" * 60)
    print("Training Gradient Boosting Classifier")
    print("=" * 60)

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        min_samples_split=10,
        subsample=0.8,
        random_state=42
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"\nCross-validation accuracy: {scores.mean():.1%} (+/- {scores.std()*2:.1%})")

    # Train final model
    model.fit(X_scaled, y)

    # Feature importance
    importances = model.feature_importances_
    names = UnifiedFeatures.feature_names()
    sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

    print("\nFeature Importances (comprehensive training):")
    print("-" * 60)
    for i in sorted_idx:
        bar = "█" * int(importances[i] * 40)
        print(f"  {names[i]:25s} {importances[i]:.3f} {bar}")

    # Analyze by generation type
    print("\n" + "=" * 60)
    print("Accuracy by Generation Type")
    print("=" * 60)

    y_pred = model.predict(X_scaled)

    for gen_type in set(m['gen_type'] for m in metadata):
        mask = np.array([m['gen_type'] == gen_type for m in metadata])
        if sum(mask) > 0:
            acc = (y_pred[mask] == y[mask]).mean()
            print(f"  {gen_type:15s}: {acc:.1%}")

    # Analyze by problem size
    print("\nAccuracy by Problem Size:")
    for n_vars in sorted(set(m['n_vars'] for m in metadata)):
        mask = np.array([m['n_vars'] == n_vars for m in metadata])
        if sum(mask) > 0:
            acc = (y_pred[mask] == y[mask]).mean()
            print(f"  n={n_vars:3d}: {acc:.1%}")

    return model, scaler


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("COMPREHENSIVE LEAK DETECTOR TRAINING")
    print("=" * 70)
    print()
    print("Training on diverse instance types to discover more leak sources:")
    print("  • Uniform random (standard)")
    print("  • Planted solution (guaranteed SAT)")
    print("  • Forced UNSAT (added contradictions)")
    print("  • Community structure (grouped variables)")
    print("  • Biased polarity (non-uniform signs)")
    print()
    print("Varying problem sizes: 50, 75, 100, 150 variables")
    print()

    # Generate dataset
    X, y, metadata = generate_comprehensive_dataset(
        n_instances=3000,  # More samples
        var_sizes=[50, 75, 100, 150],
        alpha_range=(3.5, 5.0),
        gen_types=['uniform', 'planted', 'forced_unsat', 'community', 'biased'],
        n_workers=4,
        verbose=True
    )

    if not X:
        print("No instances generated!")
        return

    # Train
    model, scaler = train_comprehensive_model(X, y, metadata)

    if model is None:
        return

    # Save
    output_path = '/Users/bryan/millennium-problems/problems/p-vs-np/phi_sat/comprehensive_detector.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
        }, f)
    print(f"\nModel saved to: {output_path}")

    # Final analysis
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("""
By training on diverse instance types, the model learns:

1. What "true randomness" looks like (uniform instances)
2. What structure SAT instances have (planted solutions)
3. What structure UNSAT instances have (forced contradictions)
4. How community structure affects satisfiability
5. How polarity bias correlates with outcomes

This diversity helps discover leak sources that wouldn't appear
in uniform-only training data.
""")


if __name__ == "__main__":
    main()
