#!/usr/bin/env python3
"""
Generation Method Fingerprinting

Key insight: The method used to generate a "random" SAT instance
is ITSELF information that leaks satisfiability.

Different generation methods leave fingerprints:
- Python random vs numpy vs C rand()
- cnfgen vs satlib vs custom generators
- Academic benchmarks vs industrial instances
- Different seeding strategies

If we can detect HOW something was generated, we gain information
about what it likely IS (SAT or UNSAT).

This is a META-LEAK: information about the information.
"""

import random
import math
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class GenerationFingerprint:
    """Features that identify how an instance was likely generated."""

    # RNG fingerprints
    chi_square_uniformity: float      # How uniform is the distribution?
    serial_correlation: float         # Correlation between adjacent choices
    runs_test_statistic: float        # Runs test for randomness
    gap_distribution_entropy: float   # Entropy of gaps between same variable

    # Tool/method fingerprints
    clause_ordering_entropy: float    # How ordered are clauses?
    variable_first_occurrence: float  # When do variables first appear?
    literal_pattern_score: float      # Repeated literal patterns
    modular_bias: float               # Bias in variable selection mod small primes

    # Structure fingerprints
    block_structure_score: float      # Evidence of blocked generation
    sequential_variable_bias: float   # Preference for sequential variables
    position_entropy: float           # Entropy of literal positions in clauses

    def to_vector(self) -> List[float]:
        return [
            self.chi_square_uniformity,
            self.serial_correlation,
            self.runs_test_statistic,
            self.gap_distribution_entropy,
            self.clause_ordering_entropy,
            self.variable_first_occurrence,
            self.literal_pattern_score,
            self.modular_bias,
            self.block_structure_score,
            self.sequential_variable_bias,
            self.position_entropy,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        return [
            'chi_square_uniformity',
            'serial_correlation',
            'runs_test_statistic',
            'gap_distribution_entropy',
            'clause_ordering_entropy',
            'variable_first_occurrence',
            'literal_pattern_score',
            'modular_bias',
            'block_structure_score',
            'sequential_variable_bias',
            'position_entropy',
        ]


def extract_generation_fingerprint(clauses: List[List[int]], n_vars: int) -> GenerationFingerprint:
    """Extract features that fingerprint the generation method."""

    if not clauses:
        return GenerationFingerprint(*([0.0] * 11))

    # Flatten all variables (absolute values)
    all_vars = [abs(lit) for clause in clauses for lit in clause]
    all_lits = [lit for clause in clauses for lit in clause]

    # === RNG FINGERPRINTS ===

    # Chi-square uniformity test
    var_counts = Counter(all_vars)
    expected = len(all_vars) / n_vars
    chi_sq = sum((var_counts.get(v, 0) - expected)**2 / expected
                 for v in range(1, n_vars + 1))
    # Normalize by degrees of freedom
    chi_square_uniformity = chi_sq / (n_vars - 1) if n_vars > 1 else 0

    # Serial correlation (correlation between adjacent variable choices)
    if len(all_vars) > 1:
        mean_var = sum(all_vars) / len(all_vars)
        numerator = sum((all_vars[i] - mean_var) * (all_vars[i+1] - mean_var)
                       for i in range(len(all_vars) - 1))
        denominator = sum((v - mean_var)**2 for v in all_vars)
        serial_correlation = numerator / denominator if denominator > 0 else 0
    else:
        serial_correlation = 0

    # Runs test (count runs of increasing/decreasing)
    if len(all_vars) > 1:
        runs = 1
        for i in range(1, len(all_vars)):
            if (all_vars[i] > all_vars[i-1]) != (all_vars[i-1] > all_vars[i-2] if i > 1 else True):
                runs += 1
        expected_runs = (2 * len(all_vars) - 1) / 3
        runs_test_statistic = (runs - expected_runs) / math.sqrt(len(all_vars)) if len(all_vars) > 0 else 0
    else:
        runs_test_statistic = 0

    # Gap distribution (distance between occurrences of same variable)
    var_positions = defaultdict(list)
    for i, v in enumerate(all_vars):
        var_positions[v].append(i)

    gaps = []
    for positions in var_positions.values():
        for i in range(1, len(positions)):
            gaps.append(positions[i] - positions[i-1])

    if gaps:
        gap_counter = Counter(gaps)
        total_gaps = len(gaps)
        gap_probs = [c / total_gaps for c in gap_counter.values()]
        gap_distribution_entropy = -sum(p * math.log2(p) for p in gap_probs if p > 0)
        # Normalize
        max_entropy = math.log2(len(gap_counter)) if len(gap_counter) > 1 else 1
        gap_distribution_entropy /= max_entropy if max_entropy > 0 else 1
    else:
        gap_distribution_entropy = 0

    # === TOOL/METHOD FINGERPRINTS ===

    # Clause ordering entropy (are clauses sorted by some criterion?)
    clause_first_vars = [min(abs(lit) for lit in clause) for clause in clauses]
    if len(clause_first_vars) > 1:
        # Count inversions (how unsorted?)
        inversions = sum(1 for i in range(len(clause_first_vars))
                        for j in range(i+1, len(clause_first_vars))
                        if clause_first_vars[i] > clause_first_vars[j])
        max_inversions = len(clause_first_vars) * (len(clause_first_vars) - 1) / 2
        clause_ordering_entropy = inversions / max_inversions if max_inversions > 0 else 0
    else:
        clause_ordering_entropy = 0.5

    # Variable first occurrence (when does each variable first appear?)
    first_occurrence = {}
    for i, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit)
            if v not in first_occurrence:
                first_occurrence[v] = i

    # Correlation between variable number and first occurrence
    vars_with_occurrence = [(v, first_occurrence[v]) for v in first_occurrence]
    if len(vars_with_occurrence) > 1:
        var_nums = [x[0] for x in vars_with_occurrence]
        occurrences = [x[1] for x in vars_with_occurrence]
        mean_v = sum(var_nums) / len(var_nums)
        mean_o = sum(occurrences) / len(occurrences)
        cov = sum((v - mean_v) * (o - mean_o) for v, o in vars_with_occurrence)
        std_v = math.sqrt(sum((v - mean_v)**2 for v in var_nums))
        std_o = math.sqrt(sum((o - mean_o)**2 for o in occurrences))
        variable_first_occurrence = cov / (std_v * std_o) if std_v > 0 and std_o > 0 else 0
    else:
        variable_first_occurrence = 0

    # Literal pattern score (repeated exact clauses or similar patterns)
    clause_tuples = [tuple(sorted(clause)) for clause in clauses]
    unique_clauses = len(set(clause_tuples))
    literal_pattern_score = 1 - unique_clauses / len(clauses) if clauses else 0

    # Modular bias (preference for variables mod small primes)
    mod_biases = []
    for prime in [2, 3, 5, 7]:
        mod_counts = Counter(v % prime for v in all_vars)
        expected_per_class = len(all_vars) / prime
        bias = sum(abs(mod_counts.get(i, 0) - expected_per_class)
                  for i in range(prime)) / len(all_vars)
        mod_biases.append(bias)
    modular_bias = sum(mod_biases) / len(mod_biases)

    # === STRUCTURE FINGERPRINTS ===

    # Block structure (are variables used in blocks?)
    block_size = 10
    block_usage = defaultdict(int)
    for v in all_vars:
        block_usage[v // block_size] += 1

    if block_usage:
        block_counts = list(block_usage.values())
        mean_block = sum(block_counts) / len(block_counts)
        block_variance = sum((c - mean_block)**2 for c in block_counts) / len(block_counts)
        block_structure_score = block_variance / (mean_block**2) if mean_block > 0 else 0
    else:
        block_structure_score = 0

    # Sequential variable bias (do adjacent clauses use similar variables?)
    if len(clauses) > 1:
        sequential_overlaps = []
        for i in range(len(clauses) - 1):
            vars1 = set(abs(lit) for lit in clauses[i])
            vars2 = set(abs(lit) for lit in clauses[i+1])
            overlap = len(vars1 & vars2) / len(vars1 | vars2) if vars1 | vars2 else 0
            sequential_overlaps.append(overlap)
        sequential_variable_bias = sum(sequential_overlaps) / len(sequential_overlaps)
    else:
        sequential_variable_bias = 0

    # Position entropy (entropy of literal positions within clauses)
    position_counts = [Counter() for _ in range(3)]  # For 3-SAT
    for clause in clauses:
        for i, lit in enumerate(clause[:3]):
            position_counts[i][lit > 0] += 1  # Track pos/neg at each position

    position_entropy = 0
    for pc in position_counts:
        total = sum(pc.values())
        if total > 0:
            probs = [c / total for c in pc.values()]
            position_entropy -= sum(p * math.log2(p) for p in probs if p > 0)
    position_entropy /= 3  # Normalize by number of positions

    return GenerationFingerprint(
        chi_square_uniformity=chi_square_uniformity,
        serial_correlation=serial_correlation,
        runs_test_statistic=runs_test_statistic,
        gap_distribution_entropy=gap_distribution_entropy,
        clause_ordering_entropy=clause_ordering_entropy,
        variable_first_occurrence=variable_first_occurrence,
        literal_pattern_score=literal_pattern_score,
        modular_bias=modular_bias,
        block_structure_score=block_structure_score,
        sequential_variable_bias=sequential_variable_bias,
        position_entropy=position_entropy,
    )


# =============================================================================
# Different Generation Methods (to fingerprint)
# =============================================================================

def generate_python_random(n_vars: int, n_clauses: int, seed: int) -> List[List[int]]:
    """Standard Python random."""
    random.seed(seed)
    clauses = []
    for _ in range(n_clauses):
        vars = random.sample(range(1, n_vars + 1), 3)
        clause = [v if random.random() > 0.5 else -v for v in vars]
        clauses.append(clause)
    return clauses


def generate_numpy_random(n_vars: int, n_clauses: int, seed: int) -> List[List[int]]:
    """NumPy random (different RNG)."""
    rng = np.random.RandomState(seed)
    clauses = []
    for _ in range(n_clauses):
        vars = rng.choice(range(1, n_vars + 1), size=3, replace=False).tolist()
        signs = rng.random(3) > 0.5
        clause = [v if s else -v for v, s in zip(vars, signs)]
        clauses.append(clause)
    return clauses


def generate_lcg_random(n_vars: int, n_clauses: int, seed: int) -> List[List[int]]:
    """Linear Congruential Generator (old-school, more predictable)."""
    # LCG parameters (MINSTD)
    a, c, m = 48271, 0, 2147483647
    state = seed % m

    def lcg_next():
        nonlocal state
        state = (a * state + c) % m
        return state / m

    clauses = []
    for _ in range(n_clauses):
        # Select 3 distinct variables
        vars = []
        while len(vars) < 3:
            v = int(lcg_next() * n_vars) + 1
            if v not in vars:
                vars.append(v)

        clause = [v if lcg_next() > 0.5 else -v for v in vars]
        clauses.append(clause)

    return clauses


def generate_sorted_output(n_vars: int, n_clauses: int, seed: int) -> List[List[int]]:
    """Generate then sort clauses (common in some tools)."""
    random.seed(seed)
    clauses = []
    for _ in range(n_clauses):
        vars = random.sample(range(1, n_vars + 1), 3)
        clause = [v if random.random() > 0.5 else -v for v in vars]
        clauses.append(sorted(clause, key=abs))  # Sort within clause

    # Sort clauses by first variable
    clauses.sort(key=lambda c: abs(c[0]))
    return clauses


def generate_sequential_blocks(n_vars: int, n_clauses: int, seed: int) -> List[List[int]]:
    """Generate in sequential blocks (industrial pattern)."""
    random.seed(seed)
    clauses = []
    block_size = max(1, n_vars // 10)

    for _ in range(n_clauses):
        # Pick a block, then pick from within it
        block_start = random.randint(0, (n_vars - 1) // block_size) * block_size
        block_vars = list(range(block_start + 1, min(block_start + block_size + 1, n_vars + 1)))

        if len(block_vars) >= 3:
            vars = random.sample(block_vars, 3)
        else:
            vars = random.sample(range(1, n_vars + 1), 3)

        clause = [v if random.random() > 0.5 else -v for v in vars]
        clauses.append(clause)

    return clauses


# =============================================================================
# Test: Can we identify generation method?
# =============================================================================

def test_fingerprinting():
    """Test if we can distinguish generation methods from fingerprints."""

    print("=" * 70)
    print("GENERATION METHOD FINGERPRINTING TEST")
    print("=" * 70)
    print()
    print("Can we identify HOW an instance was generated from its structure?")
    print("This is a META-LEAK: information about the information source.")
    print()

    generators = {
        'python_random': generate_python_random,
        'numpy_random': generate_numpy_random,
        'lcg_random': generate_lcg_random,
        'sorted_output': generate_sorted_output,
        'sequential_blocks': generate_sequential_blocks,
    }

    n_vars = 100
    n_clauses = 420  # α ≈ 4.2

    # Collect fingerprints for each method
    method_fingerprints = {name: [] for name in generators}

    print("Generating instances from each method...")
    for name, gen_func in generators.items():
        for seed in range(50):
            clauses = gen_func(n_vars, n_clauses, seed)
            fp = extract_generation_fingerprint(clauses, n_vars)
            method_fingerprints[name].append(fp.to_vector())

    # Compare fingerprints
    print("\n" + "-" * 70)
    print("Average Fingerprint by Generation Method")
    print("-" * 70)

    feature_names = GenerationFingerprint.feature_names()

    print(f"\n{'Feature':30s}", end="")
    for name in generators:
        print(f" {name[:12]:>12s}", end="")
    print()
    print("-" * (30 + 13 * len(generators)))

    for i, feat_name in enumerate(feature_names):
        print(f"{feat_name:30s}", end="")
        for name in generators:
            values = [fp[i] for fp in method_fingerprints[name]]
            mean = sum(values) / len(values)
            print(f" {mean:12.4f}", end="")
        print()

    # Train classifier to identify generation method
    print("\n" + "=" * 70)
    print("Training Classifier to Identify Generation Method")
    print("=" * 70)

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        X = []
        y = []
        for name, fps in method_fingerprints.items():
            for fp in fps:
                X.append(fp)
                y.append(name)

        X = np.array(X)
        le = LabelEncoder()
        y = le.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(clf, X_scaled, y, cv=5)

        print(f"\nGeneration method identification accuracy: {scores.mean():.1%}")
        print(f"(Random guessing would be {100/len(generators):.1%})")

        # Feature importance for method identification
        clf.fit(X_scaled, y)
        importances = clf.feature_importances_
        sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

        print("\nMost distinguishing features for method identification:")
        for i in sorted_idx[:5]:
            print(f"  {feature_names[i]:30s}: {importances[i]:.3f}")

    except ImportError:
        print("scikit-learn required for classifier training")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
If we can identify the generation METHOD, that's additional information:

1. Different RNGs have different statistical properties
2. Different tools leave different fingerprints
3. Knowing the source helps predict the outcome

This is the META-LEAK you identified:
"The method of randomization is itself a leak of information"

Even if the instance looks random, knowing HOW it was made random
gives us an edge in predicting its satisfiability.
""")


if __name__ == "__main__":
    test_fingerprinting()
