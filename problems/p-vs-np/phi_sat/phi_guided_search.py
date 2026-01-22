#!/usr/bin/env python3
"""
φ-guided variable selection: Choose assignments that move α away from α_c.

The idea: After each assignment, the residual formula has new (n', m').
If we pick variables that maximize |α' - α_c|, we might escape the hard region faster.
"""

import random
import math
import time
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

# α_c prediction (same as phi_sat.py)
ALPHA_C_TABLE = [
    (50, 3.0), (100, 3.2), (200, 3.4), (500, 3.573),
    (2000, 4.497), (4000, 4.996),
]

def predict_alpha_c(n: int) -> float:
    if n <= ALPHA_C_TABLE[0][0]: return ALPHA_C_TABLE[0][1]
    if n >= ALPHA_C_TABLE[-1][0]: return ALPHA_C_TABLE[-1][1]
    for i in range(len(ALPHA_C_TABLE) - 1):
        n1, a1 = ALPHA_C_TABLE[i]
        n2, a2 = ALPHA_C_TABLE[i + 1]
        if n1 <= n <= n2:
            t = (math.log(n) - math.log(n1)) / (math.log(n2) - math.log(n1))
            return a1 + t * (a2 - a1)
    return 4.267


@dataclass
class Formula:
    """Mutable CNF formula for tracking residual state."""
    n_vars: int
    clauses: List[Set[int]]  # Each clause is a set of literals
    assignment: Dict[int, bool]  # var -> value

    @property
    def n_remaining(self) -> int:
        assigned = set(self.assignment.keys())
        all_vars = set()
        for clause in self.clauses:
            for lit in clause:
                all_vars.add(abs(lit))
        return len(all_vars - assigned)

    @property
    def m_remaining(self) -> int:
        return len(self.clauses)

    @property
    def alpha(self) -> float:
        if self.n_remaining == 0:
            return float('inf') if self.m_remaining > 0 else 0.0
        return self.m_remaining / self.n_remaining

    @property
    def alpha_c(self) -> float:
        return predict_alpha_c(self.n_remaining)

    @property
    def distance_from_transition(self) -> float:
        if self.n_remaining == 0:
            return float('inf')
        return (self.alpha - self.alpha_c) / self.alpha_c

    def assign(self, var: int, value: bool) -> bool:
        """
        Assign a variable. Returns False if conflict detected.
        Modifies clauses in place (unit propagation).
        """
        self.assignment[var] = value
        lit_true = var if value else -var
        lit_false = -lit_true

        new_clauses = []
        for clause in self.clauses:
            if lit_true in clause:
                # Clause satisfied, remove it
                continue
            elif lit_false in clause:
                # Remove false literal
                new_clause = clause - {lit_false}
                if len(new_clause) == 0:
                    # Conflict!
                    return False
                new_clauses.append(new_clause)
            else:
                new_clauses.append(clause)

        self.clauses = new_clauses
        return True

    def get_unassigned_vars(self) -> Set[int]:
        assigned = set(self.assignment.keys())
        all_vars = set()
        for clause in self.clauses:
            for lit in clause:
                all_vars.add(abs(lit))
        return all_vars - assigned

    def copy(self) -> 'Formula':
        return Formula(
            n_vars=self.n_vars,
            clauses=[clause.copy() for clause in self.clauses],
            assignment=self.assignment.copy()
        )

    def evaluate_assignment(self, var: int, value: bool) -> Tuple[float, int]:
        """
        Evaluate what α would be after this assignment.
        Returns (new_distance, clauses_satisfied)
        """
        lit_true = var if value else -var
        lit_false = -lit_true

        clauses_satisfied = 0
        clauses_remaining = 0
        conflict = False

        for clause in self.clauses:
            if lit_true in clause:
                clauses_satisfied += 1
            elif lit_false in clause:
                new_clause = clause - {lit_false}
                if len(new_clause) == 0:
                    conflict = True
                clauses_remaining += 1
            else:
                clauses_remaining += 1

        if conflict:
            return float('inf'), 0

        n_after = self.n_remaining - 1
        m_after = clauses_remaining

        if n_after == 0:
            return -float('inf') if m_after == 0 else float('inf'), clauses_satisfied

        alpha_after = m_after / n_after
        alpha_c_after = predict_alpha_c(n_after)
        distance_after = (alpha_after - alpha_c_after) / alpha_c_after

        return distance_after, clauses_satisfied


def phi_guided_solve(formula: Formula, strategy: str = "phi") -> Tuple[bool, Dict[int, bool], int]:
    """
    DPLL with φ-guided variable selection.

    Strategies:
    - "phi": Choose assignment that maximizes |distance from α_c|
    - "random": Random variable selection (baseline)
    - "most_clauses": Choose assignment that satisfies most clauses

    Returns: (sat, assignment, decisions)
    """
    decisions = 0

    def solve(f: Formula) -> Optional[Dict[int, bool]]:
        nonlocal decisions

        # Unit propagation
        changed = True
        while changed:
            changed = False
            for clause in f.clauses:
                if len(clause) == 1:
                    lit = next(iter(clause))
                    var = abs(lit)
                    value = lit > 0
                    if var not in f.assignment:
                        if not f.assign(var, value):
                            return None  # Conflict
                        changed = True
                        break

        # Check if done
        if len(f.clauses) == 0:
            return f.assignment

        # Choose variable
        unassigned = f.get_unassigned_vars()
        if not unassigned:
            return None  # Conflict (clauses remain but no variables)

        decisions += 1

        if strategy == "phi":
            # φ-guided: pick (var, value) that maximizes |distance|
            best_var = None
            best_value = None
            best_score = -float('inf')

            for var in unassigned:
                for value in [True, False]:
                    dist, satisfied = f.evaluate_assignment(var, value)
                    # We want to move AWAY from transition (maximize |distance|)
                    # Prefer negative distance (toward SAT) slightly
                    score = abs(dist) if dist != float('inf') else -1000
                    if dist < 0:
                        score += 0.1  # Slight preference for SAT direction

                    if score > best_score:
                        best_score = score
                        best_var = var
                        best_value = value

            var = best_var
            values_to_try = [best_value, not best_value]

        elif strategy == "most_clauses":
            # Satisfy most clauses first
            best_var = None
            best_value = None
            best_satisfied = -1

            for var in unassigned:
                for value in [True, False]:
                    _, satisfied = f.evaluate_assignment(var, value)
                    if satisfied > best_satisfied:
                        best_satisfied = satisfied
                        best_var = var
                        best_value = value

            var = best_var
            values_to_try = [best_value, not best_value]

        else:  # random
            var = random.choice(list(unassigned))
            values_to_try = [True, False]
            random.shuffle(values_to_try)

        # Try assignments
        for value in values_to_try:
            f_copy = f.copy()
            if f_copy.assign(var, value):
                result = solve(f_copy)
                if result is not None:
                    return result

        return None  # Both branches failed

    result = solve(formula)
    if result is not None:
        return True, result, decisions
    else:
        return False, {}, decisions


def generate_random_3sat(n: int, m: int, seed: int) -> Formula:
    random.seed(seed)
    clauses = []
    for _ in range(m):
        vars = random.sample(range(1, n + 1), 3)
        clause = set(v if random.random() > 0.5 else -v for v in vars)
        clauses.append(clause)
    return Formula(n_vars=n, clauses=clauses, assignment={})


# Test the idea
print("=" * 70)
print("φ-GUIDED VARIABLE SELECTION TEST")
print("=" * 70)
print("""
Hypothesis: Choosing assignments that move α away from α_c
should make the problem easier (escape the hard region faster).
""")

# Test on instances near the transition (the hard ones)
n = 100
alpha_c = predict_alpha_c(n)

print(f"Testing on n={n}, α_c={alpha_c:.2f}")
print(f"Instances at α = α_c (hardest region)\n")

results = {s: {"decisions": 0, "time": 0, "solved": 0} for s in ["phi", "most_clauses", "random"]}

n_tests = 20
alpha = alpha_c  # At the transition

for seed in range(n_tests):
    m = int(n * alpha)

    for strategy in ["phi", "most_clauses", "random"]:
        formula = generate_random_3sat(n, m, seed)

        start = time.time()
        sat, assignment, decisions = phi_guided_solve(formula, strategy)
        elapsed = time.time() - start

        results[strategy]["decisions"] += decisions
        results[strategy]["time"] += elapsed
        if sat:
            results[strategy]["solved"] += 1

print(f"{'Strategy':<15} {'Solved':>8} {'Avg Decisions':>15} {'Avg Time':>12}")
print("-" * 55)
for strategy in ["phi", "most_clauses", "random"]:
    r = results[strategy]
    avg_dec = r["decisions"] / n_tests
    avg_time = r["time"] / n_tests * 1000
    print(f"{strategy:<15} {r['solved']:>8} {avg_dec:>15.1f} {avg_time:>10.2f}ms")

# Compare at different α values
print(f"\n{'='*70}")
print("COMPARISON ACROSS α VALUES")
print("=" * 70)

print(f"\n{'α/α_c':<10} {'Region':<12} {'φ-guided':>12} {'random':>12} {'Improvement':>12}")
print("-" * 60)

for offset in [-0.3, -0.15, 0.0, 0.15, 0.3]:
    alpha = alpha_c * (1 + offset)
    m = int(n * alpha)

    region = "SAT" if offset < -0.1 else "UNSAT" if offset > 0.1 else "HARD"

    phi_decisions = 0
    rand_decisions = 0

    for seed in range(10):
        formula = generate_random_3sat(n, m, seed)
        _, _, dec = phi_guided_solve(formula.copy(), "phi")
        phi_decisions += dec

        formula = generate_random_3sat(n, m, seed)
        _, _, dec = phi_guided_solve(formula.copy(), "random")
        rand_decisions += dec

    improvement = rand_decisions / phi_decisions if phi_decisions > 0 else 0
    print(f"{1+offset:<10.2f} {region:<12} {phi_decisions/10:>12.1f} {rand_decisions/10:>12.1f} {improvement:>11.2f}x")

print("""
INTERPRETATION:
- If φ-guided has fewer decisions → the heuristic helps
- Biggest impact should be near the transition (HARD region)
- Easy regions may not benefit (already easy)
""")
