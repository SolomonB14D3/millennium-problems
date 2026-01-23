#!/usr/bin/env python3
"""
φ-Triage SAT Solver

A practical SAT solver that uses φ-predicted phase transition to:
1. Allocate time budgets based on expected difficulty
2. Order instances by predicted difficulty
3. Provide confidence scores for predictions

This doesn't claim to predict SAT/UNSAT without solving - instead it uses
the φ-structure for intelligent resource allocation.
"""

import subprocess
import tempfile
import os
import random
import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

PHI = (1 + math.sqrt(5)) / 2

# Observed data points
ALPHA_C_DATA = [
    (500, 3.573), (2000, 4.497), (4000, 4.996),
    (8000, 4.996), (12000, 5.495), (24000, 6.998),
    (32000, 6.998), (64000, 9.996),
]

class Difficulty(Enum):
    TRIVIAL = "trivial"      # |distance| > 50%, solve in milliseconds
    EASY = "easy"            # |distance| > 25%, solve in seconds
    MODERATE = "moderate"    # |distance| > 10%, may take minutes
    HARD = "hard"            # |distance| < 10%, near transition

@dataclass
class InstanceProfile:
    n_vars: int
    n_clauses: int
    alpha: float
    alpha_c: float
    distance: float
    difficulty: Difficulty
    recommended_timeout: float
    predicted_sat_prob: float

def predict_alpha_c(n: int) -> float:
    """Interpolate α_c(n) from observed data."""
    if n <= ALPHA_C_DATA[0][0]:
        return ALPHA_C_DATA[0][1]
    if n >= ALPHA_C_DATA[-1][0]:
        return ALPHA_C_DATA[-1][1] + math.log(n / ALPHA_C_DATA[-1][0]) * 2.5

    for i in range(len(ALPHA_C_DATA) - 1):
        n1, a1 = ALPHA_C_DATA[i]
        n2, a2 = ALPHA_C_DATA[i + 1]
        if n1 <= n <= n2:
            t = (math.log(n) - math.log(n1)) / (math.log(n2) - math.log(n1))
            return a1 + t * (a2 - a1)
    return 4.267

def sigmoid(x: float) -> float:
    """Sigmoid function for smooth probability estimates."""
    return 1 / (1 + math.exp(-x))

def profile_instance(n_vars: int, n_clauses: int) -> InstanceProfile:
    """
    Profile a SAT instance based on φ-predicted transition.

    Returns difficulty classification and recommended timeout.
    """
    alpha = n_clauses / n_vars
    alpha_c = predict_alpha_c(n_vars)
    distance = (alpha - alpha_c) / alpha_c

    # Classify difficulty based on distance from transition
    abs_dist = abs(distance)
    if abs_dist > 0.50:
        difficulty = Difficulty.TRIVIAL
        base_timeout = 1.0
    elif abs_dist > 0.25:
        difficulty = Difficulty.EASY
        base_timeout = 5.0
    elif abs_dist > 0.10:
        difficulty = Difficulty.MODERATE
        base_timeout = 30.0
    else:
        difficulty = Difficulty.HARD
        base_timeout = 120.0

    # Scale timeout with problem size
    size_factor = math.log(n_vars + 1) / math.log(1000)  # normalized to n=1000
    recommended_timeout = base_timeout * size_factor

    # Estimate P(SAT) using sigmoid around transition
    # Steepness depends on n (sharper transition for larger n)
    steepness = 5 + 0.001 * n_vars  # increases with n
    predicted_sat_prob = 1 - sigmoid(steepness * distance)

    return InstanceProfile(
        n_vars=n_vars,
        n_clauses=n_clauses,
        alpha=alpha,
        alpha_c=alpha_c,
        distance=distance,
        difficulty=difficulty,
        recommended_timeout=recommended_timeout,
        predicted_sat_prob=predicted_sat_prob
    )

class PhiTriageSolver:
    """
    SAT solver with φ-based triage.

    Key features:
    1. Profiles instances before solving
    2. Allocates time budgets based on difficulty
    3. Orders batch jobs by expected difficulty
    4. Reports confidence scores
    """

    def __init__(self, minisat_path: str = "minisat", total_budget: float = 60.0):
        self.minisat_path = minisat_path
        self.total_budget = total_budget
        self.stats = {
            'trivial': 0, 'easy': 0, 'moderate': 0, 'hard': 0,
            'solved': 0, 'timeout': 0, 'total_time': 0.0
        }

    def solve_one(self, cnf_path: str, timeout: Optional[float] = None) -> Tuple[Optional[bool], float, InstanceProfile]:
        """
        Solve a single instance with φ-triage.

        Returns: (result, time_taken, profile)
        """
        # Profile the instance
        n, m = self._parse_cnf(cnf_path)
        profile = profile_instance(n, m)

        # Use recommended timeout if not specified
        if timeout is None:
            timeout = profile.recommended_timeout

        # Solve
        result, elapsed = self._run_minisat(cnf_path, timeout)

        # Update stats
        self.stats[profile.difficulty.value] += 1
        if result is not None:
            self.stats['solved'] += 1
        else:
            self.stats['timeout'] += 1
        self.stats['total_time'] += elapsed

        return result, elapsed, profile

    def solve_batch(self, cnf_paths: List[str], budget: Optional[float] = None) -> List[Tuple[str, Optional[bool], float, InstanceProfile]]:
        """
        Solve multiple instances with intelligent scheduling.

        Orders by expected difficulty - trivial first, hard last.
        Allocates more time budget to hard instances.
        """
        if budget is None:
            budget = self.total_budget

        # Profile all instances
        profiles = []
        for path in cnf_paths:
            n, m = self._parse_cnf(path)
            profile = profile_instance(n, m)
            profiles.append((path, profile))

        # Sort by difficulty (trivial first)
        difficulty_order = {
            Difficulty.TRIVIAL: 0,
            Difficulty.EASY: 1,
            Difficulty.MODERATE: 2,
            Difficulty.HARD: 3
        }
        profiles.sort(key=lambda x: (difficulty_order[x[1].difficulty], -x[1].predicted_sat_prob))

        # Allocate time budget
        remaining_budget = budget
        results = []

        for path, profile in profiles:
            if remaining_budget <= 0:
                results.append((path, None, 0.0, profile))
                continue

            # Allocate time: more for hard, less for trivial
            if profile.difficulty == Difficulty.TRIVIAL:
                timeout = min(1.0, remaining_budget)
            elif profile.difficulty == Difficulty.EASY:
                timeout = min(5.0, remaining_budget * 0.1)
            elif profile.difficulty == Difficulty.MODERATE:
                timeout = min(30.0, remaining_budget * 0.3)
            else:
                timeout = remaining_budget * 0.5  # Give hard instances more time

            result, elapsed = self._run_minisat(path, timeout)
            remaining_budget -= elapsed

            self.stats[profile.difficulty.value] += 1
            if result is not None:
                self.stats['solved'] += 1
            else:
                self.stats['timeout'] += 1
            self.stats['total_time'] += elapsed

            results.append((path, result, elapsed, profile))

        return results

    def _parse_cnf(self, path: str) -> Tuple[int, int]:
        """Extract n_vars and n_clauses from CNF."""
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('p cnf'):
                    parts = line.split()
                    return int(parts[2]), int(parts[3])
        raise ValueError("No 'p cnf' line found")

    def _run_minisat(self, cnf_path: str, timeout: float) -> Tuple[Optional[bool], float]:
        """Run MiniSat with timeout."""
        with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as f:
            out_path = f.name

        try:
            start = time.time()
            result = subprocess.run(
                [self.minisat_path, cnf_path, out_path],
                capture_output=True, timeout=timeout, text=True
            )
            elapsed = time.time() - start

            output = result.stdout + result.stderr
            if 'SATISFIABLE' in output:
                return True, elapsed
            elif 'UNSATISFIABLE' in output:
                return False, elapsed
            return None, elapsed
        except subprocess.TimeoutExpired:
            return None, timeout
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def print_stats(self):
        """Print solving statistics."""
        print(f"\n{'='*50}")
        print("φ-Triage Solver Statistics")
        print(f"{'='*50}")
        print(f"By difficulty:")
        print(f"  Trivial:  {self.stats['trivial']}")
        print(f"  Easy:     {self.stats['easy']}")
        print(f"  Moderate: {self.stats['moderate']}")
        print(f"  Hard:     {self.stats['hard']}")
        print(f"\nResults:")
        print(f"  Solved:   {self.stats['solved']}")
        print(f"  Timeout:  {self.stats['timeout']}")
        print(f"  Time:     {self.stats['total_time']:.2f}s")

def demo():
    """Demonstrate φ-triage solver."""
    print("=" * 60)
    print("φ-Triage SAT Solver Demo")
    print("=" * 60)

    # Show profiling
    print("\nInstance Profiling Examples:")
    print(f"{'n':>6} {'m':>6} {'α':>6} {'α_c':>6} {'dist':>8} {'diff':>10} {'timeout':>8} {'P(SAT)':>8}")
    print("-" * 70)

    examples = [
        (500, 1000),    # Very under-constrained
        (500, 1750),    # Slightly under
        (500, 2100),    # Near transition
        (500, 3000),    # Over-constrained
        (1000, 3000),   # Under for n=1000
        (1000, 4000),   # Near transition
        (1000, 6000),   # Over-constrained
        (5000, 25000),  # Large, near transition
    ]

    for n, m in examples:
        p = profile_instance(n, m)
        print(f"{n:>6} {m:>6} {p.alpha:>6.2f} {p.alpha_c:>6.2f} {p.distance:>+7.1%} "
              f"{p.difficulty.value:>10} {p.recommended_timeout:>7.1f}s {p.predicted_sat_prob:>7.1%}")

    # Solve some instances
    print("\n" + "=" * 60)
    print("Solving Demo")
    print("=" * 60)

    try:
        subprocess.run(['minisat', '--help'], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("\nMiniSat not found. Install with: brew install minisat")
        return

    solver = PhiTriageSolver()

    # Generate and solve instances
    instances = []
    for n, alpha in [(200, 2.0), (200, 3.5), (200, 5.0), (500, 3.0), (500, 4.0)]:
        m = int(n * alpha)
        cnf = generate_random_3sat(n, m, seed=42)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
            f.write(cnf)
            instances.append(f.name)

    print(f"\nSolving {len(instances)} instances with batch scheduling...")
    results = solver.solve_batch(instances, budget=30.0)

    print(f"\n{'Path':>40} {'Result':>8} {'Time':>8} {'Difficulty':>12} {'P(SAT)':>8}")
    print("-" * 80)

    for path, result, elapsed, profile in results:
        result_str = "SAT" if result == True else "UNSAT" if result == False else "TIMEOUT"
        print(f"{os.path.basename(path):>40} {result_str:>8} {elapsed:>7.3f}s "
              f"{profile.difficulty.value:>12} {profile.predicted_sat_prob:>7.1%}")

    # Cleanup
    for path in instances:
        os.remove(path)

    solver.print_stats()

def generate_random_3sat(n: int, m: int, seed: int) -> str:
    """Generate random 3-SAT."""
    random.seed(seed)
    lines = [f"p cnf {n} {m}"]
    for _ in range(m):
        vars = random.sample(range(1, n + 1), 3)
        clause = [v if random.random() > 0.5 else -v for v in vars]
        lines.append(" ".join(map(str, clause)) + " 0")
    return "\n".join(lines)

if __name__ == "__main__":
    demo()
