#!/usr/bin/env python3
"""
Level-by-Level ML Trainer

DAT Philosophy:
1. Start simple - understand the variables at each level
2. Find which variables leak information
3. Build complexity gradually
4. Look for φ-structure at boundaries

Levels:
- Level 1: 1-bit keys (trivial - learn the framework)
- Level 2: 2-bit keys (4 possibilities)
- Level 4: 4-bit keys (16 possibilities)
- Level 8: 8-bit keys (256 possibilities)
- ...
- Level 64: 64-bit keys (Bitcoin puzzle #64 territory)
- Level 256: Full keys

At each level:
1. Generate training data (inputs, outputs, all intermediates)
2. Train models to predict input bits from outputs + features
3. Measure accuracy - anything above random is a leak
4. Identify which features contribute to the leak
5. Carry insights to the next level
"""

import numpy as np
import os
import sys
import json
import pickle
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.sha256_study import sha256_instrumented, generate_training_data as gen_sha256_data
from primitives.ecdsa_study import scalar_multiply_instrumented, G


@dataclass
class LevelResult:
    """Results from training at one level."""
    level: int  # bit length
    num_samples: int
    accuracy: float  # Overall accuracy
    bit_accuracies: List[float]  # Per-bit accuracy
    random_baseline: float  # Expected random accuracy
    leak_strength: float  # accuracy - baseline
    best_features: List[str]  # Which features helped most
    training_time_s: float
    notes: str


@dataclass
class LeakReport:
    """Summary of discovered leaks across levels."""
    levels_tested: List[int]
    results: List[LevelResult]
    strongest_leak_level: int
    strongest_leak_strength: float
    pattern_notes: str


class LevelTrainer:
    """
    Train ML models level-by-level to find cryptographic leaks.
    """

    def __init__(self, target: str = 'sha256', output_dir: str = None):
        """
        Args:
            target: Which primitive to study ('sha256', 'ecdsa', 'pipeline')
            output_dir: Where to save results
        """
        self.target = target
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), '..', 'data', target
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.results: List[LevelResult] = []

    def generate_data(self, bit_length: int, num_samples: int) -> Dict[str, np.ndarray]:
        """Generate training data for a given bit length."""
        print(f"\nGenerating {num_samples} samples at {bit_length} bits...")

        if self.target == 'sha256':
            return self._generate_sha256_data(bit_length, num_samples)
        elif self.target == 'ecdsa':
            return self._generate_ecdsa_data(bit_length, num_samples)
        else:
            raise ValueError(f"Unknown target: {self.target}")

    def _generate_sha256_data(self, bit_length: int, num_samples: int) -> Dict[str, np.ndarray]:
        """Generate SHA256 training data."""
        byte_length = (bit_length + 7) // 8

        inputs = np.zeros((num_samples, bit_length), dtype=np.uint8)
        outputs = np.zeros((num_samples, 256), dtype=np.uint8)
        timings = np.zeros(num_samples, dtype=np.int64)
        hamming_trajectories = np.zeros((num_samples, 64), dtype=np.int32)

        for i in range(num_samples):
            # Generate random input of specified bit length
            data = np.random.bytes(byte_length)

            # Mask to exact bit length
            data_array = np.array(list(data), dtype=np.uint8)
            if bit_length % 8 != 0:
                mask = (1 << (bit_length % 8)) - 1
                data_array[0] &= mask
            data = bytes(data_array)

            trace = sha256_instrumented(data)

            # Extract bits (padded to bit_length)
            input_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            inputs[i, :len(input_bits)] = input_bits[:bit_length]
            outputs[i] = trace.output_bits
            timings[i] = trace.timing_ns
            hamming_trajectories[i] = trace.get_hamming_trajectory()

            if (i + 1) % max(1, num_samples // 10) == 0:
                print(f"  {i+1}/{num_samples}")

        return {
            'inputs': inputs,
            'outputs': outputs,
            'timings': timings,
            'hamming_trajectories': hamming_trajectories,
        }

    def _generate_ecdsa_data(self, bit_length: int, num_samples: int) -> Dict[str, np.ndarray]:
        """Generate ECDSA training data."""
        inputs = np.zeros((num_samples, bit_length), dtype=np.uint8)
        outputs = np.zeros((num_samples, 512), dtype=np.uint8)
        timings = np.zeros(num_samples, dtype=np.int64)
        num_additions = np.zeros(num_samples, dtype=np.int32)

        for i in range(num_samples):
            # Generate scalar with specified bit length
            max_val = (1 << bit_length) - 1
            k = np.random.randint(1, max_val + 1)

            trace = scalar_multiply_instrumented(k)

            # Extract bits
            for j in range(bit_length):
                inputs[i, j] = (k >> (bit_length - 1 - j)) & 1

            outputs[i] = trace.result_point.to_bits()
            timings[i] = trace.total_timing_ns
            num_additions[i] = trace.num_additions

            if (i + 1) % max(1, num_samples // 10) == 0:
                print(f"  {i+1}/{num_samples}")

        return {
            'inputs': inputs,
            'outputs': outputs,
            'timings': timings,
            'num_additions': num_additions,
        }

    def train_simple_model(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Train simple ML model to predict input bits from features.

        Uses multiple approaches:
        1. Linear correlation (simplest)
        2. Decision tree (captures nonlinear)
        3. Neural network (if available)

        Returns accuracy metrics.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        inputs = data['inputs']
        outputs = data['outputs']
        timings = data['timings'].reshape(-1, 1)

        num_samples, num_input_bits = inputs.shape

        # Build feature matrix
        # Features: output bits, timing, any other measured quantities
        features_list = [outputs]

        # Normalize timing
        timing_norm = (timings - timings.mean()) / (timings.std() + 1e-8)
        features_list.append(timing_norm)

        if 'hamming_trajectories' in data:
            ht = data['hamming_trajectories']
            ht_norm = (ht - ht.mean(axis=0)) / (ht.std(axis=0) + 1e-8)
            features_list.append(ht_norm)

        if 'num_additions' in data:
            na = data['num_additions'].reshape(-1, 1)
            na_norm = (na - na.mean()) / (na.std() + 1e-8)
            features_list.append(na_norm)

        features = np.hstack(features_list)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, inputs, test_size=0.2, random_state=42
        )

        results = {
            'num_samples': num_samples,
            'num_features': features.shape[1],
            'num_input_bits': num_input_bits,
            'bit_accuracies': [],
            'models': {},
        }

        # Train per-bit predictors
        for bit_idx in range(num_input_bits):
            y_train_bit = y_train[:, bit_idx]
            y_test_bit = y_test[:, bit_idx]

            # Skip if all same value
            if len(np.unique(y_train_bit)) < 2:
                results['bit_accuracies'].append(1.0 if np.mean(y_test_bit) < 0.5 else 0.0)
                continue

            # Logistic regression
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train, y_train_bit)
            lr_pred = lr.predict(X_test)
            lr_acc = accuracy_score(y_test_bit, lr_pred)

            # Decision tree
            dt = DecisionTreeClassifier(max_depth=10, random_state=42)
            dt.fit(X_train, y_train_bit)
            dt_pred = dt.predict(X_test)
            dt_acc = accuracy_score(y_test_bit, dt_pred)

            # Take best
            best_acc = max(lr_acc, dt_acc)
            results['bit_accuracies'].append(best_acc)

        results['mean_accuracy'] = np.mean(results['bit_accuracies'])
        results['random_baseline'] = 0.5
        results['leak_strength'] = results['mean_accuracy'] - results['random_baseline']

        return results

    def train_level(self, bit_length: int, num_samples: int = 10000) -> LevelResult:
        """Train at a single level and return results."""
        import time

        start = time.time()

        # Generate data
        data = self.generate_data(bit_length, num_samples)

        # Train model
        model_results = self.train_simple_model(data)

        elapsed = time.time() - start

        # Identify best features (simplified)
        best_features = ['output_bits']
        if model_results['leak_strength'] > 0.01:
            best_features.append('timing')

        result = LevelResult(
            level=bit_length,
            num_samples=num_samples,
            accuracy=model_results['mean_accuracy'],
            bit_accuracies=model_results['bit_accuracies'],
            random_baseline=0.5,
            leak_strength=model_results['leak_strength'],
            best_features=best_features,
            training_time_s=elapsed,
            notes=f"Features used: {model_results['num_features']}"
        )

        self.results.append(result)
        return result

    def train_all_levels(self, levels: List[int] = None, samples_per_level: int = 5000) -> LeakReport:
        """
        Train across all specified levels.

        Default levels follow powers of 2 up to 64.
        """
        if levels is None:
            levels = [1, 2, 4, 8, 16, 32, 64]

        print("="*60)
        print(f"LEVEL-BY-LEVEL TRAINING: {self.target.upper()}")
        print("="*60)

        for level in levels:
            print(f"\n{'='*60}")
            print(f"LEVEL {level} BITS")
            print(f"{'='*60}")

            result = self.train_level(level, samples_per_level)

            print(f"\nResults for {level}-bit:")
            print(f"  Accuracy: {result.accuracy:.4f}")
            print(f"  Random baseline: {result.random_baseline:.4f}")
            print(f"  Leak strength: {result.leak_strength:+.4f}")
            print(f"  Training time: {result.training_time_s:.1f}s")

            # Check for significant leak
            if result.leak_strength > 0.05:
                print(f"  *** SIGNIFICANT LEAK DETECTED ***")
            elif result.leak_strength > 0.01:
                print(f"  * Weak leak detected")

        # Summary
        strongest = max(self.results, key=lambda r: r.leak_strength)

        report = LeakReport(
            levels_tested=levels,
            results=self.results,
            strongest_leak_level=strongest.level,
            strongest_leak_strength=strongest.leak_strength,
            pattern_notes=self._analyze_patterns()
        )

        # Save report
        self._save_report(report)

        return report

    def _analyze_patterns(self) -> str:
        """Analyze patterns across levels (DAT perspective)."""
        if len(self.results) < 2:
            return "Need more levels for pattern analysis"

        # Check if leak strength follows any pattern with level
        levels = [r.level for r in self.results]
        strengths = [r.leak_strength for r in self.results]

        notes = []

        # Does leak decrease with level? (expected for good crypto)
        if all(strengths[i] >= strengths[i+1] for i in range(len(strengths)-1)):
            notes.append("Leak strength decreases with level (expected)")
        elif any(strengths[i] < strengths[i+1] for i in range(len(strengths)-1)):
            notes.append("ANOMALY: Leak strength increases at some levels")

        # Check for φ-related patterns (DAT signature)
        PHI = 1.618033988749895
        for i in range(len(levels) - 1):
            ratio = levels[i+1] / levels[i]
            if abs(ratio - PHI) < 0.1:
                if abs(strengths[i] / (strengths[i+1] + 1e-8) - PHI) < 0.5:
                    notes.append(f"φ-scaling hint at levels {levels[i]}-{levels[i+1]}")

        return "; ".join(notes) if notes else "No obvious patterns"

    def _save_report(self, report: LeakReport):
        """Save report to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"leak_report_{timestamp}.json")

        # Convert to serializable format
        report_dict = {
            'levels_tested': report.levels_tested,
            'strongest_leak_level': report.strongest_leak_level,
            'strongest_leak_strength': report.strongest_leak_strength,
            'pattern_notes': report.pattern_notes,
            'results': [asdict(r) for r in report.results]
        }

        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2)

        print(f"\nReport saved to: {filename}")


def main():
    """Run level-by-level training."""
    import argparse

    parser = argparse.ArgumentParser(description='Level-by-level crypto leak training')
    parser.add_argument('--target', choices=['sha256', 'ecdsa'], default='sha256',
                        help='Which primitive to study')
    parser.add_argument('--levels', type=int, nargs='+', default=[1, 2, 4, 8, 16],
                        help='Bit lengths to test')
    parser.add_argument('--samples', type=int, default=5000,
                        help='Samples per level')

    args = parser.parse_args()

    trainer = LevelTrainer(target=args.target)
    report = trainer.train_all_levels(levels=args.levels, samples_per_level=args.samples)

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Target: {args.target}")
    print(f"Levels tested: {report.levels_tested}")
    print(f"Strongest leak: {report.strongest_leak_strength:+.4f} at {report.strongest_leak_level} bits")
    print(f"Pattern notes: {report.pattern_notes}")


if __name__ == "__main__":
    main()
