#!/usr/bin/env python3
"""
Bit Effect Analysis

Study how individual input bits affect outputs at each stage.

Key questions:
1. Does flipping input bit i affect output bit j with probability ≠ 0.5?
2. Are there correlations between input bit positions and output patterns?
3. Do effects compound predictably through rounds/stages?
4. Is there φ-structure in the bit dependencies?

DAT perspective:
- The "avalanche effect" is a continuous approximation
- Real implementations are discrete (finite precision, finite rounds)
- Structure exists at the discrete-continuous boundary
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BitEffectMatrix:
    """
    Matrix showing P(output_j flips | input_i flips).

    Perfect avalanche: all entries ≈ 0.5
    Any deviation is potential leak.
    """
    input_bits: int
    output_bits: int
    matrix: np.ndarray  # shape (input_bits, output_bits)
    samples_per_entry: int

    def max_deviation(self) -> Tuple[float, int, int]:
        """Find maximum deviation from 0.5."""
        deviation = np.abs(self.matrix - 0.5)
        max_dev = deviation.max()
        max_idx = np.unravel_index(deviation.argmax(), deviation.shape)
        return max_dev, max_idx[0], max_idx[1]

    def leak_score(self) -> float:
        """Overall leak score (mean absolute deviation from 0.5)."""
        return np.mean(np.abs(self.matrix - 0.5))

    def to_dict(self) -> Dict:
        return {
            'input_bits': self.input_bits,
            'output_bits': self.output_bits,
            'matrix': self.matrix.tolist(),
            'samples_per_entry': self.samples_per_entry,
            'max_deviation': self.max_deviation()[0],
            'leak_score': self.leak_score(),
        }


def compute_bit_effect_matrix(
    transform_fn,
    input_bits: int,
    output_bits: int,
    samples_per_bit: int = 1000
) -> BitEffectMatrix:
    """
    Compute the bit effect matrix for any transformation.

    Args:
        transform_fn: Function that takes bytes, returns bytes
        input_bits: Number of input bits
        output_bits: Number of output bits
        samples_per_bit: Samples to estimate each row
    """
    input_bytes = (input_bits + 7) // 8
    output_bytes = (output_bits + 7) // 8

    matrix = np.zeros((input_bits, output_bits), dtype=np.float64)

    for input_bit in range(input_bits):
        byte_idx = input_bit // 8
        bit_mask = 1 << (7 - (input_bit % 8))

        flip_counts = np.zeros(output_bits, dtype=np.int64)

        for _ in range(samples_per_bit):
            # Random base input
            base = bytearray(np.random.bytes(input_bytes))

            # Compute base output
            base_out = transform_fn(bytes(base))
            base_bits = np.unpackbits(np.frombuffer(base_out, dtype=np.uint8))[:output_bits]

            # Flip the input bit
            flipped = bytearray(base)
            flipped[byte_idx] ^= bit_mask
            flipped_out = transform_fn(bytes(flipped))
            flipped_bits = np.unpackbits(np.frombuffer(flipped_out, dtype=np.uint8))[:output_bits]

            # Count which output bits flipped
            flip_counts += (base_bits != flipped_bits).astype(np.int64)

        matrix[input_bit] = flip_counts / samples_per_bit

        if (input_bit + 1) % max(1, input_bits // 10) == 0:
            print(f"  Bit effect: {input_bit + 1}/{input_bits}")

    return BitEffectMatrix(
        input_bits=input_bits,
        output_bits=output_bits,
        matrix=matrix,
        samples_per_entry=samples_per_bit
    )


@dataclass
class BitInteractionMatrix:
    """
    Study 2-bit interactions: does flipping bits i AND j together
    produce effects different from flipping them separately?

    If output_ij ≠ output_i XOR output_j, there's nonlinear interaction.
    """
    bit_pairs: List[Tuple[int, int]]
    interaction_strengths: np.ndarray
    samples: int

    def strongest_interaction(self) -> Tuple[float, int, int]:
        """Find strongest pairwise interaction."""
        max_idx = np.argmax(np.abs(self.interaction_strengths))
        return self.interaction_strengths[max_idx], *self.bit_pairs[max_idx]


def compute_bit_interactions(
    transform_fn,
    input_bits: int,
    output_bits: int,
    num_pairs: int = 100,
    samples: int = 500
) -> BitInteractionMatrix:
    """
    Sample pairwise bit interactions.

    For each pair (i, j), measure whether flipping both together
    produces different effects than XOR of individual flips.
    """
    input_bytes = (input_bits + 7) // 8

    # Sample random pairs
    pairs = []
    for _ in range(num_pairs):
        i = np.random.randint(0, input_bits)
        j = np.random.randint(0, input_bits)
        if i != j:
            pairs.append((min(i, j), max(i, j)))
    pairs = list(set(pairs))[:num_pairs]

    interaction_strengths = []

    for i, j in pairs:
        byte_i, mask_i = i // 8, 1 << (7 - (i % 8))
        byte_j, mask_j = j // 8, 1 << (7 - (j % 8))

        nonlinear_count = 0

        for _ in range(samples):
            base = bytearray(np.random.bytes(input_bytes))

            # Four versions: base, flip_i, flip_j, flip_both
            out_base = np.unpackbits(np.frombuffer(
                transform_fn(bytes(base)), dtype=np.uint8))[:output_bits]

            flip_i = bytearray(base)
            flip_i[byte_i] ^= mask_i
            out_i = np.unpackbits(np.frombuffer(
                transform_fn(bytes(flip_i)), dtype=np.uint8))[:output_bits]

            flip_j = bytearray(base)
            flip_j[byte_j] ^= mask_j
            out_j = np.unpackbits(np.frombuffer(
                transform_fn(bytes(flip_j)), dtype=np.uint8))[:output_bits]

            flip_both = bytearray(base)
            flip_both[byte_i] ^= mask_i
            flip_both[byte_j] ^= mask_j
            out_both = np.unpackbits(np.frombuffer(
                transform_fn(bytes(flip_both)), dtype=np.uint8))[:output_bits]

            # Compute expected XOR vs actual
            expected = out_base ^ (out_i ^ out_base) ^ (out_j ^ out_base)
            actual = out_both

            # Count disagreements
            nonlinear_count += np.sum(expected != actual)

        # Normalize by output bits and samples
        interaction_strength = nonlinear_count / (samples * output_bits)
        interaction_strengths.append(interaction_strength)

    return BitInteractionMatrix(
        bit_pairs=pairs,
        interaction_strengths=np.array(interaction_strengths),
        samples=samples
    )


def analyze_round_propagation(
    round_states: List[np.ndarray],
    input_bits: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze how input bit effects propagate through rounds.

    Args:
        round_states: List of state arrays at each round
        input_bits: Original input bits

    Returns analysis of propagation patterns.
    """
    num_rounds = len(round_states)

    # Track hamming distance from initial state
    if len(round_states) == 0:
        return {'error': 'No round states provided'}

    initial = round_states[0]
    hamming_distances = []

    for state in round_states:
        hd = np.sum(state != initial)
        hamming_distances.append(hd)

    # Look for patterns in propagation
    hd = np.array(hamming_distances)

    # Does it follow exponential growth? (avalanche)
    # Does it saturate? At what round?
    saturation_threshold = 0.95 * len(initial) * 0.5  # 95% of expected random
    saturation_round = None
    for i, d in enumerate(hd):
        if d >= saturation_threshold:
            saturation_round = i
            break

    # Check for φ-related patterns
    PHI = 1.618033988749895
    phi_correlations = []
    for i in range(1, len(hd)):
        if hd[i-1] > 0:
            ratio = hd[i] / hd[i-1]
            phi_correlations.append(abs(ratio - PHI))

    return {
        'num_rounds': num_rounds,
        'hamming_distances': hamming_distances,
        'saturation_round': saturation_round,
        'phi_correlation_mean': np.mean(phi_correlations) if phi_correlations else None,
        'growth_pattern': 'exponential' if saturation_round and saturation_round < num_rounds // 2 else 'linear'
    }


def find_phi_structure(matrix: np.ndarray) -> Dict[str, Any]:
    """
    Look for φ-related structure in a bit effect matrix.

    DAT hypothesis: φ appears at boundaries between discrete and continuous.
    In crypto, the discrete operations (XOR, rotate, add) meet
    continuous statistics (avalanche, uniformity).
    """
    PHI = 1.618033988749895
    PHI_INV = 1 / PHI  # 0.618...

    # Analyze deviation distribution
    deviations = np.abs(matrix - 0.5)

    # Check if deviation magnitudes cluster near φ-related values
    dev_values = deviations.flatten()
    dev_values = dev_values[dev_values > 0.001]  # Filter near-zero

    if len(dev_values) == 0:
        return {'phi_structure': False, 'reason': 'No significant deviations'}

    # Look for ratios between deviation levels
    sorted_devs = np.sort(dev_values)
    ratios = sorted_devs[1:] / sorted_devs[:-1]

    phi_ratio_count = np.sum(np.abs(ratios - PHI) < 0.1)
    phi_inv_ratio_count = np.sum(np.abs(ratios - PHI_INV) < 0.1)

    # Check spatial patterns (rows/columns)
    row_means = np.mean(deviations, axis=1)
    col_means = np.mean(deviations, axis=0)

    # Do any rows/columns show φ-spacing?
    row_peaks = np.where(row_means > np.mean(row_means) + np.std(row_means))[0]
    col_peaks = np.where(col_means > np.mean(col_means) + np.std(col_means))[0]

    phi_spacing_rows = []
    for i in range(len(row_peaks) - 1):
        spacing = row_peaks[i+1] - row_peaks[i]
        if spacing > 0:
            ratio = row_peaks[i+1] / row_peaks[i] if row_peaks[i] > 0 else 0
            if abs(ratio - PHI) < 0.2:
                phi_spacing_rows.append((row_peaks[i], row_peaks[i+1]))

    return {
        'phi_structure': len(phi_spacing_rows) > 0 or phi_ratio_count > len(ratios) * 0.1,
        'phi_ratio_count': int(phi_ratio_count),
        'phi_inv_ratio_count': int(phi_inv_ratio_count),
        'total_ratios': len(ratios),
        'phi_spacing_rows': phi_spacing_rows,
        'row_peaks': row_peaks.tolist(),
        'col_peaks': col_peaks.tolist(),
    }


if __name__ == "__main__":
    import hashlib

    print("="*60)
    print("BIT EFFECT ANALYSIS - SHA256")
    print("="*60)

    def sha256_transform(data: bytes) -> bytes:
        return hashlib.sha256(data).digest()

    # Small test: 32-bit input, 256-bit output
    print("\nComputing bit effect matrix (32-bit input)...")
    bem = compute_bit_effect_matrix(sha256_transform, 32, 256, samples_per_bit=200)

    max_dev, i, j = bem.max_deviation()
    print(f"\nResults:")
    print(f"  Max deviation from 0.5: {max_dev:.4f} at ({i}, {j})")
    print(f"  Leak score: {bem.leak_score():.4f}")

    # Look for φ-structure
    print("\nSearching for φ-structure...")
    phi_result = find_phi_structure(bem.matrix)
    print(f"  φ-structure detected: {phi_result['phi_structure']}")
    print(f"  φ-ratio count: {phi_result['phi_ratio_count']}/{phi_result['total_ratios']}")

    # Bit interactions
    print("\n" + "="*60)
    print("BIT INTERACTION ANALYSIS")
    print("="*60)

    print("\nComputing pairwise interactions (32-bit input)...")
    interactions = compute_bit_interactions(sha256_transform, 32, 256, num_pairs=50, samples=100)

    strength, bi, bj = interactions.strongest_interaction()
    print(f"\nStrongest interaction: {strength:.4f} between bits {bi} and {bj}")
    print(f"Mean interaction strength: {np.mean(interactions.interaction_strengths):.4f}")
