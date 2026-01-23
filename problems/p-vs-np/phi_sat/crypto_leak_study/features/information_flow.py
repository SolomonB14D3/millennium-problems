#!/usr/bin/env python3
"""
Information Flow Tracer

Map exactly where each input bit's information goes through SHA256.

For each input bit i:
- At round r, which state bits are "influenced" by bit i?
- How does influence spread vs concentrate?
- Where is the signal vs noise?

This gives us the GEOMETRY of information flow.
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.sha256_study import sha256_instrumented


def trace_single_bit_influence(
    bit_position: int,
    input_bytes: int = 4,  # 32 bits
    num_samples: int = 500
) -> Dict[str, np.ndarray]:
    """
    Trace how flipping a single input bit affects each round's state.

    Returns influence map: for each round, probability that each state bit flips.
    """
    n_rounds = 64
    state_bits = 256

    # influence[round, state_bit] = P(state_bit flips | input_bit flips)
    influence = np.zeros((n_rounds, state_bits), dtype=np.float64)

    byte_idx = bit_position // 8
    bit_mask = 1 << (7 - (bit_position % 8))

    for _ in range(num_samples):
        # Random base input
        base = bytearray(np.random.bytes(input_bytes))

        # Flipped version
        flipped = bytearray(base)
        flipped[byte_idx] ^= bit_mask

        # Run both
        trace_base = sha256_instrumented(bytes(base))
        trace_flip = sha256_instrumented(bytes(flipped))

        # Compare each round's state
        for r in range(min(n_rounds, len(trace_base.round_states))):
            state_base = trace_base.round_states[r].to_array()
            state_flip = trace_flip.round_states[r].to_array()

            # Which bits differ?
            diff = (state_base != state_flip).astype(np.float64)
            influence[r] += diff

    influence /= num_samples

    return {
        'bit_position': bit_position,
        'influence_map': influence,
        'per_round_spread': np.sum(influence > 0.1, axis=1),  # bits with >10% influence
        'per_round_total': np.sum(influence, axis=1),  # total influence mass
    }


def trace_all_input_bits(
    input_bits: int = 32,
    num_samples: int = 200
) -> Dict[str, np.ndarray]:
    """
    Trace influence for all input bits.

    Returns:
        influence_tensor: (input_bits, rounds, state_bits)
        Shows complete information flow map.
    """
    input_bytes = (input_bits + 7) // 8
    n_rounds = 64
    state_bits = 256

    influence_tensor = np.zeros((input_bits, n_rounds, state_bits), dtype=np.float32)

    print(f"Tracing {input_bits} input bits through {n_rounds} rounds...")

    for bit in range(input_bits):
        result = trace_single_bit_influence(bit, input_bytes, num_samples)
        influence_tensor[bit] = result['influence_map']

        if (bit + 1) % max(1, input_bits // 8) == 0:
            print(f"  Traced bit {bit + 1}/{input_bits}")

    return {
        'influence_tensor': influence_tensor,
        'input_bits': input_bits,
        'n_rounds': n_rounds,
        'state_bits': state_bits,
    }


def analyze_information_geometry(influence_tensor: np.ndarray) -> Dict:
    """
    Analyze the geometry of information flow.

    Questions:
    - Does influence spread uniformly or concentrate?
    - Are there "attractor" state bits that collect information?
    - Is there round-to-round structure (φ-related)?
    """
    input_bits, n_rounds, state_bits = influence_tensor.shape

    results = {}

    # 1. Spread analysis: How many state bits are "active" per input bit per round?
    active_threshold = 0.1  # 10% flip probability = "active"
    active_counts = np.sum(influence_tensor > active_threshold, axis=2)  # (input_bits, rounds)

    results['spread_per_round'] = {
        'mean': np.mean(active_counts, axis=0).tolist(),  # average over input bits
        'std': np.std(active_counts, axis=0).tolist(),
    }

    # 2. Concentration: Which state bits receive the most influence?
    total_influence_per_state = np.sum(influence_tensor, axis=(0, 1))  # sum over inputs and rounds
    top_state_bits = np.argsort(total_influence_per_state)[::-1][:20]

    results['top_influenced_state_bits'] = [
        {'bit': int(b), 'total_influence': float(total_influence_per_state[b])}
        for b in top_state_bits
    ]

    # 3. Round-to-round growth: How does spread change?
    mean_spread = np.mean(active_counts, axis=0)
    growth_ratios = []
    for r in range(1, n_rounds):
        if mean_spread[r-1] > 0:
            growth_ratios.append(mean_spread[r] / mean_spread[r-1])

    results['growth_ratios'] = growth_ratios

    # Check for φ-structure in growth
    PHI = 1.618033988749895
    phi_matches = sum(1 for r in growth_ratios if abs(r - PHI) < 0.3)
    phi_inv_matches = sum(1 for r in growth_ratios if abs(r - 1/PHI) < 0.2)

    results['phi_analysis'] = {
        'phi_ratio_matches': phi_matches,
        'phi_inv_matches': phi_inv_matches,
        'total_ratios': len(growth_ratios),
        'mean_growth_ratio': float(np.mean(growth_ratios)) if growth_ratios else 0,
    }

    # 4. Saturation: At what round does spread plateau?
    saturation_round = None
    for r in range(1, n_rounds):
        if mean_spread[r] >= 0.9 * mean_spread[-1]:  # 90% of final spread
            saturation_round = r
            break

    results['saturation_round'] = saturation_round

    # 5. Input bit correlations: Do nearby input bits influence similar state bits?
    if input_bits >= 2:
        input_correlations = np.zeros((input_bits, input_bits))
        for i in range(input_bits):
            for j in range(input_bits):
                # Correlation between influence patterns
                flat_i = influence_tensor[i].flatten()
                flat_j = influence_tensor[j].flatten()
                if np.std(flat_i) > 0 and np.std(flat_j) > 0:
                    input_correlations[i, j] = np.corrcoef(flat_i, flat_j)[0, 1]

        results['input_bit_correlation_mean'] = float(np.mean(np.abs(input_correlations)))
        results['adjacent_bit_correlation'] = float(np.mean([
            input_correlations[i, i+1] for i in range(input_bits - 1)
        ]))

    return results


def find_information_concentration(influence_tensor: np.ndarray) -> Dict:
    """
    Find where information concentrates — these are the key features.

    For each input bit, find:
    - Which round has the most "localized" influence?
    - Which state bits are "dedicated" to that input bit?
    """
    input_bits, n_rounds, state_bits = influence_tensor.shape

    concentrations = []

    for bit in range(input_bits):
        bit_influence = influence_tensor[bit]  # (rounds, state_bits)

        # For each round, compute "concentration" = how localized is the influence?
        # High concentration = few bits strongly influenced
        # Low concentration = many bits weakly influenced

        round_concentrations = []
        for r in range(n_rounds):
            round_inf = bit_influence[r]
            if np.sum(round_inf) > 0:
                # Entropy-based concentration (lower = more concentrated)
                p = round_inf / (np.sum(round_inf) + 1e-10)
                p = p[p > 0]
                entropy = -np.sum(p * np.log2(p + 1e-10))
                concentration = 1.0 / (1.0 + entropy)  # invert so higher = more concentrated
            else:
                concentration = 0
            round_concentrations.append(concentration)

        best_round = np.argmax(round_concentrations)

        # Find the most influenced state bits at the best round
        best_round_influence = bit_influence[best_round]
        top_state_bits = np.argsort(best_round_influence)[::-1][:5]

        concentrations.append({
            'input_bit': bit,
            'best_round': int(best_round),
            'best_concentration': float(round_concentrations[best_round]),
            'top_state_bits': top_state_bits.tolist(),
            'top_influences': best_round_influence[top_state_bits].tolist(),
        })

    return {
        'per_bit_concentration': concentrations,
        'mean_best_round': np.mean([c['best_round'] for c in concentrations]),
        'mean_concentration': np.mean([c['best_concentration'] for c in concentrations]),
    }


def identify_signal_vs_noise(influence_tensor: np.ndarray) -> Dict:
    """
    Identify which state bits carry signal vs noise.

    Signal: State bits that are strongly influenced by specific input bits
    Noise: State bits that are weakly/uniformly influenced by all input bits
    """
    input_bits, n_rounds, state_bits = influence_tensor.shape

    # For each state bit at each round, compute:
    # - Total influence from all input bits
    # - Variance of influence across input bits (high = signal, low = noise)

    signal_map = np.zeros((n_rounds, state_bits))
    noise_map = np.zeros((n_rounds, state_bits))

    for r in range(n_rounds):
        for s in range(state_bits):
            influences = influence_tensor[:, r, s]  # from all input bits

            total = np.sum(influences)
            variance = np.var(influences)

            if total > 0.1:  # only consider active state bits
                # High variance relative to mean = signal (specific input bits matter)
                # Low variance = noise (all input bits affect equally)
                signal_score = variance / (np.mean(influences) + 1e-10)
                signal_map[r, s] = signal_score
                noise_map[r, s] = 1.0 / (1.0 + signal_score)

    # Find top signal and noise locations
    signal_flat = signal_map.flatten()
    noise_flat = noise_map.flatten()

    top_signal_idx = np.argsort(signal_flat)[::-1][:20]
    top_noise_idx = np.argsort(noise_flat)[::-1][:20]

    def idx_to_round_bit(idx):
        return int(idx // state_bits), int(idx % state_bits)

    return {
        'signal_locations': [
            {'round': idx_to_round_bit(i)[0], 'state_bit': idx_to_round_bit(i)[1],
             'signal_score': float(signal_flat[i])}
            for i in top_signal_idx
        ],
        'noise_locations': [
            {'round': idx_to_round_bit(i)[0], 'state_bit': idx_to_round_bit(i)[1],
             'noise_score': float(noise_flat[i])}
            for i in top_noise_idx
        ],
        'total_signal_mass': float(np.sum(signal_map)),
        'total_noise_mass': float(np.sum(noise_map)),
        'signal_to_noise_ratio': float(np.sum(signal_map) / (np.sum(noise_map) + 1e-10)),
    }


def run_full_analysis(input_bits: int = 32, samples: int = 300):
    """Run complete information flow analysis."""

    print("="*70)
    print(f"INFORMATION FLOW ANALYSIS - {input_bits} INPUT BITS")
    print("="*70)

    # Trace all bits
    flow_data = trace_all_input_bits(input_bits, samples)
    influence_tensor = flow_data['influence_tensor']

    print("\n" + "="*70)
    print("GEOMETRY ANALYSIS")
    print("="*70)

    geometry = analyze_information_geometry(influence_tensor)

    print(f"\nSpread saturation round: {geometry['saturation_round']}")
    print(f"Mean growth ratio: {geometry['phi_analysis']['mean_growth_ratio']:.3f}")
    print(f"φ-like ratios: {geometry['phi_analysis']['phi_ratio_matches']}/{geometry['phi_analysis']['total_ratios']}")

    if 'adjacent_bit_correlation' in geometry:
        print(f"Adjacent input bit correlation: {geometry['adjacent_bit_correlation']:.3f}")

    print("\nTop influenced state bits:")
    for item in geometry['top_influenced_state_bits'][:10]:
        print(f"  State bit {item['bit']}: {item['total_influence']:.2f}")

    print("\n" + "="*70)
    print("CONCENTRATION ANALYSIS")
    print("="*70)

    concentration = find_information_concentration(influence_tensor)

    print(f"\nMean best round for concentration: {concentration['mean_best_round']:.1f}")
    print(f"Mean concentration score: {concentration['mean_concentration']:.3f}")

    print("\nPer-bit concentration (first 8):")
    for c in concentration['per_bit_concentration'][:8]:
        print(f"  Input bit {c['input_bit']}: best at round {c['best_round']}, "
              f"top state bits: {c['top_state_bits'][:3]}")

    print("\n" + "="*70)
    print("SIGNAL vs NOISE")
    print("="*70)

    signal_noise = identify_signal_vs_noise(influence_tensor)

    print(f"\nSignal-to-noise ratio: {signal_noise['signal_to_noise_ratio']:.2f}")

    print("\nTop signal locations (high specificity):")
    for loc in signal_noise['signal_locations'][:10]:
        print(f"  Round {loc['round']}, state bit {loc['state_bit']}: {loc['signal_score']:.2f}")

    print("\nTop noise locations (uniform influence):")
    for loc in signal_noise['noise_locations'][:5]:
        print(f"  Round {loc['round']}, state bit {loc['state_bit']}: {loc['noise_score']:.2f}")

    return {
        'flow_data': flow_data,
        'geometry': geometry,
        'concentration': concentration,
        'signal_noise': signal_noise,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bits', type=int, default=32)
    parser.add_argument('--samples', type=int, default=300)
    args = parser.parse_args()

    results = run_full_analysis(args.bits, args.samples)
