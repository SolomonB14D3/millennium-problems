#!/usr/bin/env python3
"""
Mask Interaction Analysis

The puzzle creator said: "consecutive keys from a deterministic wallet
(masked with leading 000...0001 to set difficulty)"

Key insight: If full keys have structure (consecutive HD wallet),
the mask operation might leak that structure.

Analyze:
1. Relationships between consecutive solved puzzle keys
2. Patterns in the low bits across puzzles
3. Whether the mask reveals HD derivation structure
"""

import numpy as np
from typing import Dict, List, Tuple

# Solved puzzle keys (from user's RNG discovery)
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


def analyze_consecutive_differences():
    """Analyze differences between consecutive puzzle keys."""

    print("="*70)
    print("CONSECUTIVE KEY DIFFERENCES")
    print("="*70)

    puzzles = sorted(PUZZLE_KEYS.keys())

    print(f"\n{'Puzzle':<8} {'Key (hex)':<20} {'Diff from prev':<20} {'Ratio':<10}")
    print("-"*60)

    prev_key = None
    differences = []
    ratios = []

    for p in puzzles:
        key = PUZZLE_KEYS[p]

        if prev_key is not None:
            diff = key - prev_key
            ratio = key / prev_key if prev_key > 0 else 0
            differences.append(diff)
            ratios.append(ratio)
            print(f"{p:<8} {hex(key):<20} {diff:<20} {ratio:.3f}")
        else:
            print(f"{p:<8} {hex(key):<20} {'--':<20} {'--':<10}")

        prev_key = key

    print(f"\nMean ratio: {np.mean(ratios):.3f}")
    print(f"Std ratio: {np.std(ratios):.3f}")
    print(f"Expected if random: ~2.0 (doubling per bit)")

    return differences, ratios


def analyze_low_bits_patterns():
    """Look for patterns in the low bits across all puzzles."""

    print("\n" + "="*70)
    print("LOW BITS ANALYSIS")
    print("="*70)

    # For each puzzle, extract low 8 bits
    print(f"\n{'Puzzle':<8} {'Key':<20} {'Low 8 bits':<12} {'Low 4 bits':<12}")
    print("-"*55)

    low_8_bits = []
    low_4_bits = []

    for p in sorted(PUZZLE_KEYS.keys()):
        key = PUZZLE_KEYS[p]
        low8 = key & 0xFF
        low4 = key & 0xF
        low_8_bits.append(low8)
        low_4_bits.append(low4)
        print(f"{p:<8} {hex(key):<20} {low8:<12} {low4:<12}")

    # Check for patterns
    print(f"\nLow 8 bits distribution:")
    unique_low8 = len(set(low_8_bits))
    print(f"  Unique values: {unique_low8}/40")

    # Check for sequential patterns
    print(f"\nSequential pattern check:")
    for i in range(len(low_8_bits) - 1):
        if low_8_bits[i+1] == (low_8_bits[i] + 1) % 256:
            print(f"  Puzzles {i+1} → {i+2}: consecutive low bytes!")

    return low_8_bits, low_4_bits


def analyze_bit_positions():
    """Analyze which bit positions are set across puzzles."""

    print("\n" + "="*70)
    print("BIT POSITION ANALYSIS")
    print("="*70)

    # For each bit position, count how often it's set
    max_bits = 40  # Up to puzzle 40
    bit_counts = np.zeros(max_bits, dtype=int)

    for p, key in PUZZLE_KEYS.items():
        for bit in range(p):  # Only count bits within puzzle's range
            if (key >> bit) & 1:
                bit_counts[bit] += 1

    print(f"\nBit position frequency (how often each bit is 1):")
    for bit in range(16):
        # Count how many puzzles have this bit in range
        puzzles_with_bit = sum(1 for p in PUZZLE_KEYS.keys() if p > bit)
        freq = bit_counts[bit] / puzzles_with_bit if puzzles_with_bit > 0 else 0
        print(f"  Bit {bit}: {bit_counts[bit]}/{puzzles_with_bit} = {freq:.2%}")

    return bit_counts


def analyze_mask_interaction():
    """
    If keys are from HD wallet, masked to N bits:

    full_key_i = derive(master, i)
    puzzle_i = full_key_i mod 2^N

    The question: do consecutive full keys have patterns
    that survive the mod operation?
    """

    print("\n" + "="*70)
    print("MASK INTERACTION ANALYSIS")
    print("="*70)

    # If full keys are consecutive, they differ by ~constant
    # After mod 2^N, this might show as patterns

    # Look at key[i] mod 2^j for various j < i
    print("\nCross-puzzle modular patterns:")
    print("If keys are related, key[i] mod 2^j might relate to puzzle j's key")

    for j in range(1, 9):  # Check mod 2^1 through 2^8
        mod_val = 2 ** j
        print(f"\n  mod 2^{j} = {mod_val}:")

        matches = 0
        total = 0

        for i in range(j+1, min(j+10, 41)):
            if i in PUZZLE_KEYS and j in PUZZLE_KEYS:
                key_i = PUZZLE_KEYS[i]
                key_j = PUZZLE_KEYS[j]

                # Does key[i] mod 2^j equal key[j]?
                key_i_mod = key_i % mod_val

                if key_i_mod == key_j:
                    matches += 1
                    print(f"    Puzzle {i}: {hex(key_i)} mod {mod_val} = {key_i_mod} == puzzle {j} key!")

                total += 1

        if total > 0:
            print(f"    Match rate: {matches}/{total}")


def analyze_hd_derivation_hypothesis():
    """
    HD wallets use: child = (parent + HMAC(chaincode, parent_pub || index)) mod n

    If puzzle keys are derived this way, consecutive keys have structure.
    Let's see if we can detect it.
    """

    print("\n" + "="*70)
    print("HD DERIVATION HYPOTHESIS")
    print("="*70)

    # In HD derivation, key[i+1] - key[i] ≈ constant (the HMAC increment)
    # But this is mod n (secp256k1 order), so wrapping can occur

    puzzles = sorted(PUZZLE_KEYS.keys())

    # Compute second differences (acceleration)
    # If keys are linear progression, second diff ≈ 0

    print("\nSecond differences (key[i+1] - 2*key[i] + key[i-1]):")
    print("If keys are arithmetic progression, this should be ~0")

    second_diffs = []
    for i in range(1, len(puzzles) - 1):
        p_prev = puzzles[i-1]
        p_curr = puzzles[i]
        p_next = puzzles[i+1]

        k_prev = PUZZLE_KEYS[p_prev]
        k_curr = PUZZLE_KEYS[p_curr]
        k_next = PUZZLE_KEYS[p_next]

        second_diff = k_next - 2*k_curr + k_prev
        second_diffs.append(second_diff)

        if abs(second_diff) < 1000:
            print(f"  Puzzles {p_prev}-{p_curr}-{p_next}: second_diff = {second_diff} (SMALL!)")

    print(f"\nMean |second_diff|: {np.mean(np.abs(second_diffs)):.0f}")

    # Check if ratios follow a pattern (geometric progression?)
    print("\nRatio analysis (key[i+1] / key[i]):")
    ratios = []
    for i in range(len(puzzles) - 1):
        k1 = PUZZLE_KEYS[puzzles[i]]
        k2 = PUZZLE_KEYS[puzzles[i+1]]
        if k1 > 0:
            ratio = k2 / k1
            ratios.append(ratio)

    print(f"  Mean ratio: {np.mean(ratios):.3f}")
    print(f"  Std ratio: {np.std(ratios):.3f}")
    print(f"  If geometric with base 2: expect ratio ≈ 2.0")
    print(f"  If HD wallet: expect more variance due to HMAC")


def check_known_rng_pattern():
    """
    We know puzzles 1-13 were generated with Python random.randint().
    Check if 14+ follow a different pattern.
    """

    print("\n" + "="*70)
    print("RNG vs HD WALLET COMPARISON")
    print("="*70)

    # Puzzles 1-13: confirmed Python MT RNG
    # Puzzles 14+: unknown source

    print("\nKnown RNG segment (1-13):")
    rng_ratios = []
    for i in range(1, 13):
        if i in PUZZLE_KEYS and i+1 in PUZZLE_KEYS:
            ratio = PUZZLE_KEYS[i+1] / PUZZLE_KEYS[i]
            rng_ratios.append(ratio)
            print(f"  {i} → {i+1}: ratio = {ratio:.3f}")

    print(f"  Mean: {np.mean(rng_ratios):.3f}, Std: {np.std(rng_ratios):.3f}")

    print("\nUnknown segment (14+):")
    unk_ratios = []
    for i in range(14, 40):
        if i in PUZZLE_KEYS and i+1 in PUZZLE_KEYS:
            ratio = PUZZLE_KEYS[i+1] / PUZZLE_KEYS[i]
            unk_ratios.append(ratio)
            print(f"  {i} → {i+1}: ratio = {ratio:.3f}")

    print(f"  Mean: {np.mean(unk_ratios):.3f}, Std: {np.std(unk_ratios):.3f}")

    # Compare distributions
    print("\nComparison:")
    print(f"  RNG segment std: {np.std(rng_ratios):.3f}")
    print(f"  Unknown segment std: {np.std(unk_ratios):.3f}")

    if np.std(unk_ratios) < np.std(rng_ratios):
        print("  Unknown segment has LESS variance - might be more structured!")
    else:
        print("  Unknown segment has MORE variance - might be more random")


def main():
    """Run all analyses."""

    print("="*70)
    print("BITCOIN PUZZLE MASK ANALYSIS")
    print("="*70)
    print("\nCreator claim: 'consecutive keys from deterministic wallet'")
    print("              'masked with leading 000...0001 to set difficulty'")
    print("\nQuestion: Does the mask leak information about the full keys?")

    analyze_consecutive_differences()
    analyze_low_bits_patterns()
    analyze_bit_positions()
    analyze_mask_interaction()
    analyze_hd_derivation_hypothesis()
    check_known_rng_pattern()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key observations:
1. Ratios between consecutive keys cluster around 2.0 (expected from bit growth)
2. Need to check if variance differs between RNG and unknown segments
3. The mask operation (mod 2^N) might preserve HD derivation patterns
4. Cross-puzzle modular relationships could reveal structure
    """)


if __name__ == "__main__":
    main()
