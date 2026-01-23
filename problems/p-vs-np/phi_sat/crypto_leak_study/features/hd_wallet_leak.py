#!/usr/bin/env python3
"""
HD Wallet Masking Leak Analysis

Creator quote: "consecutive keys from a deterministic wallet
               (masked with leading 000...0001 to set difficulty)"

Key insight: HD wallet keys have STRUCTURE. The masking operation
(mod 2^N with high bit set) might PRESERVE that structure.

BIP32 HD derivation (non-hardened):
  child_key = (parent_key + HMAC-SHA512(chaincode, parent_pub || index)) mod n

For consecutive indices, the keys form a (pseudo-)arithmetic sequence.

If we mask to N bits:
  masked[i] = full_key[i] mod 2^(N-1) + 2^(N-1)

Question: Do consecutive HD keys, after masking, show patterns?
"""

import numpy as np
from typing import Dict, List, Tuple
import hashlib
import hmac
from dataclasses import dataclass

# Solved puzzle keys
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

# secp256k1 order
SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def simulate_hd_derivation(master_key: int, num_keys: int, delta: int = None) -> List[int]:
    """
    Simulate simplified HD derivation.

    Real HD: child = parent + HMAC(...) mod n
    We approximate: child[i] = master + i * delta mod n

    Where delta is derived from chaincode (simulated).
    """
    if delta is None:
        # Use a "typical" HD delta (from HMAC output)
        delta = int.from_bytes(hashlib.sha256(b"chaincode_simulation").digest()[:32], 'big')
        delta = delta % SECP256K1_ORDER

    keys = []
    for i in range(num_keys):
        key = (master_key + i * delta) % SECP256K1_ORDER
        keys.append(key)

    return keys


def mask_key(full_key: int, target_bits: int) -> int:
    """
    Apply the puzzle masking:
    - Take low (target_bits - 1) bits
    - Set the high bit to 1

    This ensures key is in range [2^(N-1), 2^N - 1]
    """
    low_bits = full_key % (1 << (target_bits - 1))
    masked = low_bits | (1 << (target_bits - 1))
    return masked


def analyze_mask_preservation():
    """
    Analyze if HD structure survives masking.

    If full keys are arithmetic sequence: k[i] = k[0] + i*d
    After masking: m[i] = (k[i] mod 2^(N-1)) + 2^(N-1)

    The differences m[i+1] - m[i] depend on whether k[i+1] wrapped.
    """

    print("="*70)
    print("HD STRUCTURE PRESERVATION UNDER MASKING")
    print("="*70)

    # Simulate HD wallet with known structure
    np.random.seed(42)
    master_key = int.from_bytes(np.random.bytes(32), 'big') % SECP256K1_ORDER

    # Test different deltas
    deltas_to_test = [
        0x1,  # Simple increment
        0x100,  # Larger step
        0x10000,  # Even larger
        int.from_bytes(hashlib.sha256(b"typical_hd_delta").digest()[:32], 'big') % SECP256K1_ORDER,  # "Realistic"
    ]

    for delta in deltas_to_test:
        print(f"\n--- Delta: {hex(delta)[:20]}... ---")

        # Generate 40 consecutive HD keys
        hd_keys = simulate_hd_derivation(master_key, 40, delta)

        # Mask each key to its corresponding puzzle bit length
        masked_keys = {}
        for puzzle_num in range(1, 41):
            masked_keys[puzzle_num] = mask_key(hd_keys[puzzle_num - 1], puzzle_num)

        # Analyze the ratios (like we did for real puzzles)
        ratios = []
        for i in range(1, 40):
            if masked_keys[i] > 0:
                ratio = masked_keys[i+1] / masked_keys[i]
                ratios.append(ratio)

        print(f"Ratio stats: mean={np.mean(ratios):.3f}, std={np.std(ratios):.3f}")

        # Compare low bits across puzzles
        # If HD structure survives, low bits should be related
        print(f"Low 8-bit patterns:")
        for i in [10, 20, 30, 40]:
            if i in masked_keys:
                low8 = masked_keys[i] & 0xFF
                print(f"  Puzzle {i}: low8 = {low8} ({bin(low8)})")


def reverse_engineer_structure():
    """
    Try to find what HD parameters would produce the observed puzzle keys.

    If keys are k[i] = master + i*delta mod n, masked to i bits,
    we can constrain delta from the relationships between puzzles.
    """

    print("\n" + "="*70)
    print("REVERSE ENGINEERING HD PARAMETERS")
    print("="*70)

    # Key observation: consecutive puzzle keys of SAME bit length
    # would reveal delta directly.
    # But puzzles have DIFFERENT bit lengths, so masking differs.

    # Strategy: look at low bits that are common across puzzles
    # If puzzle 40 and puzzle 20 share low 19 bits (the overlap),
    # those bits should be related by the HD structure.

    print("\nAnalyzing bit overlaps between puzzles...")

    for p1 in [20, 30, 40]:
        for p2 in [p1 + 1, p1 + 10]:
            if p2 > 40:
                continue

            # Low min(p1-1, p2-1) bits should come from same HD sequence
            overlap_bits = min(p1 - 1, p2 - 1)

            k1 = PUZZLE_KEYS[p1]
            k2 = PUZZLE_KEYS[p2]

            # Extract overlapping low bits
            mask = (1 << overlap_bits) - 1
            low1 = k1 & mask
            low2 = k2 & mask

            diff = (low2 - low1) % (1 << overlap_bits)

            print(f"Puzzles {p1} & {p2}:")
            print(f"  Overlap: {overlap_bits} bits")
            print(f"  Low bits: {hex(low1)}, {hex(low2)}")
            print(f"  Difference (mod 2^{overlap_bits}): {diff} = {hex(diff)}")


def test_delta_hypothesis():
    """
    Test specific delta values against observed puzzle keys.

    If we find a delta that, when applied to HD derivation + masking,
    produces keys matching the observed low bits, we've found structure.
    """

    print("\n" + "="*70)
    print("TESTING DELTA HYPOTHESES")
    print("="*70)

    # From puzzles 14+ (the "more structured" ones)
    # Try to find a delta that explains the low-bit relationships

    # Strategy: between adjacent puzzles, the delta should be visible
    # in the low bits (with wrapping)

    observed_diffs = []
    for i in range(14, 40):
        k1 = PUZZLE_KEYS[i]
        k2 = PUZZLE_KEYS[i + 1]

        # The difference in low i bits
        mask = (1 << (i - 1)) - 1
        low1 = k1 & mask
        low2 = k2 & mask

        # This difference, if HD structure holds, relates to delta
        diff = (low2 - low1) % (1 << (i - 1))
        observed_diffs.append(diff)

        print(f"  Puzzles {i}→{i+1}: low-bit diff = {diff} ({hex(diff)[:20]})")

    # Check if diffs have a pattern
    print(f"\nDifference statistics:")
    print(f"  Mean: {np.mean(observed_diffs):.0f}")
    print(f"  Std: {np.std(observed_diffs):.0f}")
    print(f"  Unique: {len(set(observed_diffs))}/{len(observed_diffs)}")

    # If diffs are related, they should share common factors
    from math import gcd
    from functools import reduce

    if len(observed_diffs) > 1:
        common_gcd = reduce(gcd, [d for d in observed_diffs if d > 0])
        print(f"  GCD of all diffs: {common_gcd}")


def analyze_rng_vs_hd():
    """
    Compare the RNG segment (1-13) vs unknown segment (14+)
    through the lens of HD wallet structure.
    """

    print("\n" + "="*70)
    print("RNG vs HD STRUCTURE COMPARISON")
    print("="*70)

    # RNG segment: puzzles 1-13 (confirmed Python MT)
    # Unknown segment: puzzles 14+ (potentially HD wallet)

    # If 14+ is HD, the low bits should show more consistency

    print("\nBit consistency analysis:")
    print("(How often does bit position B have same value across adjacent puzzles)")

    for segment_name, start, end in [("RNG (1-13)", 1, 13), ("Unknown (14-40)", 14, 40)]:
        print(f"\n{segment_name}:")

        consistencies = []
        for bit_pos in range(1, 8):  # Check first 8 bit positions
            matches = 0
            total = 0

            for p in range(start, end):
                if p + 1 <= end:
                    k1 = PUZZLE_KEYS[p]
                    k2 = PUZZLE_KEYS[p + 1]

                    bit1 = (k1 >> bit_pos) & 1
                    bit2 = (k2 >> bit_pos) & 1

                    if bit1 == bit2:
                        matches += 1
                    total += 1

            if total > 0:
                consistency = matches / total
                consistencies.append(consistency)
                print(f"  Bit {bit_pos}: {consistency:.2%} consistent ({matches}/{total})")

        print(f"  Mean consistency: {np.mean(consistencies):.2%}")
        print(f"  (Random expected: 50%)")


def find_hidden_sequence():
    """
    Look for arithmetic or geometric sequences hidden in the keys.
    """

    print("\n" + "="*70)
    print("HIDDEN SEQUENCE SEARCH")
    print("="*70)

    # For HD wallet, consecutive full keys differ by delta
    # After masking, this creates a specific pattern

    # Let's look at what full keys COULD have produced these masked values
    # For puzzle N with masked key M:
    #   full_key mod 2^(N-1) = M - 2^(N-1) = M & (2^(N-1) - 1)

    print("\nRecovering possible full key low bits:")

    low_bits = {}
    for p in sorted(PUZZLE_KEYS.keys()):
        key = PUZZLE_KEYS[p]
        low = key & ((1 << (p - 1)) - 1)  # Remove the forced high bit
        low_bits[p] = low

        if p >= 14 and p <= 20:  # Focus on unknown segment start
            print(f"  Puzzle {p}: masked={hex(key)}, low {p-1} bits = {hex(low)}")

    # Check if low bits form an arithmetic sequence
    print("\nArithmetic sequence test (puzzles 14-20):")
    print("If HD: low_bits[i+1] - low_bits[i] ≈ delta mod 2^(i-1)")

    for i in range(14, 20):
        diff = low_bits[i + 1] - low_bits[i]
        mod = 1 << (i - 1)
        diff_mod = diff % mod

        print(f"  {i}→{i+1}: diff={diff}, mod 2^{i-1}={diff_mod}")


if __name__ == "__main__":
    analyze_mask_preservation()
    reverse_engineer_structure()
    test_delta_hypothesis()
    analyze_rng_vs_hd()
    find_hidden_sequence()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key findings from HD wallet leak analysis:

1. If puzzle creator used HD wallet, consecutive keys have structure:
   full_key[i] = master + i * delta (mod secp256k1_order)

2. Masking to N bits PARTIALLY preserves this:
   - Low N-1 bits come from full key
   - These should show HD derivation patterns

3. Cross-puzzle analysis can reveal delta:
   - Compare low bits across puzzles
   - Look for consistent differences

4. RNG segment (1-13) vs Unknown (14+):
   - Already found: Unknown has LESS variance
   - Need to test if bit consistency differs

5. If we can recover delta, we can PREDICT higher puzzles:
   - full_key[N] = master + N * delta
   - masked[N] = full_key[N] mod 2^(N-1) + 2^(N-1)
""")
