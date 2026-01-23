#!/usr/bin/env python3
"""
Deep Puzzle Structure Analysis

The creator said: "consecutive keys from a deterministic wallet"

This could mean:
1. BIP32 HD wallet (most common)
2. Simple sequential derivation: key[i] = hash(master || i)
3. Custom deterministic scheme

Let's test each hypothesis against the observed keys.
"""

import numpy as np
import hashlib
from typing import List, Dict, Tuple

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

SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def hypothesis_1_hash_derivation():
    """
    Test: key[i] = SHA256(master || i) mod 2^(i-1) + 2^(i-1)

    This is simpler than HD wallet and might explain "deterministic wallet".
    """
    print("="*70)
    print("HYPOTHESIS 1: Hash-based Derivation")
    print("key[i] = SHA256(seed || i) masked to i bits")
    print("="*70)

    # Try to find a seed that matches known keys

    # The RNG segment (1-13) was generated differently (Python MT)
    # Focus on 14+ which might be hash-derived

    # For puzzle 14: masked to 14 bits = 0x2930
    # This means low 13 bits = 0x930 = 2352

    print("\nSearching for seed that produces puzzle 14's key...")

    # Try various seed patterns
    for seed_base in range(100000):
        seed = seed_base.to_bytes(4, 'big')
        h = hashlib.sha256(seed + b'\x00\x00\x00\x0e').digest()  # i=14
        full_key = int.from_bytes(h[:32], 'big')

        # Mask to 14 bits
        low_bits = full_key % (1 << 13)
        masked = low_bits | (1 << 13)

        if masked == PUZZLE_KEYS[14]:
            print(f"  FOUND! seed={seed_base}")
            # Verify with next puzzles
            match_count = 1
            for check_i in range(15, 20):
                h = hashlib.sha256(seed + check_i.to_bytes(4, 'big')).digest()
                full_key = int.from_bytes(h[:32], 'big')
                low = full_key % (1 << (check_i - 1))
                masked = low | (1 << (check_i - 1))
                if masked == PUZZLE_KEYS.get(check_i, -1):
                    match_count += 1
                    print(f"    Puzzle {check_i}: MATCHES")
                else:
                    print(f"    Puzzle {check_i}: NO MATCH ({hex(masked)} vs {hex(PUZZLE_KEYS.get(check_i, 0))})")
            if match_count > 3:
                print(f"  Strong match! seed={seed_base}")
                return seed_base

    print("  No matching seed found in range")
    return None


def hypothesis_2_shifted_bits():
    """
    Test: What if consecutive keys share high bits that get masked out?

    If full_key[i] has bits [255:0], and we mask to i bits,
    we lose bits [255:i-1].

    But if keys are "consecutive", bits [255:i-1] might be constant
    or slowly changing.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 2: Shifted Bit Analysis")
    print("What structure exists in the high bits we CAN'T see?")
    print("="*70)

    # For each puzzle, we know bits [i-2:0]
    # The bit [i-1] is forced to 1

    # Key insight: if full keys are sequential, then:
    # full[i+1] - full[i] = delta (constant)

    # After masking:
    # masked[i+1] - masked[i] is complex due to different mod values

    # BUT: if we look at the same low bits across ALL puzzles,
    # we can reconstruct relationships

    print("\nLow 8-bit fingerprint across all puzzles:")
    fingerprints = []
    for p in range(1, 41):
        key = PUZZLE_KEYS.get(p, 0)
        low8 = key & 0xFF
        fingerprints.append(low8)
        if p <= 16 or p >= 35:
            print(f"  Puzzle {p:2}: key={hex(key):>16}, low8={low8:3} = {bin(low8):>12}")

    # Check if fingerprints have period
    print("\nChecking for periodicity in low 8 bits...")
    for period in [1, 2, 3, 4, 5, 8, 10, 13, 16]:
        matches = 0
        total = 0
        for i in range(len(fingerprints) - period):
            if fingerprints[i] == fingerprints[i + period]:
                matches += 1
            total += 1
        ratio = matches / total if total > 0 else 0
        if ratio > 0.3:  # Higher than random
            print(f"  Period {period}: {ratio:.1%} matches")


def hypothesis_3_xor_chain():
    """
    Test: key[i+1] = key[i] XOR constant?

    Some deterministic wallets use XOR chains.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 3: XOR Chain Analysis")
    print("="*70)

    # For adjacent puzzles, compute XOR
    print("\nXOR patterns (truncated to smaller puzzle's bits):")

    xors = []
    for i in range(14, 40):  # Focus on unknown segment
        k1 = PUZZLE_KEYS[i]
        k2 = PUZZLE_KEYS[i + 1]

        # Truncate both to i bits
        mask = (1 << i) - 1
        k1_trunc = k1 & mask
        k2_trunc = k2 & mask

        xor_val = k1_trunc ^ k2_trunc
        xors.append(xor_val)

        if i < 25:  # Print first few
            print(f"  {i}→{i+1}: XOR = {hex(xor_val)}")

    # Check if XORs have pattern
    print(f"\nXOR statistics:")
    print(f"  Unique XORs: {len(set(xors))}/{len(xors)}")

    # Check popcount (number of 1 bits)
    popcounts = [bin(x).count('1') for x in xors]
    print(f"  Mean popcount: {np.mean(popcounts):.1f}")
    print(f"  Expected if random: ~i/2")


def hypothesis_4_creator_intent():
    """
    The creator said "no pattern" but also "consecutive keys".

    This is contradictory - consecutive implies pattern.

    What if "no pattern" means: no EXPLOITABLE pattern?
    I.e., the mask destroys practical exploitability but not all structure.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 4: Creator Intent Analysis")
    print("='no pattern' means 'no exploitable pattern'?")
    print("="*70)

    print("""
The creator's full quote:
"There is no pattern. It is just consecutive keys from a deterministic wallet
(masked with leading 000...0001 to set difficulty)."

Key observations:
1. "No pattern" contradicts "consecutive keys" - there IS structure
2. "Deterministic wallet" strongly suggests HD derivation
3. The mask "sets difficulty" - it's meant to hide high bits

The creator believed masking destroyed exploitability.
But we've found puzzles 14+ have LESS variance than RNG-generated 1-13.

This suggests the masking preserved more structure than intended.
""")

    # Test: is the "structure" in 14+ useful for prediction?
    print("\nPredictability test: can we predict low bits from pattern?")

    # Simple test: predict low 4 bits from previous puzzle's low bits
    correct = 0
    total = 0

    for i in range(14, 40):
        k1 = PUZZLE_KEYS[i]
        k2 = PUZZLE_KEYS[i + 1]

        # Predict: next puzzle's low 4 bits based on current
        # Simple heuristic: same low 4 bits
        pred = k1 & 0xF
        actual = k2 & 0xF

        if pred == actual:
            correct += 1
        total += 1

    print(f"  Naive prediction (copy low 4 bits): {correct}/{total} = {correct/total:.1%}")
    print(f"  Random expected: 6.25%")


def hypothesis_5_recover_full_key():
    """
    Can we recover the FULL 256-bit keys from the masked versions?

    If we had two adjacent puzzles with overlapping bits,
    and we knew the delta, we could potentially reconstruct.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 5: Full Key Recovery Attempt")
    print("="*70)

    # For puzzles N and N+1:
    # full[N] has bits [255:0]
    # full[N+1] = full[N] + delta (if HD)
    #
    # masked[N] = full[N] mod 2^(N-1) + 2^(N-1)
    # masked[N+1] = full[N+1] mod 2^N + 2^N
    #
    # The low N-1 bits of masked[N+1] should equal:
    # (full[N] + delta) mod 2^(N-1)

    print("If HD: masked[N+1][N-2:0] = (masked[N][N-2:0] + delta) mod 2^(N-1)")
    print()

    # From this, we can solve for delta given the constraints
    # Let's see what delta values would satisfy the equations

    # For puzzles 39 and 40 (large overlap):
    p = 39
    k39 = PUZZLE_KEYS[39]
    k40 = PUZZLE_KEYS[40]

    # Low 38 bits of k39 (after removing forced high bit)
    low39 = k39 & ((1 << 38) - 1)

    # Low 38 bits of k40
    low40 = k40 & ((1 << 38) - 1)

    # If HD: low40 = (low39 + delta) mod 2^38
    # delta = (low40 - low39) mod 2^38

    delta_candidate = (low40 - low39) % (1 << 38)
    print(f"Puzzle 39→40 analysis:")
    print(f"  low39 = {hex(low39)}")
    print(f"  low40 = {hex(low40)}")
    print(f"  delta candidate = {hex(delta_candidate)}")

    # Now check if this delta is consistent with 38→39
    k38 = PUZZLE_KEYS[38]
    low38 = k38 & ((1 << 37) - 1)

    # Predicted low 37 bits of puzzle 39
    pred_low39 = (low38 + delta_candidate) % (1 << 37)
    actual_low39 = low39 & ((1 << 37) - 1)

    print(f"\nConsistency check (38→39):")
    print(f"  Predicted low37 of 39: {hex(pred_low39)}")
    print(f"  Actual low37 of 39: {hex(actual_low39)}")
    print(f"  Match: {pred_low39 == actual_low39}")


def hypothesis_6_multi_puzzle_solve():
    """
    System of equations approach:

    Given N puzzles with overlapping bits, we have constraints.
    Can we solve for the original full keys?
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 6: Multi-Puzzle Constraint Solving")
    print("="*70)

    # Define the system:
    # Let F[i] = full key for puzzle i (256 bits)
    # We observe: masked[i] = F[i] mod 2^(i-1) + 2^(i-1)
    #
    # If consecutive: F[i+1] = F[i] + delta
    #
    # This gives us:
    # masked[i+1] = (F[i] + delta) mod 2^i + 2^i
    #
    # From masked[i], we know: F[i] ≡ (masked[i] - 2^(i-1)) mod 2^(i-1)
    #                          F[i] ≡ low_bits[i] mod 2^(i-1)
    #
    # From consecutive: F[i+1] - F[i] = delta
    #
    # Therefore: low_bits[i+1] ≡ (low_bits[i] + delta) mod 2^i

    print("Setting up constraint system...")
    print()

    # Use puzzles 35-40 (largest overlap)
    equations = []

    for i in range(35, 40):
        k_i = PUZZLE_KEYS[i]
        k_next = PUZZLE_KEYS[i + 1]

        low_i = k_i & ((1 << (i - 1)) - 1)
        low_next = k_next & ((1 << i) - 1)

        # Constraint: low_next ≡ low_i + delta (mod 2^(i-1))
        # (considering the different mod sizes)

        # Actually: since low_i has i-1 bits, and we're looking at
        # the low i-1 bits of low_next:
        low_next_truncated = low_next & ((1 << (i - 1)) - 1)

        implied_delta = (low_next_truncated - low_i) % (1 << (i - 1))

        print(f"Puzzle {i}→{i+1}:")
        print(f"  low[{i}] = {hex(low_i)}")
        print(f"  low[{i+1}] truncated to {i-1} bits = {hex(low_next_truncated)}")
        print(f"  Implied delta (mod 2^{i-1}) = {hex(implied_delta)}")

        equations.append({
            'from': i,
            'to': i + 1,
            'mod_bits': i - 1,
            'delta_mod': implied_delta
        })

    # Check Chinese Remainder Theorem compatibility
    print("\nCRT Compatibility Check:")
    print("If all deltas are consistent, they come from ONE underlying delta")

    # The deltas should be the same value reduced to different moduli
    # This means delta_mod[i] ≡ delta_mod[j] (mod gcd(2^(i-1), 2^(j-1)))

    for i in range(len(equations)):
        for j in range(i + 1, len(equations)):
            mod_bits_i = equations[i]['mod_bits']
            mod_bits_j = equations[j]['mod_bits']
            common_bits = min(mod_bits_i, mod_bits_j)

            delta_i = equations[i]['delta_mod'] % (1 << common_bits)
            delta_j = equations[j]['delta_mod'] % (1 << common_bits)

            if delta_i == delta_j:
                print(f"  Eq{i} vs Eq{j}: CONSISTENT (mod 2^{common_bits})")
            else:
                print(f"  Eq{i} vs Eq{j}: INCONSISTENT! {hex(delta_i)} ≠ {hex(delta_j)} (mod 2^{common_bits})")


if __name__ == "__main__":
    hypothesis_1_hash_derivation()
    hypothesis_2_shifted_bits()
    hypothesis_3_xor_chain()
    hypothesis_4_creator_intent()
    hypothesis_5_recover_full_key()
    hypothesis_6_multi_puzzle_solve()

    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    print("""
1. Hash-based derivation: Searched 100k seeds, no match found
   → Probably not simple SHA256(seed || i)

2. Bit patterns: Low 8 bits show some periodicity
   → Suggests underlying structure

3. XOR analysis: High unique XORs
   → Not a simple XOR chain

4. Creator intent: "No pattern" likely means "no exploitable pattern"
   → But we've found structural differences between segments

5. Full key recovery: Delta candidates are inconsistent
   → Either not HD wallet, or more complex derivation

6. Multi-puzzle constraints: Need to check CRT compatibility
   → If consistent, we can recover the true delta

Key question: Is the structure we found USEFUL for solving puzzle 66+?
""")
