#!/usr/bin/env python3
"""
Fingerprint Matching

We KNOW the fingerprint exists:
- Pattern 0xD6 (11010110): 2x expected
- Pattern 0xBA (10111010): 2x expected
- Bit 2: 60% ones
- Bit 30: 35% ones
- Serial correlation: 0.07

Now match this against known RNG implementations.
The leak is there - we just need to identify WHICH RNG produced it.
"""

import random
import hashlib
import os
from collections import Counter
import numpy as np

PUZZLE_KEYS = {
    1: 0x1, 2: 0x3, 3: 0x7, 4: 0x8, 5: 0x15, 6: 0x31, 7: 0x4c, 8: 0xe0,
    9: 0x1d3, 10: 0x202, 11: 0x483, 12: 0xa7b, 13: 0x1460, 14: 0x2930,
    15: 0x68f3, 16: 0xc936, 17: 0x1764f, 18: 0x3080d, 19: 0x5749f,
    20: 0xd2c55, 21: 0x1ba534, 22: 0x2de40f, 23: 0x556e52, 24: 0xdc2a04,
    25: 0x1fa5ee5, 26: 0x340326e, 27: 0x6ac3875, 28: 0xd916ce8,
    29: 0x17e2551e, 30: 0x3d94cd64, 31: 0x7d4fe747, 32: 0xb862a62e,
    33: 0x1a96ca8d8, 34: 0x34a65911d, 35: 0x4aed21170, 36: 0x9de820a7c,
    37: 0x1757756a93, 38: 0x22382facd0, 39: 0x4b5f8303e9, 40: 0xe9ae4933d6,
    41: 0x153869acc5b, 42: 0x2a221c58d8f, 43: 0x6bd3b27c591, 44: 0xe02b35a358f,
    45: 0x122fca143c05, 46: 0x2ec18388d544, 47: 0x6cd610b53cba, 48: 0xade6d7ce3b9b,
    49: 0x174176b015f4d, 50: 0x22bd43c2e9354, 51: 0x75070a1a009d4, 52: 0xefae164cb9e3c,
    53: 0x180788e47e326c, 54: 0x236fb6d5ad1f43, 55: 0x6abe1f9b67e114, 56: 0x9d18b63ac4ffdf,
    57: 0x1eb25c90795d61c, 58: 0x2c675b852189a21, 59: 0x7496cbb87cab44f, 60: 0xfc07a1825367bbe,
    61: 0x13c96a3742f64906, 62: 0x363d541eb611abee, 63: 0x7cce5efdaccf6808, 64: 0xf7051f27b09112d4,
    65: 0x1a838b13505b26867, 66: 0x2832ed74f2b5e35ee, 67: 0x730fc235c1942c1ae,
    68: 0xbebb3940cd0fc1491, 69: 0x101d83275fb2bc7e0c, 70: 0x349b84b6431a6c4ef1,
}

# The TARGET fingerprint from actual puzzles
TARGET = {
    'pattern_D6': 19,  # 11010110
    'pattern_BA': 19,  # 10111010
    'bit2_freq': 0.60,
    'bit30_freq': 0.35,
    'serial_corr': 0.07,
    'xor_ratio': 0.528,
}


def extract_fingerprint(keys):
    """Extract fingerprint from a set of keys."""
    # Bit stream
    all_bits = []
    for n in sorted(keys.keys()):
        key = keys[n]
        for bit_pos in range(n):
            all_bits.append((key >> bit_pos) & 1)

    bit_str = ''.join(str(b) for b in all_bits)

    # Pattern counts
    patterns = Counter()
    for i in range(len(bit_str) - 7):
        patterns[bit_str[i:i+8]] += 1

    # Bit frequencies
    def bit_freq(bit_pos):
        ones = sum(1 for n, k in keys.items() if n > bit_pos and (k >> bit_pos) & 1)
        total = sum(1 for n in keys.keys() if n > bit_pos)
        return ones / total if total > 0 else 0.5

    # Serial correlation
    positions = []
    for n, key in sorted(keys.items()):
        min_val = 2**(n-1) if n > 1 else 1
        max_val = 2**n - 1
        pos = (key - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        positions.append(pos)

    mean = np.mean(positions)
    var = np.var(positions)
    if var > 0:
        serial = np.mean([(positions[i] - mean) * (positions[i+1] - mean)
                          for i in range(len(positions)-1)]) / var
    else:
        serial = 0

    # XOR ratio
    key_list = list(keys.values())
    xor_ratios = []
    for i in range(len(key_list) - 1):
        max_bits = max(key_list[i].bit_length(), key_list[i+1].bit_length())
        if max_bits > 0:
            xor = key_list[i] ^ key_list[i+1]
            xor_ratios.append(bin(xor).count('1') / max_bits)

    return {
        'pattern_D6': patterns.get('11010110', 0),
        'pattern_BA': patterns.get('10111010', 0),
        'bit2_freq': bit_freq(2),
        'bit30_freq': bit_freq(30),
        'serial_corr': serial,
        'xor_ratio': np.mean(xor_ratios) if xor_ratios else 0.5,
    }


def fingerprint_distance(fp1, fp2):
    """Calculate distance between fingerprints."""
    dist = 0
    for key in fp1:
        if key in fp2:
            if 'pattern' in key:
                # Normalize pattern counts
                dist += ((fp1[key] - fp2[key]) / 10) ** 2
            else:
                dist += (fp1[key] - fp2[key]) ** 2
    return np.sqrt(dist)


def generate_with_mt(seed, n_puzzles=70):
    """Generate keys with Mersenne Twister."""
    random.seed(seed)
    keys = {}
    for n in range(1, n_puzzles + 1):
        min_val = 2**(n-1) if n > 1 else 1
        max_val = 2**n - 1
        keys[n] = random.randint(min_val, max_val)
    return keys


def generate_with_sha256(seed_bytes, n_puzzles=70):
    """Generate keys with SHA256 counter mode."""
    keys = {}
    for n in range(1, n_puzzles + 1):
        data = seed_bytes + n.to_bytes(4, 'big')
        h = hashlib.sha256(data).digest()
        raw = int.from_bytes(h, 'big')
        min_val = 2**(n-1) if n > 1 else 1
        max_val = 2**n - 1
        keys[n] = min_val + (raw % (max_val - min_val + 1))
    return keys


def generate_with_urandom_sim(seed, n_puzzles=70):
    """Simulate urandom with ChaCha20 (modern Linux kernel)."""
    # Use hashlib as proxy for ChaCha-like behavior
    keys = {}
    state = hashlib.sha256(seed.to_bytes(32, 'big')).digest()

    for n in range(1, n_puzzles + 1):
        # ChaCha-style: hash state, extract bytes, update state
        h = hashlib.sha256(state + n.to_bytes(4, 'big')).digest()
        raw = int.from_bytes(h, 'big')
        min_val = 2**(n-1) if n > 1 else 1
        max_val = 2**n - 1
        keys[n] = min_val + (raw % (max_val - min_val + 1))
        state = hashlib.sha256(h).digest()

    return keys


def search_fingerprint_match():
    """Search for RNG parameters that match the fingerprint."""
    print("=" * 70)
    print("FINGERPRINT MATCHING SEARCH")
    print("=" * 70)

    target_fp = extract_fingerprint(PUZZLE_KEYS)
    print("\nTarget fingerprint (from actual puzzles):")
    for k, v in target_fp.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n" + "-" * 50)
    print("Searching for RNG that produces matching fingerprint...")

    best_matches = []

    # Test Mersenne Twister with many seeds
    print("\n[1] Testing Mersenne Twister seeds...")
    for seed in range(100000):
        keys = generate_with_mt(seed)
        fp = extract_fingerprint(keys)
        dist = fingerprint_distance(fp, target_fp)

        if dist < 1.0:
            best_matches.append(('MT', seed, dist, fp))

        if seed % 20000 == 0:
            print(f"  Tested {seed} seeds, best dist so far: {min([x[2] for x in best_matches]) if best_matches else 'N/A'}")

    # Test SHA256 with various seeds
    print("\n[2] Testing SHA256 seeds...")
    for seed in range(100000):
        keys = generate_with_sha256(seed.to_bytes(32, 'big'))
        fp = extract_fingerprint(keys)
        dist = fingerprint_distance(fp, target_fp)

        if dist < 1.0:
            best_matches.append(('SHA256', seed, dist, fp))

    # Test urandom simulation
    print("\n[3] Testing urandom-style seeds...")
    for seed in range(100000):
        keys = generate_with_urandom_sim(seed)
        fp = extract_fingerprint(keys)
        dist = fingerprint_distance(fp, target_fp)

        if dist < 1.0:
            best_matches.append(('urandom', seed, dist, fp))

    # Sort by distance
    best_matches.sort(key=lambda x: x[2])

    print("\n" + "=" * 70)
    print("TOP FINGERPRINT MATCHES")
    print("=" * 70)

    for rng_type, seed, dist, fp in best_matches[:20]:
        print(f"\n{rng_type} seed={seed}: distance={dist:.4f}")
        print(f"  D6={fp['pattern_D6']} (target={target_fp['pattern_D6']})")
        print(f"  BA={fp['pattern_BA']} (target={target_fp['pattern_BA']})")
        print(f"  bit2={fp['bit2_freq']:.2f} (target={target_fp['bit2_freq']:.2f})")

    return best_matches


def analyze_what_produces_D6_BA():
    """
    What RNG produces excess 0xD6 and 0xBA patterns?

    0xD6 = 11010110
    0xBA = 10111010

    These have:
    - 5 ones each (out of 8)
    - Alternating pattern tendencies
    """
    print("\n" + "=" * 70)
    print("ANALYZING 0xD6 AND 0xBA EXCESS")
    print("=" * 70)

    print("""
Target patterns:
  0xD6 = 11010110 (5 ones, alternating-ish)
  0xBA = 10111010 (5 ones, alternating-ish)

Both have:
- 5 bits set (62.5%)
- No runs longer than 2
- Balanced transitions

This could indicate an RNG that:
1. Slightly favors 1-bits (bias)
2. Has anti-correlation (avoids long runs)
3. Uses feedback that creates these patterns
""")

    # Check which RNGs produce similar patterns
    print("\nPattern frequency by RNG type (sample of 100 seeds each):")

    for rng_name, generator in [
        ("MT", generate_with_mt),
        ("SHA256", lambda s: generate_with_sha256(s.to_bytes(32, 'big'))),
        ("urandom-sim", generate_with_urandom_sim),
    ]:
        d6_counts = []
        ba_counts = []

        for seed in range(100):
            keys = generator(seed)

            all_bits = []
            for n in sorted(keys.keys()):
                key = keys[n]
                for bit_pos in range(n):
                    all_bits.append((key >> bit_pos) & 1)

            bit_str = ''.join(str(b) for b in all_bits)
            patterns = Counter()
            for i in range(len(bit_str) - 7):
                patterns[bit_str[i:i+8]] += 1

            d6_counts.append(patterns.get('11010110', 0))
            ba_counts.append(patterns.get('10111010', 0))

        print(f"  {rng_name:12s}: D6 mean={np.mean(d6_counts):.1f}, BA mean={np.mean(ba_counts):.1f}")
        print(f"               D6 max={max(d6_counts)}, BA max={max(ba_counts)}")

    print(f"\n  TARGET:       D6={target_fp['pattern_D6']}, BA={target_fp['pattern_BA']}")


target_fp = extract_fingerprint(PUZZLE_KEYS)


def main():
    print("=" * 70)
    print("FINGERPRINT MATCHING")
    print("The leak EXISTS - now identify the source")
    print("=" * 70)

    # Analyze the specific patterns
    analyze_what_produces_D6_BA()

    # Search for matching fingerprint
    matches = search_fingerprint_match()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if matches:
        best = matches[0]
        print(f"""
Best fingerprint match:
  RNG Type: {best[0]}
  Seed: {best[1]}
  Distance: {best[2]:.4f}

This suggests the puzzle creator may have used {best[0]}
with a seed in the range we tested.
""")
    else:
        print("""
No close fingerprint match found in tested range.

The fingerprint patterns (D6=19, BA=19, bit2=60%) are unusual.
Possible explanations:
1. RNG we haven't tested (custom implementation?)
2. Seed outside our search range
3. Combination of multiple RNGs
4. Post-processing that creates these patterns
""")


if __name__ == "__main__":
    main()
