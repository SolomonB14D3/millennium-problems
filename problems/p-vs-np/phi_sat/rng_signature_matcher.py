#!/usr/bin/env python3
"""
RNG Signature Matcher

Find which RNG produces patterns matching the Bitcoin puzzle fingerprint.

2015 SOTA RNGs to test:
1. OpenSSL RAND_bytes (via os.urandom on most systems)
2. Mersenne Twister (Python random)
3. PCG (newer, but was gaining traction)
4. Xorshift variants
5. ChaCha20 (used in /dev/urandom)
6. AES-CTR DRBG
7. SHA-based DRBG

Custom possibilities:
- Hash of incrementing counter
- Hash of timestamp
- Mixed/combined RNGs
- Truncated/modified output
"""

import os
import random
import hashlib
import struct
from collections import Counter
from typing import List, Dict, Tuple, Callable
import math

# Our target fingerprint from the Bitcoin puzzles
TARGET_FINGERPRINT = {
    'pattern_0xD6': 19,  # 11010110 appears 19 times
    'pattern_0xBA': 19,  # 10111010 appears 19 times
    'xor_ratio': 0.528,
    'serial_corr': 0.07,
    'position_mean': 0.5263,
    'bit2_freq': 0.60,
    'bit30_freq': 0.35,
}

# Puzzle keys for reference
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
    65: 0x1a838b13505b26867, 66: 0x2832ed74f2b5e35ee, 67: 0x730fc235c1942c1ae, 68: 0xbebb3940cd0fc1491,
    69: 0x101d83275fb2bc7e0c, 70: 0x349b84b6431a6c4ef1,
}


# =============================================================================
# RNG IMPLEMENTATIONS
# =============================================================================

class RNGBase:
    """Base class for RNG implementations."""
    def __init__(self, seed=None):
        self.seed = seed
        self.reset()

    def reset(self):
        """Reset to initial state."""
        pass

    def random_bytes(self, n: int) -> bytes:
        """Generate n random bytes."""
        raise NotImplementedError

    def random_in_range(self, min_val: int, max_val: int) -> int:
        """Generate random integer in [min_val, max_val]."""
        range_size = max_val - min_val + 1
        # How many bytes needed?
        num_bytes = (range_size.bit_length() + 7) // 8 + 1

        while True:
            raw = int.from_bytes(self.random_bytes(num_bytes), 'big')
            val = min_val + (raw % range_size)
            # Rejection sampling to avoid modulo bias
            if raw < (256 ** num_bytes) - ((256 ** num_bytes) % range_size):
                return val


class MersenneTwister(RNGBase):
    """Python's standard Mersenne Twister."""
    def reset(self):
        self.rng = random.Random(self.seed)

    def random_bytes(self, n: int) -> bytes:
        return bytes(self.rng.randint(0, 255) for _ in range(n))


class Xorshift128(RNGBase):
    """Xorshift128 - popular in 2010s for speed."""
    def reset(self):
        if self.seed is None:
            self.state = [0x12345678, 0x9ABCDEF0, 0xFEDCBA98, 0x76543210]
        else:
            # Seed the state
            self.state = [
                (self.seed) & 0xFFFFFFFF,
                (self.seed >> 32) & 0xFFFFFFFF,
                (self.seed * 0x5DEECE66D) & 0xFFFFFFFF,
                (self.seed * 0xB) & 0xFFFFFFFF,
            ]
            # Make sure no zero state
            if all(s == 0 for s in self.state):
                self.state[0] = 1

    def _next(self) -> int:
        t = self.state[3]
        s = self.state[0]
        self.state[3] = self.state[2]
        self.state[2] = self.state[1]
        self.state[1] = s
        t ^= (t << 11) & 0xFFFFFFFF
        t ^= (t >> 8) & 0xFFFFFFFF
        self.state[0] = t ^ s ^ ((s >> 19) & 0xFFFFFFFF)
        return self.state[0]

    def random_bytes(self, n: int) -> bytes:
        result = []
        for _ in range((n + 3) // 4):
            val = self._next()
            result.extend([
                val & 0xFF,
                (val >> 8) & 0xFF,
                (val >> 16) & 0xFF,
                (val >> 24) & 0xFF,
            ])
        return bytes(result[:n])


class SHA256Counter(RNGBase):
    """SHA256(seed || counter) - common DRBG pattern."""
    def reset(self):
        self.counter = 0
        if self.seed is None:
            self.seed_bytes = b'default_seed'
        elif isinstance(self.seed, int):
            self.seed_bytes = self.seed.to_bytes(32, 'big')
        else:
            self.seed_bytes = self.seed

    def random_bytes(self, n: int) -> bytes:
        result = b''
        while len(result) < n:
            data = self.seed_bytes + self.counter.to_bytes(8, 'big')
            result += hashlib.sha256(data).digest()
            self.counter += 1
        return result[:n]


class SHA256Timestamp(RNGBase):
    """SHA256(base || puzzle_number) - if creator used puzzle# as nonce."""
    def reset(self):
        self.counter = 0
        if self.seed is None:
            self.seed_bytes = b'bitcoin_puzzle'
        elif isinstance(self.seed, int):
            self.seed_bytes = self.seed.to_bytes(32, 'big')
        else:
            self.seed_bytes = self.seed

    def random_bytes(self, n: int) -> bytes:
        # This simulates generating based on puzzle number
        data = self.seed_bytes + struct.pack('>Q', self.counter)
        self.counter += 1
        return hashlib.sha256(data).digest()[:n]


class LCG(RNGBase):
    """Linear Congruential Generator - old school but still used."""
    # Common LCG parameters
    PARAMS = {
        'glibc': (1103515245, 12345, 2**31),
        'msvc': (214013, 2531011, 2**32),
        'java': (25214903917, 11, 2**48),
        'numerical_recipes': (1664525, 1013904223, 2**32),
    }

    def __init__(self, seed=None, variant='glibc'):
        self.variant = variant
        super().__init__(seed)

    def reset(self):
        self.a, self.c, self.m = self.PARAMS[self.variant]
        self.state = self.seed if self.seed else 1

    def _next(self) -> int:
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def random_bytes(self, n: int) -> bytes:
        result = []
        bits_per_call = self.m.bit_length()
        bytes_per_call = bits_per_call // 8

        while len(result) < n:
            val = self._next()
            for i in range(bytes_per_call):
                result.append((val >> (i * 8)) & 0xFF)

        return bytes(result[:n])


class OpenSSLSimulator(RNGBase):
    """
    Simulate OpenSSL's RAND_bytes behavior.

    OpenSSL 1.0.x used a FIPS 186-2 based PRNG with SHA-1.
    This is an approximation.
    """
    def reset(self):
        self.counter = 0
        if self.seed is None:
            # Simulate low-entropy seeding (common vulnerability)
            self.state = hashlib.sha1(b'openssl_default').digest()
        else:
            self.state = hashlib.sha1(str(self.seed).encode()).digest()

    def random_bytes(self, n: int) -> bytes:
        result = b''
        while len(result) < n:
            # Mix counter into state
            mix = self.state + self.counter.to_bytes(8, 'big')
            self.state = hashlib.sha1(mix).digest()
            result += self.state
            self.counter += 1
        return result[:n]


class CustomBitcoinRNG(RNGBase):
    """
    Simulate potential custom Bitcoin-era RNG.

    Bitcoin Core's GetRandBytes() mixed:
    - OpenSSL RAND_bytes
    - rdrand if available
    - Various system entropy sources
    """
    def reset(self):
        self.counter = 0
        seed_material = str(self.seed).encode() if self.seed else b'bitcoin'
        self.state = hashlib.sha256(seed_material).digest()

    def random_bytes(self, n: int) -> bytes:
        result = b''
        while len(result) < n:
            # Double SHA256 (Bitcoin's favorite)
            temp = hashlib.sha256(self.state + self.counter.to_bytes(8, 'little')).digest()
            self.state = hashlib.sha256(temp).digest()
            result += self.state
            self.counter += 1
        return result[:n]


# =============================================================================
# SIGNATURE EXTRACTION
# =============================================================================

def extract_signature(keys: Dict[int, int]) -> Dict:
    """Extract fingerprint from a set of puzzle keys."""
    # Convert keys to bit stream
    all_bits = []
    for n in sorted(keys.keys()):
        key = keys[n]
        for bit_pos in range(n):
            all_bits.append((key >> bit_pos) & 1)

    bit_str = ''.join(str(b) for b in all_bits)

    # Count 8-bit patterns
    patterns = Counter()
    for i in range(len(bit_str) - 7):
        patterns[bit_str[i:i+8]] += 1

    # Calculate metrics
    key_list = [keys[i] for i in sorted(keys.keys())]

    # XOR ratio
    xor_ones = []
    for i in range(len(key_list) - 1):
        max_bits = max(key_list[i].bit_length(), key_list[i+1].bit_length())
        xor = key_list[i] ^ key_list[i+1]
        ones = bin(xor).count('1')
        xor_ones.append(ones / max_bits if max_bits > 0 else 0)
    xor_ratio = sum(xor_ones) / len(xor_ones) if xor_ones else 0

    # Serial correlation (normalized positions)
    positions = []
    for n, key in sorted(keys.items()):
        min_val = 2**(n-1) if n > 1 else 1
        max_val = 2**n - 1
        pos = (key - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        positions.append(pos)

    mean_pos = sum(positions) / len(positions)
    var_pos = sum((p - mean_pos)**2 for p in positions) / len(positions)

    if var_pos > 0:
        serial_corr = sum((positions[i] - mean_pos) * (positions[i+1] - mean_pos)
                          for i in range(len(positions)-1)) / ((len(positions)-1) * var_pos)
    else:
        serial_corr = 0

    # Bit frequencies
    bit_freqs = {}
    for bit_pos in [2, 30]:
        ones = sum(1 for n, k in keys.items() if n > bit_pos and (k >> bit_pos) & 1)
        total = sum(1 for n in keys.keys() if n > bit_pos)
        bit_freqs[f'bit{bit_pos}_freq'] = ones / total if total > 0 else 0

    return {
        'pattern_0xD6': patterns.get('11010110', 0),
        'pattern_0xBA': patterns.get('10111010', 0),
        'xor_ratio': xor_ratio,
        'serial_corr': serial_corr,
        'position_mean': mean_pos,
        **bit_freqs,
    }


def signature_distance(sig1: Dict, sig2: Dict) -> float:
    """Calculate distance between two signatures."""
    distance = 0

    # Pattern counts (normalize by expected)
    for pattern in ['pattern_0xD6', 'pattern_0xBA']:
        expected = 9.7  # From our analysis
        v1 = sig1.get(pattern, 0) / expected
        v2 = sig2.get(pattern, 0) / expected
        distance += (v1 - v2) ** 2

    # Other metrics
    for metric in ['xor_ratio', 'serial_corr', 'position_mean']:
        v1 = sig1.get(metric, 0)
        v2 = sig2.get(metric, 0)
        distance += (v1 - v2) ** 2

    # Bit frequencies
    for bit in ['bit2_freq', 'bit30_freq']:
        v1 = sig1.get(bit, 0.5)
        v2 = sig2.get(bit, 0.5)
        distance += (v1 - v2) ** 2

    return math.sqrt(distance)


# =============================================================================
# MAIN SEARCH
# =============================================================================

def generate_keys_with_rng(rng: RNGBase, n_puzzles: int = 70) -> Dict[int, int]:
    """Generate puzzle keys using a specific RNG."""
    keys = {}
    for n in range(1, n_puzzles + 1):
        min_val = 2**(n-1) if n > 1 else 1
        max_val = 2**n - 1
        keys[n] = rng.random_in_range(min_val, max_val)
    return keys


def search_rng_signatures():
    """Search for RNG that matches the puzzle fingerprint."""
    print("=" * 70)
    print("RNG SIGNATURE SEARCH")
    print("=" * 70)

    # Get target signature
    target_sig = extract_signature(PUZZLE_KEYS)
    print("\nTarget signature (from actual puzzles):")
    for k, v in target_sig.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # RNGs to test
    rng_classes = [
        ('Mersenne Twister', MersenneTwister),
        ('Xorshift128', Xorshift128),
        ('SHA256-Counter', SHA256Counter),
        ('SHA256-Timestamp', SHA256Timestamp),
        ('LCG (glibc)', lambda s: LCG(s, 'glibc')),
        ('LCG (MSVC)', lambda s: LCG(s, 'msvc')),
        ('LCG (Java)', lambda s: LCG(s, 'java')),
        ('OpenSSL Simulator', OpenSSLSimulator),
        ('Custom Bitcoin RNG', CustomBitcoinRNG),
    ]

    print("\n" + "=" * 70)
    print("TESTING RNG TYPES")
    print("=" * 70)

    best_matches = []

    for rng_name, rng_class in rng_classes:
        print(f"\nTesting {rng_name}...")

        # Try many seeds
        min_distance = float('inf')
        best_seed = None

        seeds_to_try = list(range(100000))  # First 100k seeds

        for seed in seeds_to_try:
            try:
                rng = rng_class(seed)
                keys = generate_keys_with_rng(rng)
                sig = extract_signature(keys)
                dist = signature_distance(sig, target_sig)

                if dist < min_distance:
                    min_distance = dist
                    best_seed = seed

            except Exception as e:
                continue

            if seed % 20000 == 0:
                print(f"  Tested {seed} seeds, best distance so far: {min_distance:.4f}")

        print(f"  Best seed: {best_seed}, distance: {min_distance:.4f}")
        best_matches.append((rng_name, best_seed, min_distance))

        # Show signature for best seed
        rng = rng_class(best_seed)
        keys = generate_keys_with_rng(rng)
        sig = extract_signature(keys)
        print(f"  Signature at best seed:")
        print(f"    pattern_0xD6: {sig['pattern_0xD6']} (target: {target_sig['pattern_0xD6']})")
        print(f"    pattern_0xBA: {sig['pattern_0xBA']} (target: {target_sig['pattern_0xBA']})")
        print(f"    xor_ratio: {sig['xor_ratio']:.4f} (target: {target_sig['xor_ratio']:.4f})")

    # Sort by distance
    best_matches.sort(key=lambda x: x[2])

    print("\n" + "=" * 70)
    print("RESULTS RANKED BY SIGNATURE MATCH")
    print("=" * 70)

    for rank, (rng_name, seed, distance) in enumerate(best_matches, 1):
        print(f"{rank}. {rng_name}: seed={seed}, distance={distance:.4f}")

    return best_matches


def deep_search_best_rng(rng_name: str, rng_class, seed_ranges: List[range]):
    """Deep search a specific RNG with more seeds."""
    print(f"\n{'=' * 70}")
    print(f"DEEP SEARCH: {rng_name}")
    print("=" * 70)

    target_sig = extract_signature(PUZZLE_KEYS)

    best_matches = []

    for seed_range in seed_ranges:
        print(f"\nSearching seed range {seed_range.start} to {seed_range.stop}...")

        for seed in seed_range:
            try:
                rng = rng_class(seed)
                keys = generate_keys_with_rng(rng)
                sig = extract_signature(keys)
                dist = signature_distance(sig, target_sig)

                if dist < 1.0:  # Keep only close matches
                    best_matches.append((seed, dist, sig))

            except:
                continue

            if seed % 100000 == 0 and seed > seed_range.start:
                print(f"  Tested {seed - seed_range.start} seeds...")

    # Sort and show best
    best_matches.sort(key=lambda x: x[1])

    print(f"\nTop 10 matches for {rng_name}:")
    for seed, dist, sig in best_matches[:10]:
        print(f"  Seed {seed}: distance={dist:.4f}")
        print(f"    patterns: D6={sig['pattern_0xD6']}, BA={sig['pattern_0xBA']}")

    return best_matches


def main():
    print("=" * 70)
    print("RNG SIGNATURE MATCHER")
    print("Finding the source of the Bitcoin puzzle fingerprint")
    print("=" * 70)

    # First pass: test all RNG types
    results = search_rng_signatures()

    # The best match tells us which RNG family to focus on
    if results:
        best_rng = results[0][0]
        print(f"\n\nBest match: {best_rng}")
        print("This RNG type produces the closest signature to the puzzles.")


if __name__ == "__main__":
    main()
