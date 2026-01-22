#!/usr/bin/env python3
"""
SHA256-Based Seed Search

The ML classifier gave SHA256-Counter 28.9% probability.
Common patterns in 2015:
  key[n] = SHA256(seed || n) mod range
  key[n] = SHA256(seed || str(n)) mod range
  key[n] = int(SHA256(seed || n).hexdigest()[:?], 16) mod range

Try various string seeds and derivation methods.
"""

import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

PUZZLE_KEYS = {
    1: 0x1, 2: 0x3, 3: 0x7, 4: 0x8, 5: 0x15, 6: 0x31, 7: 0x4c, 8: 0xe0,
    9: 0x1d3, 10: 0x202, 11: 0x483, 12: 0xa7b, 13: 0x1460, 14: 0x2930,
    15: 0x68f3, 16: 0xc936, 17: 0x1764f, 18: 0x3080d, 19: 0x5749f,
    20: 0xd2c55, 21: 0x1ba534, 22: 0x2de40f, 23: 0x556e52, 24: 0xdc2a04,
    25: 0x1fa5ee5, 26: 0x340326e, 27: 0x6ac3875, 28: 0xd916ce8,
    29: 0x17e2551e, 30: 0x3d94cd64,
}

FIRST_N = 20  # Check first 20 puzzles


def sha256_method1(seed_bytes: bytes, n: int) -> int:
    """SHA256(seed || n as 4-byte big-endian) mod range"""
    data = seed_bytes + n.to_bytes(4, 'big')
    h = hashlib.sha256(data).digest()
    raw = int.from_bytes(h, 'big')

    min_val = 2**(n-1) if n > 1 else 1
    max_val = 2**n - 1
    range_size = max_val - min_val + 1

    return min_val + (raw % range_size)


def sha256_method2(seed_bytes: bytes, n: int) -> int:
    """SHA256(seed || str(n)) mod range"""
    data = seed_bytes + str(n).encode()
    h = hashlib.sha256(data).digest()
    raw = int.from_bytes(h, 'big')

    min_val = 2**(n-1) if n > 1 else 1
    max_val = 2**n - 1
    range_size = max_val - min_val + 1

    return min_val + (raw % range_size)


def sha256_method3(seed_bytes: bytes, n: int) -> int:
    """SHA256(seed || n as little-endian) mod range"""
    data = seed_bytes + n.to_bytes(4, 'little')
    h = hashlib.sha256(data).digest()
    raw = int.from_bytes(h, 'big')

    min_val = 2**(n-1) if n > 1 else 1
    max_val = 2**n - 1
    range_size = max_val - min_val + 1

    return min_val + (raw % range_size)


def sha256_method4(seed_bytes: bytes, n: int) -> int:
    """Double SHA256 (Bitcoin style): SHA256(SHA256(seed || n))"""
    data = seed_bytes + n.to_bytes(4, 'big')
    h1 = hashlib.sha256(data).digest()
    h2 = hashlib.sha256(h1).digest()
    raw = int.from_bytes(h2, 'big')

    min_val = 2**(n-1) if n > 1 else 1
    max_val = 2**n - 1
    range_size = max_val - min_val + 1

    return min_val + (raw % range_size)


def sha256_method5(seed_bytes: bytes, n: int) -> int:
    """Take first n bits of hash"""
    data = seed_bytes + n.to_bytes(4, 'big')
    h = hashlib.sha256(data).digest()
    raw = int.from_bytes(h, 'big')

    # Take top n bits, ensure in valid range
    key = (raw >> (256 - n))
    min_val = 2**(n-1) if n > 1 else 1

    # Ensure high bit is set
    if key < min_val:
        key |= min_val

    return key


def sha256_method6(seed_bytes: bytes, n: int) -> int:
    """HMAC-style: SHA256(seed XOR pad || n)"""
    pad = bytes([0x36] * len(seed_bytes))
    xored = bytes(a ^ b for a, b in zip(seed_bytes.ljust(32, b'\x00'), (pad * 2)[:32]))
    data = xored + n.to_bytes(4, 'big')
    h = hashlib.sha256(data).digest()
    raw = int.from_bytes(h, 'big')

    min_val = 2**(n-1) if n > 1 else 1
    max_val = 2**n - 1
    range_size = max_val - min_val + 1

    return min_val + (raw % range_size)


METHODS = [
    ("SHA256(seed||n_be) mod range", sha256_method1),
    ("SHA256(seed||str(n)) mod range", sha256_method2),
    ("SHA256(seed||n_le) mod range", sha256_method3),
    ("Double SHA256 (Bitcoin)", sha256_method4),
    ("SHA256 top n bits", sha256_method5),
    ("HMAC-style", sha256_method6),
]


def test_seed_string(seed_str: str) -> list:
    """Test a string seed with all methods."""
    seed_bytes = seed_str.encode() if isinstance(seed_str, str) else seed_str
    results = []

    for method_name, method_func in METHODS:
        matches = 0
        matched_puzzles = []

        for n in range(1, FIRST_N + 1):
            generated = method_func(seed_bytes, n)
            if generated == PUZZLE_KEYS.get(n):
                matches += 1
                matched_puzzles.append(n)

        if matches >= 3:
            results.append((seed_str, method_name, matches, matched_puzzles))

    return results


def generate_seed_candidates():
    """Generate string seeds to try."""
    candidates = []

    # Common words
    words = [
        "bitcoin", "puzzle", "satoshi", "nakamoto", "btc", "crypto",
        "secret", "private", "key", "privatekey", "private_key",
        "random", "seed", "master", "root", "wallet",
        "challenge", "bounty", "reward", "prize",
        "test", "testing", "debug", "dev",
        "1", "2", "3", "password", "pass", "123456",
    ]
    candidates.extend(words)

    # With numbers
    for word in words[:10]:
        for i in range(100):
            candidates.append(f"{word}{i}")
            candidates.append(f"{word}_{i}")

    # Combinations
    for w1 in ["bitcoin", "btc", "puzzle"]:
        for w2 in ["puzzle", "key", "secret", "challenge"]:
            candidates.append(f"{w1}_{w2}")
            candidates.append(f"{w1}{w2}")
            candidates.append(f"{w1} {w2}")

    # Years and dates
    candidates.extend([str(y) for y in range(2009, 2020)])
    candidates.extend(["2015-01-15", "20150115", "2015/01/15"])

    # Hex strings
    candidates.extend([
        "deadbeef", "cafebabe", "0000000000000000",
        "ffffffffffffffff", "0123456789abcdef",
    ])

    # Bitcoin genesis block hash (partial)
    candidates.append("000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f")
    candidates.append("genesis")

    # Empty and simple
    candidates.extend(["", " ", "\n", "\t", "\x00"])

    # Numeric strings
    for i in range(10000):
        candidates.append(str(i))

    return list(set(candidates))


def search_string_seeds():
    """Search string-based seeds."""
    print("=" * 70)
    print("SHA256 STRING SEED SEARCH")
    print("=" * 70)

    candidates = generate_seed_candidates()
    print(f"Testing {len(candidates)} seed strings with {len(METHODS)} methods...")
    print()

    all_results = []
    best = 0

    for i, seed_str in enumerate(candidates):
        results = test_seed_string(seed_str)

        for seed, method, matches, puzzles in results:
            if matches > best:
                best = matches
                print(f"NEW BEST: '{seed}' with {method}: {matches}/{FIRST_N} matches")
                print(f"  Matched puzzles: {puzzles}")

            if matches >= 5:
                all_results.append((seed, method, matches, puzzles))

        if i % 5000 == 0 and i > 0:
            print(f"  Progress: {i}/{len(candidates)}, best so far: {best}")

    return sorted(all_results, key=lambda x: -x[2])


def search_numeric_seeds():
    """Search numeric seeds (as bytes)."""
    print("\n" + "=" * 70)
    print("SHA256 NUMERIC SEED SEARCH")
    print("=" * 70)

    all_results = []
    best = 0

    # Try first 10M numeric seeds
    print("Testing numeric seeds 0 to 10,000,000...")

    for seed_int in range(10_000_000):
        seed_bytes = seed_int.to_bytes(8, 'big')

        for method_name, method_func in METHODS:
            matches = 0
            matched_puzzles = []

            for n in range(1, FIRST_N + 1):
                generated = method_func(seed_bytes, n)
                if generated == PUZZLE_KEYS.get(n):
                    matches += 1
                    matched_puzzles.append(n)

            if matches > best:
                best = matches
                print(f"NEW BEST: seed={seed_int} with {method_name}: {matches}/{FIRST_N}")
                print(f"  Matched puzzles: {matched_puzzles}")

            if matches >= 5:
                all_results.append((seed_int, method_name, matches, matched_puzzles))

        if seed_int % 1_000_000 == 0 and seed_int > 0:
            print(f"  Progress: {seed_int//1_000_000}M, best: {best}")

    return sorted(all_results, key=lambda x: -x[2])


def reverse_engineer():
    """Try to reverse engineer the hash from known outputs."""
    print("\n" + "=" * 70)
    print("REVERSE ENGINEERING ATTEMPT")
    print("=" * 70)

    print("""
If key[n] = SHA256(seed || n) mod range, then:
- We know key[n] for n=1..70
- We know the range constraints
- We need to find 'seed' such that the equation holds

For small puzzles, the modulo operation loses information.
For large puzzles (n > 32), most of the hash output is used.

Let's check if any puzzle's key could directly be part of a SHA256 output...
""")

    # For puzzle 20 (20 bits), we have key = 0xd2c55
    # This could be the top/bottom 20 bits of SHA256(something)

    print("Checking if puzzle keys appear in common SHA256 outputs...")

    test_inputs = [
        b"", b"0", b"1", b"bitcoin", b"puzzle", b"satoshi",
        b"test", b"key", b"private", b"secret",
    ]

    for n in [15, 16, 17, 18, 19, 20]:
        key = PUZZLE_KEYS[n]
        print(f"\nPuzzle {n}: looking for 0x{key:x} in SHA256 outputs...")

        for inp in test_inputs:
            for suffix in range(1000):
                data = inp + str(suffix).encode()
                h = hashlib.sha256(data).hexdigest()

                # Check if key appears in first few hex chars
                key_hex = f"{key:x}"
                if key_hex in h[:20]:
                    print(f"  FOUND: SHA256({data}) contains {key_hex}")


def main():
    print("=" * 70)
    print("SHA256-BASED SEED SEARCH")
    print("Looking for seed where key[n] = f(SHA256(seed, n))")
    print("=" * 70)

    # String seeds first (faster)
    string_results = search_string_seeds()

    if string_results:
        print(f"\nTop string seed results:")
        for seed, method, matches, puzzles in string_results[:10]:
            print(f"  '{seed}' [{method}]: {matches} matches - {puzzles}")

    # Numeric seeds
    numeric_results = search_numeric_seeds()

    if numeric_results:
        print(f"\nTop numeric seed results:")
        for seed, method, matches, puzzles in numeric_results[:10]:
            print(f"  {seed} [{method}]: {matches} matches - {puzzles}")

    # Combine results
    all_results = string_results + numeric_results
    all_results.sort(key=lambda x: -x[2])

    print("\n" + "=" * 70)
    print("OVERALL BEST RESULTS")
    print("=" * 70)

    if all_results:
        for seed, method, matches, puzzles in all_results[:10]:
            print(f"  {seed} [{method}]: {matches}/{FIRST_N} matches")
    else:
        print("No significant matches found with SHA256 methods.")

    # Try reverse engineering
    reverse_engineer()


if __name__ == "__main__":
    main()
