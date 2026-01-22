#!/usr/bin/env python3
"""
Deep Search: Python Mersenne Twister

Based on research, the Bitcoin puzzle creator likely used:
- Python script (most common for quick 2015 scripting)
- random.randint(2**(n-1), 2**n - 1) for each puzzle
- Possibly a SINGLE seed for reproducibility

If we find the seed, we can predict unsolved puzzles.

Search strategies:
1. Sequential seeds (0, 1, 2, ...)
2. Timestamp-based seeds (2015 era: ~1420000000 to ~1450000000)
3. Common "magic" seeds (42, 1337, 12345, etc.)
4. Bitcoin-related seeds (hashes, block numbers, etc.)
"""

import random
import hashlib
import time
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Actual puzzle keys
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


def generate_sequence(seed: int, n_puzzles: int = 70) -> Dict[int, int]:
    """Generate puzzle keys with Python's random module."""
    random.seed(seed)
    keys = {}
    for n in range(1, n_puzzles + 1):
        min_val = 2**(n-1) if n > 1 else 1
        max_val = 2**n - 1
        keys[n] = random.randint(min_val, max_val)
    return keys


def count_matches(generated: Dict[int, int], actual: Dict[int, int]) -> Tuple[int, List[int]]:
    """Count how many keys match."""
    matches = 0
    matched_puzzles = []
    for n in actual:
        if n in generated and generated[n] == actual[n]:
            matches += 1
            matched_puzzles.append(n)
    return matches, matched_puzzles


def test_seed(seed: int) -> Tuple[int, int, List[int]]:
    """Test a single seed. Returns (seed, match_count, matched_puzzles)."""
    generated = generate_sequence(seed)
    matches, matched = count_matches(generated, PUZZLE_KEYS)
    return seed, matches, matched


def search_sequential(start: int, end: int, report_interval: int = 1000000) -> List[Tuple[int, int, List[int]]]:
    """Search sequential seed range."""
    results = []
    best_so_far = 0

    for seed in range(start, end):
        seed, matches, matched = test_seed(seed)

        if matches >= 5:  # Keep anything with 5+ matches
            results.append((seed, matches, matched))

        if matches > best_so_far:
            best_so_far = matches
            print(f"  New best: seed={seed}, matches={matches}, puzzles={matched}")

        if seed % report_interval == 0 and seed > start:
            print(f"  Progress: tested {seed - start:,} seeds...")

    return results


def search_timestamps_2015():
    """
    Search timestamp-based seeds from 2015.

    The puzzle transaction was created January 15, 2015.
    Block 339849 contained the funding transaction.
    Timestamp range: Jan 1, 2015 to Feb 1, 2015
    """
    print("\n" + "=" * 70)
    print("TIMESTAMP SEARCH: January 2015")
    print("=" * 70)

    # January 2015 Unix timestamps
    jan_1_2015 = 1420070400
    feb_1_2015 = 1422748800

    results = []
    best = 0

    print(f"Testing {feb_1_2015 - jan_1_2015:,} timestamps...")

    for ts in range(jan_1_2015, feb_1_2015):
        _, matches, matched = test_seed(ts)

        if matches >= 5:
            results.append((ts, matches, matched))

        if matches > best:
            best = matches
            # Convert to human readable
            from datetime import datetime
            dt = datetime.fromtimestamp(ts)
            print(f"  New best: {dt} (ts={ts}), matches={matches}")

    return results


def search_common_seeds():
    """Search common/magic seeds that programmers often use."""
    print("\n" + "=" * 70)
    print("COMMON SEED SEARCH")
    print("=" * 70)

    common_seeds = [
        # Obvious choices
        0, 1, 42, 1337, 12345, 123456, 1234567, 12345678,
        # Hex patterns
        0xDEADBEEF, 0xCAFEBABE, 0xBEEFCAFE, 0xFEEDFACE,
        # Years
        2009, 2010, 2011, 2012, 2013, 2014, 2015,
        # Bitcoin related
        21000000,  # Max BTC
        100000000,  # Satoshis in 1 BTC
        # Block numbers around puzzle creation
        339849,  # Funding transaction block
        339800, 339900, 340000,
        # Genesis block hash as int
        int("000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f", 16) % (2**32),
    ]

    # Also add first 10000 integers
    common_seeds.extend(range(10000))

    results = []

    for seed in common_seeds:
        _, matches, matched = test_seed(seed)
        if matches >= 3:
            results.append((seed, matches, matched))
            print(f"  Seed {seed}: {matches} matches - puzzles {matched}")

    return results


def search_bitcoin_hashes():
    """
    Search seeds derived from Bitcoin block hashes.

    The puzzle was funded in block 339849.
    Maybe the creator used a block hash as seed?
    """
    print("\n" + "=" * 70)
    print("BITCOIN HASH-BASED SEED SEARCH")
    print("=" * 70)

    # We don't have actual block hashes, but we can try common derivations
    # For now, try hashing strings like "bitcoin", "puzzle", etc.

    test_strings = [
        b"bitcoin", b"puzzle", b"satoshi", b"nakamoto",
        b"bitcoin puzzle", b"bitcoin_puzzle", b"btc_puzzle",
        b"secret", b"private", b"key", b"privatekey",
        b"1", b"2", b"3", b"test", b"random",
        b"ECDSA", b"secp256k1",
    ]

    # Try various hash derivations
    results = []

    for s in test_strings:
        # Try SHA256
        h = hashlib.sha256(s).digest()
        seed = int.from_bytes(h[:4], 'big')  # First 4 bytes as seed
        _, matches, matched = test_seed(seed)
        if matches >= 3:
            results.append((f"sha256({s})[:4]", seed, matches, matched))
            print(f"  sha256({s})[:4] = {seed}: {matches} matches")

        # Try full hash as seed (mod 2^32)
        seed = int.from_bytes(h, 'big') % (2**32)
        _, matches, matched = test_seed(seed)
        if matches >= 3:
            results.append((f"sha256({s}) mod 2^32", seed, matches, matched))
            print(f"  sha256({s}) mod 2^32 = {seed}: {matches} matches")

    return results


def deep_search_parallel(start: int, end: int, n_workers: int = None):
    """Parallel search over a large range."""
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()

    print(f"\n{'=' * 70}")
    print(f"PARALLEL SEARCH: {start:,} to {end:,}")
    print(f"Using {n_workers} workers")
    print("=" * 70)

    chunk_size = (end - start) // n_workers
    chunks = [(start + i * chunk_size, start + (i + 1) * chunk_size) for i in range(n_workers)]
    chunks[-1] = (chunks[-1][0], end)  # Make sure we cover everything

    all_results = []

    def search_chunk(args):
        chunk_start, chunk_end = args
        results = []
        for seed in range(chunk_start, chunk_end):
            _, matches, matched = test_seed(seed)
            if matches >= 5:
                results.append((seed, matches, matched))
        return results

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(search_chunk, chunk): chunk for chunk in chunks}

        for future in as_completed(futures):
            chunk = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                print(f"  Chunk {chunk[0]:,}-{chunk[1]:,} done: {len(results)} candidates")
            except Exception as e:
                print(f"  Chunk {chunk} failed: {e}")

    return sorted(all_results, key=lambda x: -x[1])


def analyze_partial_matches():
    """
    If no single seed produces all matches, maybe:
    1. Different seed per puzzle
    2. Seed changed partway through
    3. Some manual adjustments
    """
    print("\n" + "=" * 70)
    print("PARTIAL MATCH ANALYSIS")
    print("=" * 70)

    # Find best seeds for different puzzle ranges
    ranges = [
        (1, 10, "First 10 puzzles"),
        (1, 20, "First 20 puzzles"),
        (1, 35, "First 35 puzzles (solved early)"),
        (36, 70, "Later puzzles"),
    ]

    for start, end, desc in ranges:
        print(f"\n{desc} (puzzles {start}-{end}):")

        # Create subset of keys
        subset = {n: PUZZLE_KEYS[n] for n in range(start, end + 1)}

        best = 0
        best_seed = None

        for seed in range(1000000):
            random.seed(seed)
            # Skip to puzzle `start`
            for _ in range(1, start):
                n = _
                random.randint(2**(n-1) if n > 1 else 1, 2**n - 1)

            # Generate for our range
            matches = 0
            for n in range(start, end + 1):
                min_val = 2**(n-1) if n > 1 else 1
                max_val = 2**n - 1
                generated = random.randint(min_val, max_val)
                if generated == PUZZLE_KEYS.get(n):
                    matches += 1

            if matches > best:
                best = matches
                best_seed = seed

        print(f"  Best seed: {best_seed}, matches: {best}/{end - start + 1}")


def main():
    print("=" * 70)
    print("PYTHON MERSENNE TWISTER DEEP SEARCH")
    print("Looking for the single seed that generated all puzzles")
    print("=" * 70)

    # Quick common seeds first
    print("\n[1/4] Testing common seeds...")
    common_results = search_common_seeds()

    # 2015 timestamps
    print("\n[2/4] Testing 2015 timestamps...")
    timestamp_results = search_timestamps_2015()

    # Bitcoin-related hashes
    print("\n[3/4] Testing Bitcoin hash-derived seeds...")
    hash_results = search_bitcoin_hashes()

    # Large sequential search
    print("\n[4/4] Sequential search (first 10M seeds)...")
    sequential_results = search_sequential(0, 10_000_000, report_interval=1_000_000)

    # Combine and rank all results
    all_results = []
    for seed, matches, matched in common_results + timestamp_results + sequential_results:
        all_results.append((seed, matches, matched))

    all_results.sort(key=lambda x: -x[1])

    print("\n" + "=" * 70)
    print("TOP RESULTS")
    print("=" * 70)

    for seed, matches, matched in all_results[:20]:
        print(f"  Seed {seed:>12}: {matches} matches - puzzles {matched}")

    # Also check partial matches
    print("\n[BONUS] Analyzing if seed changed partway through...")
    analyze_partial_matches()


if __name__ == "__main__":
    main()
