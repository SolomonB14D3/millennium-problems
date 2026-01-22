#!/usr/bin/env python3
"""
Mersenne Twister Seed Search

The ML classifier says it's likely Mersenne Twister (Python random).
Now find the seed.

Search space: 2^32 = 4,294,967,296 possibilities
But we can prioritize:
1. 2015 timestamps (puzzle created ~Jan 15, 2015)
2. Common programmer seeds
3. Bitcoin-related values
4. Sequential from 0
"""

import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import sys

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
}

# Just use first 40 puzzles for faster matching
FIRST_N = 15  # Match first 15 for speed


def test_seed(seed: int) -> tuple:
    """Test a single seed. Returns (seed, matches, first_mismatch)."""
    random.seed(seed)

    matches = 0
    first_mismatch = None

    for n in range(1, FIRST_N + 1):
        min_val = 2**(n-1) if n > 1 else 1
        max_val = 2**n - 1
        generated = random.randint(min_val, max_val)

        if generated == PUZZLE_KEYS[n]:
            matches += 1
        elif first_mismatch is None:
            first_mismatch = n

    return seed, matches, first_mismatch


def search_range(start: int, end: int) -> list:
    """Search a range of seeds."""
    results = []
    best = 0

    for seed in range(start, end):
        _, matches, _ = test_seed(seed)

        if matches > best:
            best = matches
            results.append((seed, matches))

        if matches >= 8:  # Keep high matches
            results.append((seed, matches))

    return results


def parallel_search(start: int, end: int, n_workers: int = None):
    """Parallel search over seed range."""
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()

    chunk_size = 1_000_000  # 1M seeds per chunk
    chunks = []
    for i in range(start, end, chunk_size):
        chunks.append((i, min(i + chunk_size, end)))

    print(f"Searching {end - start:,} seeds using {n_workers} workers...")
    print(f"Split into {len(chunks)} chunks of ~{chunk_size:,} seeds each")

    all_results = []
    best_global = 0
    processed = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(search_range, c[0], c[1]): c for c in chunks}

        for future in as_completed(futures):
            chunk = futures[future]
            try:
                results = future.result()
                for seed, matches in results:
                    if matches > best_global:
                        best_global = matches
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        print(f"  NEW BEST: seed={seed}, matches={matches}/{FIRST_N} "
                              f"({processed:,} tested, {rate:,.0f}/sec)")
                    if matches >= 8:
                        all_results.append((seed, matches))

                processed += chunk[1] - chunk[0]

            except Exception as e:
                print(f"  Chunk {chunk} failed: {e}")

    return sorted(all_results, key=lambda x: -x[1])


def search_priority_seeds():
    """Search high-priority seed candidates first."""
    print("=" * 70)
    print("PRIORITY SEED SEARCH")
    print("=" * 70)

    priority_seeds = []

    # 1. Common programmer seeds
    priority_seeds.extend([
        0, 1, 42, 123, 1337, 31337,
        12345, 123456, 1234567, 12345678, 123456789,
        0xDEADBEEF, 0xCAFEBABE, 0xBEEFCAFE,
        999, 1000, 9999, 10000,
    ])

    # 2. Years and dates
    priority_seeds.extend([2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016])
    priority_seeds.extend([20150115, 20150114, 20150116])  # YYYYMMDD

    # 3. Bitcoin-related
    priority_seeds.extend([
        21000000,       # Max BTC supply
        100000000,      # Satoshis per BTC
        339849,         # Block number of puzzle tx
        339850, 339848,
        0,              # Genesis
    ])

    # 4. 2015 timestamps (Jan 1 - Feb 28, 2015)
    jan_1_2015 = 1420070400
    mar_1_2015 = 1425168000
    priority_seeds.extend(range(jan_1_2015, mar_1_2015, 60))  # Every minute

    # 5. First 100k sequential
    priority_seeds.extend(range(100000))

    print(f"Testing {len(priority_seeds):,} priority seeds...")

    best = 0
    best_seeds = []

    for i, seed in enumerate(priority_seeds):
        if seed < 0 or seed > 2**32:
            continue

        _, matches, first_mismatch = test_seed(seed)

        if matches > best:
            best = matches
            print(f"  NEW BEST: seed={seed}, matches={matches}/{FIRST_N}")
            best_seeds.append((seed, matches))

        if i % 100000 == 0 and i > 0:
            print(f"  Progress: {i:,}/{len(priority_seeds):,}")

    return best_seeds


def full_search():
    """Full 32-bit search."""
    print("\n" + "=" * 70)
    print("FULL 32-BIT SEARCH")
    print("=" * 70)

    # Search in chunks
    total = 2**32
    searched = 0

    # Start with first 100M
    print("\n[Phase 1] First 100 million seeds...")
    results = parallel_search(0, 100_000_000)

    if results and results[0][1] >= FIRST_N:
        return results

    print(f"\nBest so far: {results[0] if results else 'none'}")

    # Continue with next chunks if needed
    print("\n[Phase 2] Seeds 100M - 1B...")
    results2 = parallel_search(100_000_000, 1_000_000_000)
    results.extend(results2)

    return sorted(results, key=lambda x: -x[1])


def verify_seed(seed: int, n_puzzles: int = 70):
    """Verify a seed by generating all puzzles."""
    print(f"\n{'=' * 70}")
    print(f"VERIFYING SEED: {seed}")
    print("=" * 70)

    random.seed(seed)

    matches = 0
    for n in range(1, n_puzzles + 1):
        min_val = 2**(n-1) if n > 1 else 1
        max_val = 2**n - 1
        generated = random.randint(min_val, max_val)
        actual = PUZZLE_KEYS.get(n)

        if actual is not None:
            match = "✓" if generated == actual else "✗"
            if generated == actual:
                matches += 1
            print(f"  Puzzle {n:2d}: generated=0x{generated:x}, actual=0x{actual:x} {match}")
        else:
            print(f"  Puzzle {n:2d}: generated=0x{generated:x} (no actual to compare)")

    print(f"\nTotal matches: {matches}/{len(PUZZLE_KEYS)}")
    return matches


def main():
    print("=" * 70)
    print("MERSENNE TWISTER SEED SEARCH")
    print(f"Looking for seed that generates first {FIRST_N} puzzle keys")
    print("=" * 70)

    # Phase 1: Priority seeds
    priority_results = search_priority_seeds()

    if priority_results:
        best_seed, best_matches = max(priority_results, key=lambda x: x[1])
        print(f"\nBest from priority search: seed={best_seed}, matches={best_matches}")

        if best_matches >= FIRST_N:
            print("\n*** FOUND PERFECT MATCH! ***")
            verify_seed(best_seed)
            return

    # Phase 2: Full search
    print("\nNo perfect match in priority seeds. Starting full search...")
    full_results = full_search()

    if full_results:
        best_seed, best_matches = full_results[0]
        print(f"\nBest overall: seed={best_seed}, matches={best_matches}")
        verify_seed(best_seed)

        print("\nTop 10 seeds:")
        for seed, matches in full_results[:10]:
            print(f"  seed={seed}: {matches} matches")


if __name__ == "__main__":
    main()
