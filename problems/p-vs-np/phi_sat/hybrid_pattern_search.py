#!/usr/bin/env python3
"""
Hybrid Pattern Search

If the method changed mid-stream, that change was programmed.
The pattern of change is itself a fingerprint.

Hypotheses:
1. Different seed per puzzle: seed[n] = f(n, master_seed)
2. Grouped seeds: puzzles 1-8 one seed, 9-16 another, etc.
3. Seed derived from puzzle number: seed = n * k + offset
4. Two RNGs interleaved
5. Break at 8 is significant (8 bits = 1 byte boundary)
"""

import random
import hashlib
from typing import Dict, List, Tuple

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


def find_per_puzzle_seeds():
    """
    For each puzzle, find if there's a simple seed that generates it.
    seed[n] = n, seed[n] = n * k, seed[n] = hash(n), etc.
    """
    print("=" * 70)
    print("PER-PUZZLE SEED SEARCH")
    print("If each puzzle has its own seed, find the pattern")
    print("=" * 70)

    # For each puzzle, what seed would generate it?
    puzzle_seeds = {}

    for n in range(1, 41):
        target = PUZZLE_KEYS.get(n)
        if target is None:
            continue

        min_val = 2**(n-1) if n > 1 else 1
        max_val = 2**n - 1

        # Try seeds 0 to 10M to find one that generates this exact puzzle
        found_seeds = []

        for seed in range(1_000_000):
            random.seed(seed)
            # Generate puzzle n directly
            generated = random.randint(min_val, max_val)

            if generated == target:
                found_seeds.append(seed)
                if len(found_seeds) >= 5:
                    break

        if found_seeds:
            puzzle_seeds[n] = found_seeds
            print(f"Puzzle {n:2d}: key=0x{target:x}, seeds that work: {found_seeds[:5]}")
        else:
            print(f"Puzzle {n:2d}: key=0x{target:x}, no simple seed found in 0-1M")

    # Look for pattern in the seeds
    print("\n" + "-" * 50)
    print("Looking for pattern in per-puzzle seeds...")

    if puzzle_seeds:
        # Check if seeds follow arithmetic progression
        first_seeds = {n: seeds[0] for n, seeds in puzzle_seeds.items() if seeds}

        print("\nFirst seed for each puzzle:")
        for n in sorted(first_seeds.keys()):
            print(f"  Puzzle {n}: seed = {first_seeds[n]}")

        # Check differences
        print("\nDifferences between consecutive puzzle seeds:")
        prev_seed = None
        prev_n = None
        for n in sorted(first_seeds.keys()):
            if prev_seed is not None:
                diff = first_seeds[n] - prev_seed
                print(f"  Puzzle {prev_n} -> {n}: diff = {diff}")
            prev_seed = first_seeds[n]
            prev_n = n

    return puzzle_seeds


def find_grouped_seeds():
    """
    Maybe puzzles are grouped: 1-8, 9-16, 17-24, etc.
    Each group has its own seed.
    """
    print("\n" + "=" * 70)
    print("GROUPED SEED SEARCH")
    print("Looking for seeds per byte-boundary groups")
    print("=" * 70)

    groups = [
        (1, 8, "1 byte"),
        (9, 16, "2 bytes"),
        (17, 24, "3 bytes"),
        (25, 32, "4 bytes"),
        (33, 40, "5 bytes"),
    ]

    for start, end, desc in groups:
        print(f"\nGroup {start}-{end} ({desc}):")

        best_seed = None
        best_matches = 0

        for seed in range(10_000_000):
            random.seed(seed)

            # Skip to start
            for n in range(1, start):
                min_val = 2**(n-1) if n > 1 else 1
                max_val = 2**n - 1
                random.randint(min_val, max_val)

            # Generate for our group
            matches = 0
            for n in range(start, end + 1):
                if n not in PUZZLE_KEYS:
                    continue
                min_val = 2**(n-1) if n > 1 else 1
                max_val = 2**n - 1
                generated = random.randint(min_val, max_val)
                if generated == PUZZLE_KEYS[n]:
                    matches += 1

            if matches > best_matches:
                best_matches = matches
                best_seed = seed
                if matches >= (end - start + 1) // 2:
                    print(f"  Good seed: {seed} matches {matches}/{end-start+1}")

            if seed % 2_000_000 == 0 and seed > 0:
                print(f"  Progress: {seed//1_000_000}M, best: {best_matches}")

        print(f"  Best for group: seed={best_seed}, matches={best_matches}")


def find_seed_formula():
    """
    Maybe seed = f(n) for each puzzle.
    Try: seed = n, seed = n^2, seed = n * k, seed = hash(n)
    """
    print("\n" + "=" * 70)
    print("SEED FORMULA SEARCH")
    print("Looking for seed = f(puzzle_number)")
    print("=" * 70)

    formulas = [
        ("seed = n", lambda n: n),
        ("seed = n - 1", lambda n: n - 1),
        ("seed = n * 2", lambda n: n * 2),
        ("seed = n * 100", lambda n: n * 100),
        ("seed = n * 1000", lambda n: n * 1000),
        ("seed = n^2", lambda n: n ** 2),
        ("seed = 2^n", lambda n: 2 ** n),
        ("seed = n + 1000", lambda n: n + 1000),
        ("seed = n + 10000", lambda n: n + 10000),
        ("seed = hash(n) mod 2^32", lambda n: int(hashlib.sha256(str(n).encode()).hexdigest(), 16) % (2**32)),
        ("seed = hash(n) first 4 bytes", lambda n: int.from_bytes(hashlib.sha256(str(n).encode()).digest()[:4], 'big')),
    ]

    for formula_name, formula_func in formulas:
        matches = 0
        matched = []

        for n in range(1, 41):
            if n not in PUZZLE_KEYS:
                continue

            seed = formula_func(n)
            random.seed(seed)

            min_val = 2**(n-1) if n > 1 else 1
            max_val = 2**n - 1
            generated = random.randint(min_val, max_val)

            if generated == PUZZLE_KEYS[n]:
                matches += 1
                matched.append(n)

        if matches >= 3:
            print(f"{formula_name}: {matches}/40 matches - {matched[:10]}...")


def find_offset_multiplier():
    """
    seed = n * multiplier + offset
    Brute force search for multiplier and offset.
    """
    print("\n" + "=" * 70)
    print("MULTIPLIER + OFFSET SEARCH")
    print("Looking for seed = n * k + offset")
    print("=" * 70)

    best = 0
    best_params = None

    for multiplier in range(1, 100000):
        for offset in range(0, 1000):
            matches = 0

            for n in range(1, 21):  # First 20 puzzles
                if n not in PUZZLE_KEYS:
                    continue

                seed = n * multiplier + offset
                random.seed(seed)

                min_val = 2**(n-1) if n > 1 else 1
                max_val = 2**n - 1
                generated = random.randint(min_val, max_val)

                if generated == PUZZLE_KEYS[n]:
                    matches += 1

            if matches > best:
                best = matches
                best_params = (multiplier, offset)
                print(f"  NEW BEST: k={multiplier}, offset={offset}, matches={matches}/20")

        if multiplier % 10000 == 0:
            print(f"  Progress: k={multiplier}, best={best}")

    print(f"\nBest: k={best_params[0]}, offset={best_params[1]}, matches={best}")


def analyze_break_at_8():
    """
    Why does the break happen at puzzle 8?
    8 bits = 1 byte. Maybe byte boundary matters.
    """
    print("\n" + "=" * 70)
    print("ANALYZING THE BREAK AT PUZZLE 8")
    print("=" * 70)

    print("""
Puzzle 8 is the last 1-byte puzzle (values 128-255 fit in 1 byte).
Puzzle 9 is the first 2-byte puzzle (values 256-511 need 2 bytes).

This could indicate:
1. The code processed byte-sized chunks differently
2. Different RNG or seeding for multi-byte keys
3. A loop boundary in the generation code

Let's check if puzzle 9-16 (2-byte) has a consistent seed...
""")

    # Find best seed for puzzles 9-16 specifically
    print("Searching for seed that generates puzzles 9-16...")

    best = 0
    best_seed = None

    for seed in range(10_000_000):
        random.seed(seed)

        # Generate puzzles 9-16
        matches = 0
        for n in range(9, 17):
            min_val = 2**(n-1)
            max_val = 2**n - 1
            generated = random.randint(min_val, max_val)
            if n in PUZZLE_KEYS and generated == PUZZLE_KEYS[n]:
                matches += 1

        if matches > best:
            best = matches
            best_seed = seed
            print(f"  Seed {seed}: matches {matches}/8 for puzzles 9-16")

        if seed % 2_000_000 == 0 and seed > 0:
            print(f"  Progress: {seed//1_000_000}M")

    print(f"\nBest for 9-16: seed={best_seed}, matches={best}/8")

    # Check 17-24 (3-byte puzzles)
    print("\nSearching for seed that generates puzzles 17-24...")

    best = 0
    best_seed = None

    for seed in range(10_000_000):
        random.seed(seed)

        matches = 0
        for n in range(17, 25):
            min_val = 2**(n-1)
            max_val = 2**n - 1
            generated = random.randint(min_val, max_val)
            if n in PUZZLE_KEYS and generated == PUZZLE_KEYS[n]:
                matches += 1

        if matches > best:
            best = matches
            best_seed = seed
            if matches >= 4:
                print(f"  Seed {seed}: matches {matches}/8")

        if seed % 2_000_000 == 0 and seed > 0:
            print(f"  Progress: {seed//1_000_000}M")

    print(f"\nBest for 17-24: seed={best_seed}, matches={best}/8")


def main():
    print("=" * 70)
    print("HYBRID PATTERN SEARCH")
    print("The method change at puzzle 9 is itself programmed")
    print("=" * 70)

    # Find per-puzzle seeds
    per_puzzle_seeds = find_per_puzzle_seeds()

    # Try formula-based seeds
    find_seed_formula()

    # Analyze the break at 8
    analyze_break_at_8()


if __name__ == "__main__":
    main()
