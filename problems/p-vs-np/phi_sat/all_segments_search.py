#!/usr/bin/env python3
"""
Search for seeds for each puzzle segment.
"""
import random
from multiprocessing import Pool, cpu_count

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


def search_for_puzzle_start(start_puzzle, search_range=100_000_000):
    """Find seed that generates consecutive puzzles starting at start_puzzle."""
    target = PUZZLE_KEYS[start_puzzle]
    min_val = 2**(start_puzzle-1) if start_puzzle > 1 else 1
    max_val = 2**start_puzzle - 1

    best_seed = None
    best_matches = 0

    for seed in range(search_range):
        random.seed(seed)
        gen = random.randint(min_val, max_val)

        if gen == target:
            # Count consecutive matches
            random.seed(seed)
            matches = 0
            for n in range(start_puzzle, start_puzzle + 15):
                if n not in PUZZLE_KEYS:
                    break
                n_min = 2**(n-1) if n > 1 else 1
                n_max = 2**n - 1
                g = random.randint(n_min, n_max)
                if g == PUZZLE_KEYS[n]:
                    matches += 1
                else:
                    break

            if matches > best_matches:
                best_matches = matches
                best_seed = seed

    return start_puzzle, best_seed, best_matches


def main():
    print("FINDING SEEDS FOR ALL PUZZLE SEGMENTS")
    print("=" * 70)
    print()

    segments = []
    current = 1

    # Already found segments
    print("KNOWN SEGMENTS:")
    print("  Puzzles 1-8:  seed 34378104 (8 consecutive matches)")
    print("  Puzzles 9-11: seed 78372297 (3 consecutive matches)")
    print("  Puzzles 12-13: seed 2408880 (2 consecutive matches)")

    # Search for remaining starting points
    print()
    print("SEARCHING FOR REMAINING SEGMENTS...")
    print()

    for start in [14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]:
        if start not in PUZZLE_KEYS:
            continue

        result = search_for_puzzle_start(start, 50_000_000)
        start_puzzle, best_seed, matches = result

        if matches >= 2:
            print(f"  Puzzles {start_puzzle}-{start_puzzle+matches-1}: "
                  f"seed {best_seed} ({matches} consecutive matches)")

            # Verify
            random.seed(best_seed)
            for n in range(start_puzzle, start_puzzle + matches + 2):
                if n not in PUZZLE_KEYS:
                    break
                n_min = 2**(n-1) if n > 1 else 1
                n_max = 2**n - 1
                gen = random.randint(n_min, n_max)
                actual = PUZZLE_KEYS[n]
                match = '✓' if gen == actual else '✗'
                print(f"    Puzzle {n}: gen=0x{gen:x}, actual=0x{actual:x} {match}")


if __name__ == "__main__":
    main()
