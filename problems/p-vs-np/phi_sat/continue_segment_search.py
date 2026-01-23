#!/usr/bin/env python3
"""
Continue searching for seed segments starting at puzzle 14.
"""
import random
from multiprocessing import Pool, cpu_count

PUZZLE_KEYS = {
    14: 0x2930, 15: 0x68f3, 16: 0xc936, 17: 0x1764f, 18: 0x3080d, 19: 0x5749f,
    20: 0xd2c55, 21: 0x1ba534, 22: 0x2de40f, 23: 0x556e52, 24: 0xdc2a04,
    25: 0x1fa5ee5, 26: 0x340326e, 27: 0x6ac3875, 28: 0xd916ce8,
    29: 0x17e2551e, 30: 0x3d94cd64, 31: 0x7d4fe747, 32: 0xb862a62e,
}


def search_chunk(args):
    start_puzzle, chunk_start, chunk_end = args
    target = PUZZLE_KEYS[start_puzzle]
    min_val = 2**(start_puzzle-1)
    max_val = 2**start_puzzle - 1
    found = []

    for seed in range(chunk_start, chunk_end):
        random.seed(seed)
        gen = random.randint(min_val, max_val)

        if gen == target:
            # Count consecutive matches
            random.seed(seed)
            matches = 0
            for n in range(start_puzzle, start_puzzle + 20):
                if n not in PUZZLE_KEYS:
                    break
                n_min = 2**(n-1)
                n_max = 2**n - 1
                g = random.randint(n_min, n_max)
                if g == PUZZLE_KEYS[n]:
                    matches += 1
                else:
                    break

            if matches >= 2:
                found.append((seed, matches))

    return found


def main():
    print("SEARCHING FOR SEEDS STARTING AT PUZZLE 14")
    print("=" * 70)

    start_puzzle = 14
    total_range = 100_000_000
    chunk_size = 5_000_000

    chunks = []
    for start in range(0, total_range, chunk_size):
        end = min(start + chunk_size, total_range)
        chunks.append((start_puzzle, start, end))

    print(f"Searching 0-{total_range//1_000_000}M using {cpu_count()} cores...")
    print(f"Looking for seed where puzzle {start_puzzle} = 0x{PUZZLE_KEYS[start_puzzle]:x}")
    print()

    all_results = []
    with Pool(cpu_count()) as p:
        results = p.map(search_chunk, chunks)

    for r in results:
        all_results.extend(r)

    # Sort by matches
    all_results.sort(key=lambda x: -x[1])

    print(f"Found {len(all_results)} seeds with 2+ consecutive matches")
    print()
    print("Top seeds:")
    for seed, matches in all_results[:10]:
        print(f"  Seed {seed}: {matches} consecutive matches")

        # Verify
        random.seed(seed)
        for n in range(start_puzzle, start_puzzle + matches + 1):
            if n not in PUZZLE_KEYS:
                break
            n_min = 2**(n-1)
            n_max = 2**n - 1
            gen = random.randint(n_min, n_max)
            actual = PUZZLE_KEYS[n]
            match = '✓' if gen == actual else '✗'
            print(f"    Puzzle {n}: gen=0x{gen:x}, actual=0x{actual:x} {match}")
        print()


if __name__ == "__main__":
    main()
