#!/usr/bin/env python3
"""
Fast parallel search for seed that generates puzzle 9.
"""
import random
from multiprocessing import Pool, cpu_count

PUZZLE_KEYS = {
    9: 0x1d3, 10: 0x202, 11: 0x483, 12: 0xa7b, 13: 0x1460, 14: 0x2930,
    15: 0x68f3, 16: 0xc936, 17: 0x1764f, 18: 0x3080d, 19: 0x5749f,
    20: 0xd2c55,
}


def search_chunk(args):
    start, end, target = args
    found = []

    for seed in range(start, end):
        random.seed(seed)
        if random.randint(256, 511) == target:
            # Check consecutive matches
            random.seed(seed)
            matches = 0
            for n in range(9, 21):
                min_val = 2**(n-1)
                max_val = 2**n - 1
                gen = random.randint(min_val, max_val)
                if gen == PUZZLE_KEYS.get(n):
                    matches += 1
                else:
                    break
            found.append((seed, matches))

    return found


def main():
    print("FAST PARALLEL SEARCH FOR PUZZLE 9+ SEED")
    print("=" * 70)

    target = 0x1d3  # Puzzle 9 value
    chunk_size = 5_000_000
    total_range = 100_000_000  # Search 100M

    chunks = []
    for start in range(0, total_range, chunk_size):
        end = min(start + chunk_size, total_range)
        chunks.append((start, end, target))

    print(f"Searching 0-{total_range//1_000_000}M using {cpu_count()} cores...")
    print(f"Looking for seed where randint(256, 511) == 0x{target:x}")
    print()

    all_results = []
    with Pool(cpu_count()) as p:
        results = p.map(search_chunk, chunks)

    for r in results:
        all_results.extend(r)

    print(f"Found {len(all_results)} seeds generating puzzle 9 = 0x{target:x}")

    # Sort by matches
    all_results.sort(key=lambda x: -x[1])

    print()
    print("Top seeds (by consecutive matches):")
    for seed, matches in all_results[:20]:
        print(f"  Seed {seed}: {matches} consecutive matches starting at puzzle 9")

    # Show best match details
    if all_results:
        best_seed = all_results[0][0]
        print()
        print(f"Best seed {best_seed} verification:")
        random.seed(best_seed)
        for n in range(9, 21):
            min_val = 2**(n-1)
            max_val = 2**n - 1
            gen = random.randint(min_val, max_val)
            actual = PUZZLE_KEYS.get(n)
            match = '✓' if gen == actual else '✗'
            print(f"  Puzzle {n}: generated=0x{gen:x}, actual=0x{actual:x} {match}")


if __name__ == "__main__":
    main()
