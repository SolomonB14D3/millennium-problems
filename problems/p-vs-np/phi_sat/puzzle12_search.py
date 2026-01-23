#!/usr/bin/env python3
"""
Search for seed that generates puzzle 12+.
"""
import random
from multiprocessing import Pool, cpu_count

PUZZLE_KEYS = {
    12: 0xa7b, 13: 0x1460, 14: 0x2930, 15: 0x68f3, 16: 0xc936,
    17: 0x1764f, 18: 0x3080d, 19: 0x5749f, 20: 0xd2c55, 21: 0x1ba534,
    22: 0x2de40f, 23: 0x556e52, 24: 0xdc2a04,
}


def search_chunk(args):
    start, end, target = args
    found = []

    for seed in range(start, end):
        random.seed(seed)
        min_val = 2**(12-1)  # 2048
        max_val = 2**12 - 1  # 4095
        if random.randint(min_val, max_val) == target:
            # Check consecutive matches
            random.seed(seed)
            matches = 0
            for n in range(12, 25):
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
    print("SEARCH FOR PUZZLE 12+ SEED")
    print("=" * 70)

    target = PUZZLE_KEYS[12]  # 0xa7b
    chunk_size = 5_000_000
    total_range = 100_000_000

    chunks = []
    for start in range(0, total_range, chunk_size):
        end = min(start + chunk_size, total_range)
        chunks.append((start, end, target))

    print(f"Searching 0-{total_range//1_000_000}M using {cpu_count()} cores...")
    print(f"Looking for seed where randint(2048, 4095) == 0x{target:x}")
    print()

    all_results = []
    with Pool(cpu_count()) as p:
        results = p.map(search_chunk, chunks)

    for r in results:
        all_results.extend(r)

    print(f"Found {len(all_results)} seeds generating puzzle 12 = 0x{target:x}")

    # Sort by matches
    all_results.sort(key=lambda x: -x[1])

    print()
    print("Top seeds (by consecutive matches):")
    for seed, matches in all_results[:20]:
        print(f"  Seed {seed}: {matches} consecutive matches starting at puzzle 12")

    # Verify best
    if all_results:
        best_seed = all_results[0][0]
        print()
        print(f"Best seed {best_seed} verification:")
        random.seed(best_seed)
        for n in range(12, 25):
            min_val = 2**(n-1)
            max_val = 2**n - 1
            gen = random.randint(min_val, max_val)
            actual = PUZZLE_KEYS.get(n)
            match = '✓' if gen == actual else '✗'
            print(f"  Puzzle {n}: generated=0x{gen:x}, actual=0x{actual:x} {match}")


if __name__ == "__main__":
    main()
