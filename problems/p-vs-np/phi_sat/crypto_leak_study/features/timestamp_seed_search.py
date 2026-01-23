#!/usr/bin/env python3
"""
Timestamp-Based Seed Search

The puzzle was created in 2015. Common poor seeding practices:
1. time.time() as seed (Unix timestamp)
2. time.time() * 1000 (milliseconds)
3. Date-based: YYYYMMDD, YYYYMMDDHH, etc.

Search for seeds that produce puzzles 14-15 (the transition point).
"""

import random
import time
from datetime import datetime, timedelta
import numpy as np
from typing import List, Tuple

PUZZLE_KEYS = {
    14: 0x2930,
    15: 0x68f3,
    16: 0xc936,
    17: 0x1764f,
    18: 0x3080d,
    19: 0x5749f,
    20: 0xd2c55,
}


def test_seed(seed: int, start_puzzle: int = 14, num_puzzles: int = 3) -> int:
    """Test a seed and return number of matching puzzles."""
    random.seed(seed)
    matches = 0

    for p in range(start_puzzle, start_puzzle + num_puzzles):
        if p not in PUZZLE_KEYS:
            break

        max_val = (1 << p) - 1
        min_val = 1 << (p - 1)
        generated = random.randint(min_val, max_val)

        if generated == PUZZLE_KEYS[p]:
            matches += 1
        else:
            break  # Stop at first mismatch

    return matches


def search_unix_timestamps():
    """Search Unix timestamps from 2015."""
    print("="*70)
    print("UNIX TIMESTAMP SEARCH (2015)")
    print("="*70)

    # Puzzle was created around Jan 2015
    # Search Jan 1, 2015 to Dec 31, 2015

    start_date = datetime(2015, 1, 1)
    end_date = datetime(2015, 12, 31, 23, 59, 59)

    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    print(f"Searching {end_ts - start_ts:,} timestamps...")

    best_matches = 0
    best_seeds = []

    for ts in range(start_ts, end_ts + 1):
        matches = test_seed(ts)

        if matches > best_matches:
            best_matches = matches
            best_seeds = [(ts, datetime.fromtimestamp(ts))]
            print(f"  New best: {matches} matches, seed={ts} ({datetime.fromtimestamp(ts)})")
        elif matches == best_matches and matches > 0:
            best_seeds.append((ts, datetime.fromtimestamp(ts)))

        if ts % 1000000 == 0:
            print(f"  Progress: {ts - start_ts:,} / {end_ts - start_ts:,}")

    print(f"\nBest: {best_matches} consecutive matches")
    if best_matches >= 2:
        print("Found promising timestamps:")
        for seed, dt in best_seeds[:10]:
            print(f"  {seed} = {dt}")

    return best_matches, best_seeds


def search_milliseconds():
    """Search millisecond timestamps."""
    print("\n" + "="*70)
    print("MILLISECOND TIMESTAMP SEARCH (2015)")
    print("="*70)

    # Jan 1, 2015 in milliseconds
    start_ms = 1420070400000
    # Search 1 hour of milliseconds (3.6M values)
    end_ms = start_ms + 3600000

    print(f"Searching {end_ms - start_ms:,} millisecond timestamps...")

    best_matches = 0

    for ms in range(start_ms, end_ms):
        matches = test_seed(ms)

        if matches > best_matches:
            best_matches = matches
            ts_sec = ms // 1000
            print(f"  New best: {matches} matches, seed={ms} ({datetime.fromtimestamp(ts_sec)})")

        if ms % 500000 == 0:
            print(f"  Progress: {ms - start_ms:,} / {end_ms - start_ms:,}")

    print(f"\nBest from milliseconds: {best_matches} consecutive matches")
    return best_matches


def search_date_formats():
    """Search date-based seeds (YYYYMMDD, etc.)."""
    print("\n" + "="*70)
    print("DATE FORMAT SEARCH")
    print("="*70)

    formats_found = []

    # YYYYMMDD format
    print("\nFormat: YYYYMMDD")
    for year in range(2014, 2018):
        for month in range(1, 13):
            for day in range(1, 32):
                try:
                    dt = datetime(year, month, day)
                    seed = int(dt.strftime("%Y%m%d"))
                    matches = test_seed(seed)
                    if matches >= 2:
                        print(f"  {seed}: {matches} matches")
                        formats_found.append((seed, matches, f"{year}-{month:02d}-{day:02d}"))
                except:
                    pass

    # YYYYMMDDHH format
    print("\nFormat: YYYYMMDDHH")
    for year in range(2015, 2016):
        for month in range(1, 13):
            for day in range(1, 29):  # Conservative
                for hour in range(24):
                    try:
                        seed = int(f"{year}{month:02d}{day:02d}{hour:02d}")
                        matches = test_seed(seed)
                        if matches >= 2:
                            print(f"  {seed}: {matches} matches")
                            formats_found.append((seed, matches, f"{year}-{month:02d}-{day:02d} {hour:02d}:00"))
                    except:
                        pass

    # Simple incrementing from interesting numbers
    print("\nFormat: Sequential from key values")
    for base in [0, 1, 14, 100, 1000, 10000, 100000, 1000000]:
        for offset in range(10000):
            seed = base + offset
            matches = test_seed(seed)
            if matches >= 2:
                print(f"  {seed}: {matches} matches")
                formats_found.append((seed, matches, f"base={base}+{offset}"))

    return formats_found


def search_derived_from_known_seeds():
    """
    We know seeds 34378104, 78372297, 2408880 for puzzles 1-13.
    Maybe puzzle 14+ uses a seed derived from these?
    """
    print("\n" + "="*70)
    print("DERIVED SEED SEARCH")
    print("="*70)

    known_seeds = [34378104, 78372297, 2408880]

    # Try various combinations
    candidates = []

    # Sum of seeds
    candidates.append(sum(known_seeds))

    # Products
    candidates.append(known_seeds[0] * known_seeds[1])
    candidates.append(known_seeds[1] * known_seeds[2])

    # XOR
    candidates.append(known_seeds[0] ^ known_seeds[1] ^ known_seeds[2])

    # Hash-like operations
    import hashlib
    for s in known_seeds:
        h = hashlib.sha256(str(s).encode()).hexdigest()
        candidates.append(int(h[:8], 16))
        candidates.append(int(h[:16], 16) % (2**32))

    # Concatenations
    candidates.append(int(str(known_seeds[0])[:4] + str(known_seeds[1])[:4]))
    candidates.append(int(str(known_seeds[2]) + str(known_seeds[0])[:3]))

    # Plus/minus variations
    for s in known_seeds:
        for delta in range(-1000, 1001):
            candidates.append(s + delta)

    print(f"Testing {len(candidates)} derived candidates...")

    best = 0
    for seed in candidates:
        if seed < 0:
            continue
        matches = test_seed(seed)
        if matches > best:
            best = matches
            print(f"  seed={seed}: {matches} matches")

    return best


def exhaustive_search_around_14():
    """
    Exhaustively search seeds that produce puzzle 14.
    Then check if any also produce 15.
    """
    print("\n" + "="*70)
    print("EXHAUSTIVE SEARCH FOR PUZZLE 14+15 MATCH")
    print("="*70)

    # Puzzle 14: 0x2930 = 10544
    # Range: [8192, 16383]

    # For a seed to produce 10544 from randint(8192, 16383),
    # the MT state must give a specific output.

    # Let's search a larger range than before
    print("Searching 100M seeds...")

    found = []
    for seed in range(100000000):
        random.seed(seed)
        gen_14 = random.randint(1 << 13, (1 << 14) - 1)

        if gen_14 == PUZZLE_KEYS[14]:
            gen_15 = random.randint(1 << 14, (1 << 15) - 1)
            if gen_15 == PUZZLE_KEYS[15]:
                gen_16 = random.randint(1 << 15, (1 << 16) - 1)
                print(f"  FOUND! seed={seed}")
                print(f"    Puzzle 14: {hex(gen_14)} (expected {hex(PUZZLE_KEYS[14])})")
                print(f"    Puzzle 15: {hex(gen_15)} (expected {hex(PUZZLE_KEYS[15])})")
                print(f"    Puzzle 16: {hex(gen_16)} (expected {hex(PUZZLE_KEYS[16])})")
                if gen_16 == PUZZLE_KEYS[16]:
                    print("    *** THREE MATCHES! ***")
                found.append(seed)

        if seed % 10000000 == 0 and seed > 0:
            print(f"  Searched {seed:,}...")

    if not found:
        print("  No seed found matching both 14 and 15")

    return found


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--unix', action='store_true')
    parser.add_argument('--ms', action='store_true')
    parser.add_argument('--date', action='store_true')
    parser.add_argument('--derived', action='store_true')
    parser.add_argument('--exhaustive', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all or args.derived:
        search_derived_from_known_seeds()

    if args.all or args.date:
        search_date_formats()

    if args.all or args.unix:
        search_unix_timestamps()

    if args.exhaustive:
        exhaustive_search_around_14()

    if not any([args.unix, args.ms, args.date, args.derived, args.exhaustive, args.all]):
        # Quick default: derived and date formats
        search_derived_from_known_seeds()
        search_date_formats()
