#!/usr/bin/env python3
"""
Download elliptic curve data from LMFDB for BSD analysis.
"""

import json
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
BASE_URL = "https://www.lmfdb.org/api/ec_curvedata/"

def download_curves(target_count=50000, output_file="lmfdb_curves_large.json"):
    """Download curves from LMFDB API with pagination."""

    all_curves = []
    offset = 0
    batch_size = 100  # API limit per request

    print(f"Downloading up to {target_count} curves from LMFDB...")

    while len(all_curves) < target_count:
        url = f"{BASE_URL}?_format=json&_fields=lmfdb_label,rank,torsion,torsion_structure,conductor&_count={batch_size}&_offset={offset}"

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode())

            curves = data.get('data', [])
            if not curves:
                print(f"No more curves at offset {offset}")
                break

            all_curves.extend(curves)
            offset += len(curves)

            if len(all_curves) % 1000 == 0 or len(all_curves) < 500:
                print(f"  Downloaded {len(all_curves)} curves...")

            # Rate limiting
            time.sleep(0.1)

        except Exception as e:
            print(f"Error at offset {offset}: {e}")
            break

    # Save to file
    output_path = DATA_DIR / output_file
    with open(output_path, 'w') as f:
        json.dump(all_curves, f)

    print(f"\nSaved {len(all_curves)} curves to {output_path}")
    return all_curves

def quick_stats(curves):
    """Print quick statistics."""
    print("\n" + "=" * 50)
    print("QUICK STATISTICS")
    print("=" * 50)

    # Torsion distribution
    torsion_counts = {}
    for c in curves:
        t = c.get('torsion', 0)
        torsion_counts[t] = torsion_counts.get(t, 0) + 1

    print("\nTorsion distribution:")
    for t in sorted(torsion_counts.keys()):
        print(f"  Torsion {t:2d}: {torsion_counts[t]:6d} curves")

    # Check Mazur's theorem
    print("\nMazur's theorem check:")
    allowed = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}
    observed = set(torsion_counts.keys())

    print(f"  Allowed torsion orders: {sorted(allowed)}")
    print(f"  Observed torsion orders: {sorted(observed)}")

    violations = observed - allowed
    if violations:
        print(f"  ⚠ VIOLATIONS: {violations}")
    else:
        print(f"  ✓ All observed torsion orders are allowed by Mazur's theorem")

    missing = allowed - observed
    if missing:
        print(f"  Missing from sample: {sorted(missing)}")

    # Check for 11
    if 11 in observed:
        print(f"  ⚠ FOUND TORSION 11: This would violate Mazur's theorem!")
    else:
        print(f"  ✓ No curves with torsion 11 (as expected)")

if __name__ == "__main__":
    # Download curves
    curves = download_curves(target_count=50000)

    # Show stats
    quick_stats(curves)
