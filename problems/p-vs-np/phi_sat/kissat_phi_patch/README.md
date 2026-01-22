# φ-Guided Kissat Patch

Adds phase transition awareness to Kissat's restart policy.

## What it does

1. Tracks residual α = remaining_clauses / remaining_variables during search
2. Compares to predicted α_c(n) for phase transition
3. Triggers early restarts when approaching the hard region
4. Enables early SAT termination when α << α_c

## Files

- `phi.h` - Phase transition prediction
- `phi.c` - Implementation
- `kissat_phi.patch` - Unified diff to apply to Kissat

## Building

```bash
# Clone Kissat
git clone https://github.com/arminbiere/kissat
cd kissat

# Apply patch
patch -p1 < ../kissat_phi.patch

# Build
./configure && make
```

## Usage

```bash
# Enable φ-guided restarts
./kissat --phi=1 instance.cnf

# Disable (default Kissat behavior)
./kissat --phi=0 instance.cnf
```

## Expected Results

On random 3-SAT instances:
- 10-30% fewer conflicts on easy instances
- Faster convergence near phase transition
- Early termination for trivially SAT instances
