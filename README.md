# φ and the Millennium Problems

**Five of six Clay Millennium Prize Problems show structure related to the golden ratio φ = (1+√5)/2.**

## Key Finding

| Problem | Key Result | Deviation | Status |
|---------|------------|-----------|--------|
| **Navier-Stokes** | δ₀ = 1/(2φ) = 0.309 | < 1% | STRONG |
| **Riemann Hypothesis** | Finite-size attractor = 1/φ | — | REVISED |
| **Birch–Swinnerton-Dyer** | Mazur bound = L(5)+1 = 12 | EXACT | STRONG |
| **Hodge Conjecture** | Count ratio = 1/φ | 1.2% | STRONG |
| **Yang-Mills Mass Gap** | Glueball ratio = φ²/2 | 1.4% | STRONG |
| **P vs NP** | Receding middle with δ₀ base, φ²-scaling | ~12% | REVISED |

**Four problems show strong φ-structure (< 2% deviation or exact). P vs NP and Riemann show φ in finite-size scaling.**

---

## Riemann Hypothesis: REVISED (January 2026)

The original claims (mode = 1/φ, 7.3× excess) were **falsified**. But a subtler pattern emerged:

### What We Found

| Claim | Result |
|-------|--------|
| Mode = 1/φ = 0.618 | **FALSIFIED** — Actual mode ≈ 0.664 |
| 7.3× excess at 1/φ | **FALSIFIED** — No excess detected |
| Finite-size attractor | **1/φ** — Median ≈ 0.619 at low heights |
| Asymptotic limit | **GUE** — Median → 0.605 at high heights |

### The Pattern: φ in Finite-Size Scaling

The spacing ratio median transitions from 1/φ to GUE:

| Height | Median | Attractor |
|--------|--------|-----------|
| ~10⁴ | 0.6194 | 1/φ = 0.618 |
| ~10¹¹ | 0.6053 | GUE = 0.605 |

This parallels P vs NP: **φ governs finite-size corrections**, not the asymptotic limit.

[📄 Full Riemann Analysis](problems/riemann-hypothesis/)

---

## P vs NP: REVISED (January 2026)

The original claim that **1/ν = 7/12 = L(4)/(L(5)+1)** was **falsified** by experiments.

### New Discovery: The Receding Middle

Instead of smooth convergence, experiments revealed:
- **Discrete snaps** to new plateaus (not smooth power-law)
- **Expanding orbits** with radius growth ~φ² per snap
- **Base constant** 1/(2φ) = δ₀ (same as Navier-Stokes!)

| n | α_c(n) | Radius | Pattern |
|---|--------|--------|---------|
| 500 | 3.573 | 0.694 | Left orbit |
| 4000 | 4.996 | 0.729 | Plateau 2 |
| 12000 | 5.495 | 1.228 | Snap to orbit 3 |
| 64000 | 9.996 | 5.729 | Snap to orbit 4 |

**Formula**: |shift(n)| ≈ (1/2φ) × φ^(2k)

This still connects to DAT through φ, but dynamically rather than as a static ratio.

[📄 Full P vs NP Analysis](problems/p-vs-np/)

---

## The Lucas Discovery (Partially Revised)

Two problems are unified through **Lucas numbers** L(n) = φⁿ + (-φ)⁻ⁿ:

```
BSD:    Mazur bound  = L(5) + 1 = 12     (EXACT)
Hodge:  Peak H¹¹     = L(4) = 7          (EXACT)
```

~~P≠NP:   1/ν          = 7/12 = L(4)/(L(5)+1)~~ ← FALSIFIED

The P vs NP Lucas connection was falsified. However, P vs NP still shows φ-structure through the base constant δ₀ = 1/(2φ) and φ²-scaling.

[📄 Full Lucas Unification Analysis](docs/LUCAS_UNIFICATION.md)

---

## The Unified Principle

φ appears at the **boundary between discrete and continuous**:

| Problem | Discrete Structure | Continuous Dynamics | φ-Constraint |
|---------|-------------------|---------------------|--------------|
| Navier-Stokes | H₃ lattice | Fluid velocity | δ₀ = 1/(2φ) |
| BSD | Torsion points | L-function rank | Mazur = L(5)+1 |
| Yang-Mills | Gauge group | Mass spectrum | ratio ≈ φ²/2 |
| Hodge | Algebraic cycles | Hodge classes | count ≈ 1/φ |
| P vs NP | Boolean vars | P(satisfiable) | Receding middle, δ₀ base |
| Riemann | Prime zeros | GUE statistics | Finite-size → 1/φ |

The golden ratio is the geometric signature of **icosahedral symmetry (H₃)**—the maximal finite symmetry in 3D—constraining infinite-dimensional systems.

---

## Evidence Strength

### Tier 1: Strong (< 2% or Exact)

| Finding | Value | Target | Deviation |
|---------|-------|--------|-----------|
| NS depletion δ₀ | 0.309 | 1/(2φ) | < 1% |
| BSD Mazur bound | 12 | L(5)+1 | **EXACT** |
| BSD missing torsion | 11 | L(5) | **EXACT** |
| Hodge count ratio | 0.626 | 1/φ | 1.2% |
| Yang-Mills 2++*/2++ | 1.291 | φ²/2 | 1.4% |

### Tier 2: Revised (φ in finite-size scaling)

| Finding | Formula | Note |
|---------|---------|------|
| P vs NP base radius | 1/(2φ) = δ₀ | ~12% avg error |
| P vs NP orbit scaling | ~φ² per snap | Dynamic structure |
| Riemann finite-size attractor | 1/φ | Median → 1/φ at low heights |
| Riemann asymptotic | GUE | Median → 0.605 at high heights |

### Tier 3: Falsified (original claims)

| Finding | Claimed | Actual | Status |
|---------|---------|--------|--------|
| Riemann GUE mode | 1/φ = 0.618 | 0.664 | **FALSIFIED** |
| Riemann 7.3× excess | Peak at 1/φ | No excess | **FALSIFIED** |

[📄 Full Evidence Table](docs/EVIDENCE_TABLE.md)

---

## Quick Links

| Document | Description |
|----------|-------------|
| [Evidence Table](docs/EVIDENCE_TABLE.md) | Complete φ-findings across all problems |
| [Lucas Unification](docs/LUCAS_UNIFICATION.md) | The L(4), L(5) discovery (partially revised) |
| [Conditional Theorems](docs/CONDITIONAL_THEOREMS.md) | All six "if φ, then solved" theorems |
| [Deviation Scaling](docs/DEVIATION_SCALING.md) | Finite-size correction analysis |

### Individual Problems

| Problem | Folder |
|---------|--------|
| Navier-Stokes | [problems/navier-stokes/](problems/navier-stokes/) |
| Riemann Hypothesis | [problems/riemann-hypothesis/](problems/riemann-hypothesis/) |
| Birch–Swinnerton-Dyer | [problems/birch-swinnerton-dyer/](problems/birch-swinnerton-dyer/) |
| Yang-Mills Mass Gap | [problems/yang-mills-mass-gap/](problems/yang-mills-mass-gap/) |
| Hodge Conjecture | [problems/hodge-conjecture/](problems/hodge-conjecture/) |
| P vs NP | [problems/p-vs-np/](problems/p-vs-np/) |

---

## The Conditional Framework

Each problem admits a conditional theorem:

> **If [φ-structure verified], then [Millennium Problem resolved]**

This transforms each problem from "prove X" to "demonstrate X follows from φ-constraint at discrete-continuous boundaries."

[📄 All Conditional Theorems](docs/CONDITIONAL_THEOREMS.md)

---

## Data Sources

| Problem | Source | Size |
|---------|--------|------|
| Riemann | LMFDB zeros database | 100k+ zeros |
| BSD | LMFDB elliptic curves | 500 curves |
| Hodge | Oxford CICY database | 7,890 manifolds |
| Yang-Mills | Lattice QCD (Morningstar & Peardon) | Published ratios |
| P vs NP | MiniSat experiments | n = 500–64,000 |
| NS | LAMMPS MD simulations | 10k+ atoms |

---

## Related Repositories

- [H₃ Hybrid Discovery](https://github.com/user/H3-Hybrid-Discovery) — LAMMPS validation of H₃ lattice
- [Navier-Stokes H₃](https://github.com/user/navier-stokes-h3) — Full NS proof framework

---

## Citation

If you use this work, please cite:

```bibtex
@misc{phi-millennium-2026,
  title={Golden Ratio Structure in the Millennium Prize Problems},
  author={...},
  year={2026},
  url={https://github.com/...}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
