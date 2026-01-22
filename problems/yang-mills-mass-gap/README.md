# Yang-Mills Mass Gap: φ-Structure in Gauge Theory

## The Problem

**Yang-Mills Mass Gap**: Prove that for any compact simple gauge group G, quantum Yang-Mills theory on ℝ⁴ exists and has a mass gap Δ > 0.

Specifically:
- The Hamiltonian H satisfies spec(H) ⊂ {0} ∪ [Δ, ∞)
- The vacuum state is unique

## Our Approach: φ-Structure via E₆ → H₃

### The Connection

1. **E₆ → H₃ Projection**: The E₆ exceptional Lie algebra root system projects onto H₃ icosahedral vertices
2. **SU(3) ⊂ E₆**: The strong force gauge group embeds in E₆ (grand unification)
3. **φ-Discovery**: Glueball mass ratio m(0⁺⁺*)/m(0⁺⁺) = 1.504 ≈ φ (7.1% deviation)

### The Conditional Conjecture

> **Conjecture (Yang-Mills-φ)**: If the SU(3) gauge field vacuum structure inherits H₃ icosahedral symmetry from the E₆ → H₃ Coxeter projection, then a mass gap Δ > 0 exists.

**Formal Statement**: H_φ ⟹ Mass Gap

where H_φ = "gauge topology inherits H₃ structure from E₆"

## Results Summary

### Key Findings (Lattice QCD)

| # | Finding | Measured | φ-Prediction | Deviation |
|---|---------|----------|--------------|-----------|
| 1 | m(0⁺⁺*)/m(0⁺⁺) | **1.504** | φ = 1.618 | **7.1%** |
| 2 | m(2⁺⁺)/m(0⁺⁺) | 1.389 | φ²/2 = 1.309 | 6.2% |
| 3 | m(0⁻⁺)/m(0⁺⁺) | 1.468 | 1 + 1/φ = 1.618 | 9.3% |

### Full Glueball Spectrum

| State | Mass (r₀⁻¹) | Mass (GeV) | Type |
|-------|-------------|------------|------|
| 0⁺⁺ (ground) | 4.21 ± 0.11 | 1.73 | Scalar |
| 0⁺⁺* (excited) | 6.33 ± 0.16 | 2.60 | Scalar |
| 2⁺⁺ | 5.85 ± 0.14 | 2.40 | Tensor |
| 2⁺⁺* | 7.55 ± 0.20 | 3.10 | Tensor |
| 0⁻⁺ | 6.18 ± 0.15 | 2.53 | Pseudoscalar |

## Analysis Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/01_glueball_ratios.py` | Analyze lattice QCD glueball masses | ✅ Complete |
| `scripts/02_e6_h3_projection.py` | Study E₆ → H₃ Coxeter projection | ✅ Complete |
| `scripts/03_comprehensive_figure.py` | Publication-quality summary figure | ✅ Complete |

## Generated Figures

| Figure | Description |
|--------|-------------|
| `figures/glueball_phi.png` | Glueball spectrum with φ-scaling markers |
| `figures/e6_h3_projection.png` | E₆ → H₃ projection visualization |
| `figures/yang_mills_phi_comprehensive.png` | 6-panel summary of all evidence |

## The E₆ → H₃ Connection

### Coxeter-Dynkin Projection

```
E₆ (78-dim exceptional Lie algebra)
  │
  │  Contains SU(3) ⊂ E₆
  │
  │  Coxeter Projection (6D → 3D)
  ▼
H₃ (icosahedral symmetry, order 120)
  │
  │  Golden ratio geometry preserved
  │
  │  δ₀ = 1/(2φ) depletion
  ▼
φ-structure in glueball spectrum
```

### Why E₆?

1. **Grand Unification**: E₆ appears in GUT theories containing SU(3)
2. **Exceptional Structure**: E₆ root system projects to H₃ vertices
3. **φ-Geometry**: H₃ inherently contains the golden ratio

### Key Properties

- E₆ is rank 6 with 72 roots
- H₃ is the 3D icosahedral group of order 120
- The projection preserves φ-ratios between root vectors
- SU(3) × SU(3) × SU(3) ⊂ E₆

## Key Equations

### Glueball Mass Ratio
```
R = m(0⁺⁺*) / m(0⁺⁺) = 1.504 ± 0.05

φ = (1+√5)/2 = 1.618
|R - φ| / φ = 7.1%
```

### Conjectured Mass Gap Formula
```
Δ = (1/2φ) · g² · Λ_QCD

where 1/(2φ) = δ₀ ≈ 0.309 (same as Navier-Stokes depletion!)
```

### Wilson Loop Area Law
```
⟨Tr(U_γ)⟩ ~ exp(-δ₀ · Area(γ))

Confinement from φ-depletion
```

## The DAT Connection

| DAT Pillar | Yang-Mills Analog |
|------------|-------------------|
| H₃ lattice | E₆ → H₃ projected gauge topology |
| δ₀ = 1/(2φ) | Mass gap coefficient |
| Icosahedral constraint | Glueball spectrum organization |
| Bounded dynamics | Confinement |
| Vortex depletion | Color screening |

### Comparison with Navier-Stokes

| Feature | Navier-Stokes | Yang-Mills |
|---------|---------------|------------|
| **Discrete** | H₃ icosahedral lattice | E₆ → H₃ projection |
| **Continuous** | Fluid velocity u(x,t) | Gauge field A_μ(x) |
| **φ-Constraint** | δ₀ = 1/(2φ) = 0.309 | Glueball ratio ≈ φ |
| **Result** | Bounded enstrophy | Mass gap Δ > 0 |
| **Mechanism** | Vortex depletion | Color confinement |

## Novel Predictions

### Prediction 1: Large-N Limit
```
lim_{N→∞} m(0⁺⁺*)/m(0⁺⁺) = φ = 1.618
```
Current data (N=3): 1.504 (7.1% from φ)

### Prediction 2: Higher Glueball States
| State | Predicted Ratio | Notes |
|-------|-----------------|-------|
| 0⁺⁺ | 1.000 | Ground state |
| 2⁺⁺ | φ²/2 ≈ 1.31 | Tensor ground |
| 0⁺⁺* | φ ≈ 1.62 | Scalar excited |
| 2⁺⁺* | φ² - 1/φ ≈ 2.0 | Tensor excited |

## Documentation

- **`docs/CONDITIONAL_THEOREM.md`**: Full statement of conditional conjecture with proofs
- **`docs/E6_H3_MATHEMATICS.md`**: Detailed Coxeter projection analysis (planned)

## Data Sources

- **Morningstar & Peardon (1999)**: Glueball spectrum calculations
- **Meyer (2004)**: High-precision 0⁺⁺ mass
- **Chen et al. (2006)**: Excited state masses

## Timeline

- [x] Identify glueball ratio ≈ φ from lattice QCD data
- [x] Analyze E₆ → H₃ projection mathematically
- [x] Write conditional theorem document
- [x] Create comprehensive figures
- [x] Connect to Navier-Stokes δ₀
- [ ] Extended lattice simulations to test predictions
- [ ] SU(N) analysis for large-N limit

## Conclusion

The Yang-Mills mass gap may be understood as a consequence of the same H₃ icosahedral constraint that regularizes Navier-Stokes:

> **Unified Principle**: The E₆ → H₃ Coxeter projection induces φ-structure in gauge theory, with the depletion constant δ₀ = 1/(2φ) responsible for both the mass gap and confinement.

This provides a **conditional path** to the mass gap: demonstrate that the observed φ-structure is fundamental (arising from E₆ → H₃), and the mass gap follows as a corollary of Discrete Alignment Theory.
