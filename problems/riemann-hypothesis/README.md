# Riemann Hypothesis: φ-Structure Claims FALSIFIED

## Status: FALSIFIED (January 2026)

The original claims of φ-structure in zeta zero spacings were **falsified** by analysis of 100,000 Odlyzko zeros.

---

## The Problem

**Riemann Hypothesis**: All non-trivial zeros of ζ(s) lie on the critical line Re(s) = 1/2.

## Original Claims (All Falsified)

| Claim | Expected | Actual | Status |
|-------|----------|--------|--------|
| GUE spacing mode = 1/φ | 0.618 | 0.664 | **FALSIFIED** (7.4% off) |
| 7.3× excess at 1/φ | Strong peak | No excess | **FALSIFIED** |
| Min spacing ≈ 1/φ² | 0.382 | Not verified | **FALSIFIED** |
| Median ≈ 1/φ | 0.618 | 0.6195 (low), 0.605 (high) | **COINCIDENTAL** |

---

## What Analysis Showed

### Data Sources

| Dataset | Zeros | Height Range | Source |
|---------|-------|--------------|--------|
| zeros1 | 100,000 | 14 to 74,920 | Odlyzko |
| zeros3 | 10,000 | ~10¹² | Odlyzko |

### Mode Analysis

The spacing ratio distribution mode is **not** at 1/φ:

```
Claimed:  mode = 1/φ = 0.6180
Actual:   mode ≈ 0.664
Deviation: 7.4%
```

### Excess at 1/φ

The claimed 7.3× excess at 1/φ was **not detected**:

```
Claimed:  7.3× density excess at r = 1/φ
Actual:   No statistically significant excess
Result:   FALSIFIED
```

### Median Analysis

The median at low heights is near 1/φ, but this is **coincidental**:

| Height | Median | vs 1/φ | Interpretation |
|--------|--------|--------|----------------|
| ~10⁴ | 0.6195 | +0.24% | Within finite-N GUE range (0.615–0.622) |
| ~10¹² | 0.6049 | **-2.1%** | Converges to GUE universality |

The median **escapes** from 1/φ at high heights, proving there is no "binding" to the golden ratio.

---

## Why the Connection Failed

### 1. Finite-N Coincidence

At low heights, the spacing ratio median falls in the range 0.615–0.622 due to finite-size GUE effects. The value 1/φ = 0.618 happens to be in this range, but so are many other values (0.619, 5/8 = 0.625, etc.).

### 2. Escape to Universality

At height ~10¹², the median drops to 0.6049, converging to infinite-N GUE universality (~0.605–0.610). The "connection" to 1/φ evaporates.

### 3. No Persistent Structure

Unlike P vs NP where φ²-scaling **strengthens** with problem size, Riemann's "φ-connection" **weakens** and disappears with increasing height.

---

## Comparison with P vs NP

| Aspect | P vs NP | Riemann |
|--------|---------|---------|
| Original claim | 1/ν = 7/12 | mode = 1/φ |
| Status | FALSIFIED | FALSIFIED |
| Salvaged? | YES — receding middle with δ₀, φ² scaling | NO |
| φ-structure | **Persists/strengthens** with n | **Evaporates** at high heights |
| Conclusion | Genuine dynamic φ-structure | No φ-structure (pure GUE) |

P vs NP was salvaged because the data revealed a deeper φ-connection (base constant δ₀ = 1/(2φ), scaling by φ²). Riemann showed no such salvageable structure — the spacing statistics are pure GUE.

---

## The Actual Statistics

Zeta zeros follow **GUE random matrix universality**:

- **Level repulsion**: Zeros repel each other (P(s) ~ s² as s → 0)
- **Universal statistics**: Spacing distribution follows GUE prediction
- **No φ-structure**: The golden ratio does not appear in any verified way

This is well-established in random matrix theory (Montgomery-Odlyzko law) and requires no φ-hypothesis.

---

## Oscillations Are Noise

The observed oscillations in running median are **not** a coherent signal:

| Evidence | Finding |
|----------|---------|
| Power spectrum | Flat — no dominant frequency |
| Autocorrelation | Drops quickly to zero |
| Half-period distribution | Exponential — random crossings |

The "orbital" appearance is simply statistical noise smoothed by GUE correlations.

---

## Conclusion

**The Riemann Hypothesis has no verified connection to the golden ratio.**

The spacing ratio statistics follow GUE random matrix universality. The proximity of the median to 1/φ at low heights is a finite-size coincidence that disappears at high heights.

Unlike P vs NP (where falsifying the original claim led to discovering a deeper φ-structure), Riemann shows no salvageable φ-connection. The hypothesis space has been thoroughly explored:

- Mode: Not at 1/φ
- Excess at 1/φ: None detected
- Median: Finite-N coincidence, escapes at high heights
- Oscillations: Noise, not signal

**Status: NO EVIDENCE FOR φ-STRUCTURE**

---

## Scripts (Historical)

These scripts implemented the original (falsified) analysis:

| Script | Purpose | Result |
|--------|---------|--------|
| `scripts/01_gue_mode_analysis.py` | Compute GUE mode | Mode ≠ 1/φ |
| `scripts/02_zero_spacing_analysis.py` | Spacing distribution | GUE, not φ |
| `scripts/03_phi_excess_detection.py` | Test for excess at 1/φ | No excess |
| `scripts/04_comprehensive_figure.py` | Summary figure | Claims falsified |

---

## Data Sources

- **Odlyzko zeros**: 100,000 zeros at low height (zeros1), 10,000 at ~10¹² (zeros3)
- **Analysis**: Local unfolding, spacing ratios, KDE, frequency analysis
- **Conclusion**: Pure GUE universality, no φ-structure
