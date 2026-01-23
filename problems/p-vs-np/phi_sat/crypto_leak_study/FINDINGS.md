# Bitcoin Puzzle Cryptographic Leak Study - Key Findings

## Summary

Analysis of the Bitcoin puzzle keys (1-40 solved) to find exploitable patterns.

## Confirmed Facts

### 1. RNG Discovery (Puzzles 1-13)
- **Puzzles 1-8**: Generated with Python `random.randint()`, seed `34378104`
- **Puzzles 9-11**: Seed `78372297`
- **Puzzles 12-13**: Seed `2408880`
- **Verification**: 100% match confirmed

### 2. Puzzles 14+ Use Different Method
- Searched 10M seeds: No seed matches both puzzle 14 AND 15
- Many seeds match puzzle 14 alone (by chance)
- **Conclusion**: Different generation method after puzzle 13

## Structural Analysis

### 3. Variance Comparison
| Segment | Ratio Std Dev | Interpretation |
|---------|---------------|----------------|
| RNG (1-13) | 0.592 | Higher variance |
| Unknown (14+) | **0.388** | Lower variance |
| Simulated Hardened HD | 0.569 | Similar to RNG |

**Key insight**: Real puzzles (14+) are MORE STRUCTURED than:
- The RNG segment (1-13)
- Simulated hardened HD wallet

### 4. CRT Consistency Test
- Real puzzles: ALL equations **inconsistent**
- Simulated HD: ALL equations **inconsistent**

**Conclusion**: CRT inconsistency doesn't distinguish between methods.
Real puzzles are NOT simple HD wallet (constant delta).

### 5. Cross-Puzzle Bit Correlations
Found negative correlations between bit positions:
- Bit 1 vs Bit 5: **-0.318**
- Bit 2 vs Bit 7: **-0.449**

This suggests some bits are anti-correlated across puzzles.

## ML Pattern Hunting

### 6. SHA256 Output-Only Learning (16-bit inputs)
- Tree models achieve **70.9% accuracy** (vs 50% random)
- Neural networks achieve only **~50%** (fail to find pattern)
- **Conclusion**: Pattern is NON-LINEAR, trees find it

### 7. Transfer Learning (16→32 bits)
- Direct model transfer: **0% efficiency** (fails completely)
- Probe-based transfer: Experiments running...

### 8. Generalization on Puzzle Keys
- Train on puzzles 1-30, test on 31-40
- Position prediction: Model predicts constant mean (no pattern)
- Bit prediction: ~50% accuracy (random)
- **Conclusion**: No simple learnable pattern from puzzle number

## Creator's Statement Analysis

> "There is no pattern. It is just consecutive keys from a deterministic wallet
> (masked with leading 000...0001 to set difficulty)"

### What This Means:
1. Keys come from consecutive indices in a deterministic wallet
2. Full keys are 256 bits, masked to puzzle difficulty
3. Creator believed masking destroyed exploitability

### What We Found:
1. Keys 14+ have **less variance** than expected - more structure preserved
2. Not simple HD (non-hardened) - CRT inconsistencies prove this
3. Could be hardened HD or custom scheme
4. The mask (mod 2^N) might preserve more than intended

## Implications for Puzzle 66+

### What We Know:
- Puzzle 66 key is in range [2^65, 2^66 - 1]
- That's ~3.7 × 10^19 possible values
- Lower variance suggests keys cluster (but where?)

### What Would Help:
1. Solved puzzles 41-65 (more data points)
2. Knowledge of exact deterministic scheme
3. Side-channel information

### Current Prediction Confidence:
**LOW** - Models don't generalize, no exploitable pattern found yet.

## Experiments In Progress

1. **Probing Transfer**: Using 16-bit trained models to probe 32-bit system
2. **Feature Importance**: SHAP-like analysis on tree models
3. **Autoencoder**: Learning bit reconstruction patterns

## Code Organization

```
crypto_leak_study/
├── primitives/
│   ├── sha256_study.py      # Instrumented SHA256
│   └── ecdsa_study.py       # Instrumented ECDSA
├── features/
│   ├── output_only_power.py # ML from hash output alone
│   ├── model_transfer.py    # Transfer learning experiments
│   ├── correlation_map.py   # Output-input correlations
│   ├── differential_analysis.py # Δinput → Δoutput
│   ├── mask_analysis.py     # Puzzle mask interaction
│   ├── hd_wallet_leak.py    # HD wallet hypothesis testing
│   ├── puzzle_structure_deep.py # Deep structure analysis
│   ├── advanced_ml.py       # SHAP, autoencoder, control
│   ├── puzzle_transformer.py # Transformer for puzzles
│   └── probing_transfer.py  # Probe-based transfer
└── FINDINGS.md              # This file
```

## Next Steps

1. Wait for running experiments to complete
2. Analyze feature importance results
3. Try different probe strategies if current fails
4. Consider time-based seeds for puzzles 14+ (timestamp hypothesis)
