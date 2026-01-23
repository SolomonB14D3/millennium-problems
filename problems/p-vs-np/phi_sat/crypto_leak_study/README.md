# Cryptographic Leak Study

## Philosophy

**There is no randomness in deterministic programs — only structure we haven't found yet.**

This project studies the cryptographic pipeline (ECDSA → SHA256 → RIPEMD160) to find
structure that ML can detect but mathematical analysis has missed.

## DAT Approach

1. **Phase 1: Understand the Variables**
   - Map all inputs, outputs, intermediate states
   - Measure everything: timing, memory access patterns, bit transitions
   - Generate unlimited training data (we control inputs)

2. **Phase 2: Discover Which Variables Leak**
   - Train ML to predict input bits from output + measurements
   - Start simple (1-bit effects), build to complex (n-bit interactions)
   - Look for φ-structure at boundaries (DAT signature)

3. **Phase 3: Exploit the Leak**
   - Any correlation better than random is a crack
   - Compound small advantages across the pipeline
   - Test on Bitcoin puzzles as validation

## Components

```
crypto_leak_study/
├── primitives/           # Individual crypto functions
│   ├── sha256_study.py   # SHA256 in isolation
│   ├── ecdsa_study.py    # ECDSA point multiplication
│   └── ripemd160_study.py
├── pipeline/             # Full key→address pipeline
│   └── bitcoin_pipeline.py
├── features/             # Feature extraction
│   ├── timing.py         # Timing measurements
│   ├── bit_effects.py    # Bit propagation analysis
│   └── intermediate.py   # Intermediate state capture
├── ml/                   # Machine learning
│   ├── level_trainer.py  # Level-by-level training
│   └── leak_detector.py  # Pattern detection
├── data/                 # Generated datasets
└── experiments/          # Experiment scripts
```

## The Pipeline We're Studying

```
Private Key (256-bit integer)
    ↓
    ↓  ECDSA point multiplication on secp256k1
    ↓  k * G where G is generator point
    ↓
Public Key (512-bit: x,y coordinates)
    ↓
    ↓  SHA256
    ↓
Hash (256-bit)
    ↓
    ↓  RIPEMD160
    ↓
Address Hash (160-bit)
    ↓
    ↓  Base58Check encoding
    ↓
Bitcoin Address (string)
```

## Key Insight

We're not trying to "break" these primitives mathematically.
We're asking: **Can ML see patterns that humans formalized away?**

The avalanche effect is high sensitivity, not true randomness.
High sensitivity means small causes → large effects.
But the effects are DETERMINISTIC — structure exists.
