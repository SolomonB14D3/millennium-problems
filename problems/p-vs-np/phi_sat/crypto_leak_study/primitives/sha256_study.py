#!/usr/bin/env python3
"""
SHA256 Deep Study

Goal: Understand every variable in SHA256, measure everything,
find structure that leaks information about the input.

SHA256 internals:
- 64 rounds of mixing
- Each round has intermediate state (8 x 32-bit words)
- Operations: rotations, shifts, XOR, addition mod 2^32
- Message schedule expansion (16 → 64 words)

What we measure:
- All 64 intermediate states (512 bits each = 32,768 bits total)
- Bit transition counts at each round
- Hamming weights throughout
- Timing variations (if any)
- Carry propagation patterns in additions
"""

import hashlib
import struct
import time
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

# SHA256 constants
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

# Initial hash values
H_INIT = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]


def rotr(x: int, n: int) -> int:
    """Right rotate 32-bit integer."""
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF


def shr(x: int, n: int) -> int:
    """Right shift."""
    return x >> n


def ch(x: int, y: int, z: int) -> int:
    """Choice function."""
    return (x & y) ^ (~x & z) & 0xFFFFFFFF


def maj(x: int, y: int, z: int) -> int:
    """Majority function."""
    return (x & y) ^ (x & z) ^ (y & z)


def sigma0(x: int) -> int:
    """Σ0 function."""
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)


def sigma1(x: int) -> int:
    """Σ1 function."""
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)


def gamma0(x: int) -> int:
    """σ0 function (message schedule)."""
    return rotr(x, 7) ^ rotr(x, 18) ^ shr(x, 3)


def gamma1(x: int) -> int:
    """σ1 function (message schedule)."""
    return rotr(x, 17) ^ rotr(x, 19) ^ shr(x, 10)


@dataclass
class SHA256State:
    """Complete state capture at one round."""
    round: int
    a: int
    b: int
    c: int
    d: int
    e: int
    f: int
    g: int
    h: int
    w: int  # Message schedule word for this round
    t1: int  # Temporary value 1
    t2: int  # Temporary value 2

    def to_array(self) -> np.ndarray:
        """Convert to bit array (256 bits for a-h)."""
        state = [self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h]
        bits = []
        for word in state:
            for i in range(32):
                bits.append((word >> (31 - i)) & 1)
        return np.array(bits, dtype=np.uint8)

    def hamming_weight(self) -> int:
        """Total hamming weight of state."""
        return bin(self.a ^ self.b ^ self.c ^ self.d ^
                   self.e ^ self.f ^ self.g ^ self.h).count('1')


@dataclass
class SHA256Trace:
    """Complete execution trace of SHA256."""
    input_bytes: bytes
    input_bits: np.ndarray
    message_schedule: List[int]  # 64 words
    round_states: List[SHA256State]  # 64 states
    output_bytes: bytes
    output_bits: np.ndarray
    timing_ns: int

    def get_intermediate_matrix(self) -> np.ndarray:
        """Get all intermediate states as bit matrix (64 x 256)."""
        return np.array([s.to_array() for s in self.round_states])

    def get_hamming_trajectory(self) -> np.ndarray:
        """Hamming weight at each round."""
        return np.array([s.hamming_weight() for s in self.round_states])


def sha256_instrumented(data: bytes) -> SHA256Trace:
    """
    SHA256 with full instrumentation.

    Returns complete execution trace including all intermediate states.
    """
    start_time = time.perf_counter_ns()

    # Pad message
    msg = bytearray(data)
    msg_len = len(data)
    msg.append(0x80)
    while (len(msg) + 8) % 64 != 0:
        msg.append(0x00)
    msg += struct.pack('>Q', msg_len * 8)

    # Initialize hash state
    h0, h1, h2, h3, h4, h5, h6, h7 = H_INIT

    all_round_states = []
    all_message_schedule = []

    # Process each 512-bit block
    for block_start in range(0, len(msg), 64):
        block = msg[block_start:block_start + 64]

        # Parse block into 16 32-bit words
        w = list(struct.unpack('>16I', bytes(block)))

        # Extend to 64 words
        for i in range(16, 64):
            w.append((gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16]) & 0xFFFFFFFF)

        all_message_schedule.extend(w)

        # Initialize working variables
        a, b, c, d, e, f, g, h = h0, h1, h2, h3, h4, h5, h6, h7

        # 64 rounds
        for i in range(64):
            t1 = (h + sigma1(e) + ch(e, f, g) + K[i] + w[i]) & 0xFFFFFFFF
            t2 = (sigma0(a) + maj(a, b, c)) & 0xFFFFFFFF

            # Capture state BEFORE update
            state = SHA256State(
                round=i,
                a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h,
                w=w[i], t1=t1, t2=t2
            )
            all_round_states.append(state)

            # Update
            h = g
            g = f
            f = e
            e = (d + t1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (t1 + t2) & 0xFFFFFFFF

        # Add to hash
        h0 = (h0 + a) & 0xFFFFFFFF
        h1 = (h1 + b) & 0xFFFFFFFF
        h2 = (h2 + c) & 0xFFFFFFFF
        h3 = (h3 + d) & 0xFFFFFFFF
        h4 = (h4 + e) & 0xFFFFFFFF
        h5 = (h5 + f) & 0xFFFFFFFF
        h6 = (h6 + g) & 0xFFFFFFFF
        h7 = (h7 + h) & 0xFFFFFFFF

    end_time = time.perf_counter_ns()

    # Final hash
    output = struct.pack('>8I', h0, h1, h2, h3, h4, h5, h6, h7)

    # Convert to bit arrays
    input_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    output_bits = np.unpackbits(np.frombuffer(output, dtype=np.uint8))

    return SHA256Trace(
        input_bytes=data,
        input_bits=input_bits,
        message_schedule=all_message_schedule,
        round_states=all_round_states,
        output_bytes=output,
        output_bits=output_bits,
        timing_ns=end_time - start_time
    )


def verify_implementation():
    """Verify our instrumented SHA256 matches hashlib."""
    test_cases = [
        b"",
        b"abc",
        b"hello world",
        bytes(range(256)),
        b"\x00" * 64,
        b"\xff" * 64,
    ]

    print("Verifying SHA256 implementation...")
    for data in test_cases:
        trace = sha256_instrumented(data)
        expected = hashlib.sha256(data).digest()

        if trace.output_bytes != expected:
            print(f"MISMATCH for input {data[:20]}...")
            print(f"  Got:      {trace.output_bytes.hex()}")
            print(f"  Expected: {expected.hex()}")
            return False

    print("✓ All test cases pass")
    return True


def analyze_bit_effect(bit_position: int, num_samples: int = 1000) -> Dict[str, Any]:
    """
    Analyze the effect of flipping a single input bit.

    Returns statistics on how output bits change when input bit is flipped.
    """
    input_size = 32  # 256 bits = 32 bytes

    output_changes = np.zeros(256, dtype=np.float64)
    timing_diffs = []
    hamming_diffs = []

    for _ in range(num_samples):
        # Random base input
        base = np.random.bytes(input_size)
        base_array = np.array(list(base), dtype=np.uint8)

        # Create flipped version
        byte_idx = bit_position // 8
        bit_idx = 7 - (bit_position % 8)
        flipped_array = base_array.copy()
        flipped_array[byte_idx] ^= (1 << bit_idx)
        flipped = bytes(flipped_array)

        # Run both
        trace_base = sha256_instrumented(base)
        trace_flipped = sha256_instrumented(flipped)

        # Compare outputs
        diff = trace_base.output_bits ^ trace_flipped.output_bits
        output_changes += diff

        # Timing difference
        timing_diffs.append(trace_flipped.timing_ns - trace_base.timing_ns)

        # Hamming trajectory difference
        h_base = trace_base.get_hamming_trajectory()
        h_flip = trace_flipped.get_hamming_trajectory()
        hamming_diffs.append(h_flip - h_base)

    output_changes /= num_samples

    return {
        'bit_position': bit_position,
        'output_flip_probability': output_changes,  # Should be ~0.5 for all if truly random
        'mean_timing_diff_ns': np.mean(timing_diffs),
        'std_timing_diff_ns': np.std(timing_diffs),
        'hamming_trajectory_diff': np.mean(hamming_diffs, axis=0),
        'num_samples': num_samples
    }


def generate_training_data(num_samples: int, input_bits: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate training data for ML.

    Returns:
        inputs: (num_samples, input_bits) - input bit arrays
        outputs: (num_samples, 256) - output bit arrays
        intermediates: (num_samples, 64, 256) - all intermediate states
    """
    input_bytes = input_bits // 8

    inputs = np.zeros((num_samples, input_bits), dtype=np.uint8)
    outputs = np.zeros((num_samples, 256), dtype=np.uint8)
    intermediates = np.zeros((num_samples, 64, 256), dtype=np.uint8)

    for i in range(num_samples):
        data = np.random.bytes(input_bytes)
        trace = sha256_instrumented(data)

        inputs[i] = trace.input_bits[:input_bits]
        outputs[i] = trace.output_bits
        intermediates[i] = trace.get_intermediate_matrix()

        if (i + 1) % 1000 == 0:
            print(f"Generated {i+1}/{num_samples} samples")

    return inputs, outputs, intermediates


if __name__ == "__main__":
    # Verify implementation
    if not verify_implementation():
        exit(1)

    print("\n" + "="*60)
    print("SHA256 VARIABLE EXPLORATION")
    print("="*60)

    # Single trace exploration
    test_input = b"Hello, DAT!"
    trace = sha256_instrumented(test_input)

    print(f"\nInput: {test_input}")
    print(f"Output: {trace.output_bytes.hex()}")
    print(f"Timing: {trace.timing_ns} ns")
    print(f"Message schedule words: {len(trace.message_schedule)}")
    print(f"Round states captured: {len(trace.round_states)}")

    print("\nHamming weight trajectory:")
    trajectory = trace.get_hamming_trajectory()
    print(f"  Min: {trajectory.min()}, Max: {trajectory.max()}, Mean: {trajectory.mean():.1f}")

    # Bit effect analysis for first few bits
    print("\n" + "="*60)
    print("BIT EFFECT ANALYSIS (sampling)")
    print("="*60)

    for bit in [0, 1, 127, 128, 255]:
        print(f"\nAnalyzing bit {bit}...")
        result = analyze_bit_effect(bit, num_samples=100)

        flip_probs = result['output_flip_probability']
        print(f"  Output flip probability: min={flip_probs.min():.3f}, max={flip_probs.max():.3f}, mean={flip_probs.mean():.3f}")
        print(f"  Timing diff: {result['mean_timing_diff_ns']:.1f} ± {result['std_timing_diff_ns']:.1f} ns")

        # Any deviation from 0.5 is potential leak
        deviation = np.abs(flip_probs - 0.5)
        max_dev_bit = np.argmax(deviation)
        print(f"  Max deviation from 0.5: {deviation.max():.4f} at output bit {max_dev_bit}")
