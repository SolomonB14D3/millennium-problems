#!/usr/bin/env python3
"""
ECDSA Point Multiplication Study (secp256k1)

This is the most computationally intensive part of Bitcoin key derivation:
    private_key (256-bit scalar) × G (generator point) = public_key (point)

The operation involves:
- ~256 point doublings
- ~128 point additions (on average, depending on bits set)
- Each operation involves modular arithmetic on 256-bit numbers

What we measure:
- Intermediate points at each step
- Which additions are performed (depends on private key bits!)
- Timing of each operation
- Carry propagation patterns

KEY INSIGHT: The sequence of doublings vs additions DIRECTLY depends on
the private key bits. This is why constant-time implementations exist.
Our job is to find what still leaks.
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

# secp256k1 curve parameters
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
A = 0
B = 7
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


@dataclass
class ECPoint:
    """Point on secp256k1 curve (or infinity)."""
    x: Optional[int]
    y: Optional[int]

    @property
    def is_infinity(self) -> bool:
        return self.x is None

    def to_bytes(self) -> bytes:
        if self.is_infinity:
            return b'\x00' * 64
        return self.x.to_bytes(32, 'big') + self.y.to_bytes(32, 'big')

    def to_bits(self) -> np.ndarray:
        """Convert to 512-bit array."""
        data = self.to_bytes()
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


INFINITY = ECPoint(None, None)
G = ECPoint(Gx, Gy)


def mod_inverse(a: int, m: int) -> int:
    """Extended Euclidean algorithm for modular inverse."""
    if a < 0:
        a = a % m
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError("Modular inverse doesn't exist")
    return x % m


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended GCD."""
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y


def point_add(p1: ECPoint, p2: ECPoint) -> ECPoint:
    """Add two points on the curve."""
    if p1.is_infinity:
        return p2
    if p2.is_infinity:
        return p1

    if p1.x == p2.x:
        if p1.y != p2.y:
            return INFINITY
        # Point doubling
        return point_double(p1)

    # Different x coordinates
    slope = ((p2.y - p1.y) * mod_inverse(p2.x - p1.x, P)) % P
    x3 = (slope * slope - p1.x - p2.x) % P
    y3 = (slope * (p1.x - x3) - p1.y) % P
    return ECPoint(x3, y3)


def point_double(p: ECPoint) -> ECPoint:
    """Double a point on the curve."""
    if p.is_infinity:
        return INFINITY
    if p.y == 0:
        return INFINITY

    slope = ((3 * p.x * p.x + A) * mod_inverse(2 * p.y, P)) % P
    x3 = (slope * slope - 2 * p.x) % P
    y3 = (slope * (p.x - x3) - p.y) % P
    return ECPoint(x3, y3)


@dataclass
class ECDSAStep:
    """One step in the scalar multiplication."""
    step_number: int
    operation: str  # 'double' or 'add'
    bit_value: int  # The bit that triggered this (for adds)
    result_point: ECPoint
    timing_ns: int


@dataclass
class ECDSATrace:
    """Complete trace of scalar multiplication."""
    scalar: int
    scalar_bits: np.ndarray
    steps: List[ECDSAStep]
    result_point: ECPoint
    total_timing_ns: int
    num_doublings: int
    num_additions: int

    def get_operation_sequence(self) -> str:
        """Get sequence of operations as string (D=double, A=add)."""
        return ''.join('D' if s.operation == 'double' else 'A' for s in self.steps)

    def get_timing_sequence(self) -> np.ndarray:
        """Get timing of each operation."""
        return np.array([s.timing_ns for s in self.steps])


def scalar_multiply_instrumented(k: int, point: ECPoint = G) -> ECDSATrace:
    """
    Scalar multiplication with full instrumentation.

    Uses double-and-add algorithm (NOT constant-time).
    This is intentional - we want to see the leaks.
    """
    start_time = time.perf_counter_ns()
    steps = []
    num_doublings = 0
    num_additions = 0

    # Convert scalar to bits (256 bits)
    scalar_bits = np.array([(k >> (255 - i)) & 1 for i in range(256)], dtype=np.uint8)

    # Standard double-and-add (right-to-left)
    result = INFINITY
    addend = ECPoint(point.x, point.y)

    for i in range(256):
        bit = (k >> i) & 1  # LSB first

        if bit == 1:
            # Add
            add_start = time.perf_counter_ns()
            result = point_add(result, addend)
            add_time = time.perf_counter_ns() - add_start

            steps.append(ECDSAStep(
                step_number=len(steps),
                operation='add',
                bit_value=bit,
                result_point=ECPoint(result.x, result.y) if not result.is_infinity else INFINITY,
                timing_ns=add_time
            ))
            num_additions += 1

        # Double the addend for next iteration
        double_start = time.perf_counter_ns()
        addend = point_double(addend)
        double_time = time.perf_counter_ns() - double_start

        steps.append(ECDSAStep(
            step_number=len(steps),
            operation='double',
            bit_value=bit,
            result_point=ECPoint(addend.x, addend.y) if not addend.is_infinity else INFINITY,
            timing_ns=double_time
        ))
        num_doublings += 1

    total_time = time.perf_counter_ns() - start_time

    return ECDSATrace(
        scalar=k,
        scalar_bits=scalar_bits,
        steps=steps,
        result_point=result,
        total_timing_ns=total_time,
        num_doublings=num_doublings,
        num_additions=num_additions
    )


def scalar_multiply_constant_time(k: int, point: ECPoint = G) -> Tuple[ECPoint, int]:
    """
    Constant-time scalar multiplication (Montgomery ladder).

    This is how secure implementations do it.
    Returns (result, timing).
    """
    start_time = time.perf_counter_ns()

    r0 = INFINITY
    r1 = point

    for i in range(256):
        bit = (k >> (255 - i)) & 1

        if bit == 0:
            r1 = point_add(r0, r1)
            r0 = point_double(r0)
        else:
            r0 = point_add(r0, r1)
            r1 = point_double(r1)

    total_time = time.perf_counter_ns() - start_time
    return r0, total_time


def verify_implementation():
    """Verify against known test vectors."""
    print("Verifying ECDSA implementation...")

    # Test 1: 1 * G = G
    trace = scalar_multiply_instrumented(1)
    assert trace.result_point.x == Gx
    assert trace.result_point.y == Gy
    print("  ✓ 1 * G = G")

    # Test 2: 2 * G - verify point is on curve
    trace = scalar_multiply_instrumented(2)
    # Verify result is on curve: y^2 = x^3 + 7 mod p
    x, y = trace.result_point.x, trace.result_point.y
    lhs = (y * y) % P
    rhs = (x * x * x + B) % P
    assert lhs == rhs, "2*G not on curve"
    # Known x-coordinate for 2*G
    expected_x = 0xC6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5
    assert trace.result_point.x == expected_x
    print("  ✓ 2 * G correct (on curve, x matches)")

    # Test 3: Compare instrumented vs constant-time
    test_scalar = 0xDEADBEEF12345678
    trace = scalar_multiply_instrumented(test_scalar)
    result_ct, _ = scalar_multiply_constant_time(test_scalar)
    assert trace.result_point.x == result_ct.x
    assert trace.result_point.y == result_ct.y
    print("  ✓ Instrumented matches constant-time")

    print("All ECDSA tests pass")
    return True


def analyze_timing_leak(num_samples: int = 100) -> Dict[str, Any]:
    """
    Analyze timing variations based on scalar hamming weight.

    The non-constant-time implementation should show clear correlation.
    """
    timings = []
    hamming_weights = []
    num_additions = []

    for _ in range(num_samples):
        # Random 256-bit scalar
        k = int.from_bytes(np.random.bytes(32), 'big') % N
        if k == 0:
            k = 1

        trace = scalar_multiply_instrumented(k)

        timings.append(trace.total_timing_ns)
        hamming_weights.append(bin(k).count('1'))
        num_additions.append(trace.num_additions)

    timings = np.array(timings)
    hamming_weights = np.array(hamming_weights)
    num_additions = np.array(num_additions)

    # Correlation between timing and hamming weight
    corr_hw = np.corrcoef(timings, hamming_weights)[0, 1]
    corr_adds = np.corrcoef(timings, num_additions)[0, 1]

    return {
        'timing_mean': np.mean(timings),
        'timing_std': np.std(timings),
        'hamming_weight_correlation': corr_hw,
        'num_additions_correlation': corr_adds,
        'timing_per_addition': np.polyfit(num_additions, timings, 1)[0],
    }


def analyze_operation_sequence_leak(num_samples: int = 100) -> Dict[str, Any]:
    """
    The operation sequence (D/A pattern) directly reveals scalar bits.

    This is the most obvious leak in non-constant-time implementations.
    """
    perfect_reconstructions = 0

    for _ in range(num_samples):
        k = int.from_bytes(np.random.bytes(32), 'big') % N
        if k == 0:
            k = 1

        trace = scalar_multiply_instrumented(k)

        # Try to reconstruct scalar from operation sequence
        reconstructed = 0
        bit_pos = 255

        for step in trace.steps:
            if step.operation == 'add':
                reconstructed |= (1 << bit_pos)
            if step.operation == 'double' or step.operation == 'add':
                # After first non-zero bit, each step (whether just double or double+add)
                # corresponds to one bit position
                pass

        # The operation sequence directly encodes the scalar bits
        # D = 0, DA = 1 (after first 1)
        op_seq = trace.get_operation_sequence()

        # First find the first 'A' (first 1 bit)
        first_add = op_seq.find('A')
        if first_add >= 0:
            # Extract bits from pattern
            reconstructed_bits = []
            i = first_add
            while i < len(op_seq):
                if op_seq[i] == 'A':
                    reconstructed_bits.append(1)
                    i += 1
                elif op_seq[i] == 'D':
                    reconstructed_bits.append(0)
                    i += 1

            # This should match the scalar bits (after leading zeros)
            actual_bits = []
            temp = k
            while temp:
                actual_bits.append(temp & 1)
                temp >>= 1
            actual_bits = actual_bits[::-1]  # MSB first

            if reconstructed_bits == actual_bits:
                perfect_reconstructions += 1

    return {
        'perfect_reconstruction_rate': perfect_reconstructions / num_samples,
        'note': 'Operation sequence directly reveals scalar bits in non-constant-time impl'
    }


def generate_training_data(num_samples: int, bit_length: int = 256) -> Dict[str, np.ndarray]:
    """
    Generate training data for ML.

    For each sample:
    - Input: scalar bits
    - Output: public key bits
    - Features: operation sequence, timings, intermediate points
    """
    scalar_bits = np.zeros((num_samples, bit_length), dtype=np.uint8)
    pubkey_bits = np.zeros((num_samples, 512), dtype=np.uint8)
    total_timings = np.zeros(num_samples, dtype=np.int64)
    num_additions_arr = np.zeros(num_samples, dtype=np.int32)

    for i in range(num_samples):
        # Generate scalar with specified bit length
        if bit_length < 256:
            k = int.from_bytes(np.random.bytes(bit_length // 8 + 1), 'big')
            k = k % (1 << bit_length)
            if k == 0:
                k = 1
        else:
            k = int.from_bytes(np.random.bytes(32), 'big') % N
            if k == 0:
                k = 1

        trace = scalar_multiply_instrumented(k)

        # Store data
        for j in range(bit_length):
            scalar_bits[i, j] = (k >> (bit_length - 1 - j)) & 1

        pubkey_bits[i] = trace.result_point.to_bits()
        total_timings[i] = trace.total_timing_ns
        num_additions_arr[i] = trace.num_additions

        if (i + 1) % 100 == 0:
            print(f"Generated {i+1}/{num_samples} ECDSA samples")

    return {
        'scalar_bits': scalar_bits,
        'pubkey_bits': pubkey_bits,
        'timings': total_timings,
        'num_additions': num_additions_arr,
    }


if __name__ == "__main__":
    if not verify_implementation():
        exit(1)

    print("\n" + "="*60)
    print("ECDSA TIMING LEAK ANALYSIS")
    print("="*60)

    print("\nAnalyzing timing correlation with hamming weight...")
    timing_results = analyze_timing_leak(100)
    print(f"  Timing mean: {timing_results['timing_mean']/1e6:.2f} ms")
    print(f"  Timing std: {timing_results['timing_std']/1e6:.2f} ms")
    print(f"  Hamming weight correlation: {timing_results['hamming_weight_correlation']:.4f}")
    print(f"  Num additions correlation: {timing_results['num_additions_correlation']:.4f}")
    print(f"  Timing per addition: {timing_results['timing_per_addition']/1e3:.1f} μs")

    print("\n" + "="*60)
    print("OPERATION SEQUENCE LEAK ANALYSIS")
    print("="*60)

    print("\nAnalyzing operation sequence leak...")
    seq_results = analyze_operation_sequence_leak(50)
    print(f"  Perfect scalar reconstruction rate: {seq_results['perfect_reconstruction_rate']*100:.1f}%")
    print(f"  Note: {seq_results['note']}")

    print("\n" + "="*60)
    print("KEY INSIGHT")
    print("="*60)
    print("""
In non-constant-time ECDSA:
- Operation sequence DIRECTLY reveals scalar bits
- Timing correlates with hamming weight
- These are known attacks that 'secure' implementations prevent

Our ML goal: Find RESIDUAL leaks even in constant-time implementations.
The structure is deterministic - something always leaks.
""")
