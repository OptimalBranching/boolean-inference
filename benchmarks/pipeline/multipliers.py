"""Deterministic factoring targets and structure-preserving multipliers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .circuit import (
    assignment,
    const,
    nary,
    sha256_file,
    unary,
    validate_circuit,
    var,
)

def is_prime(value: int) -> bool:
    if value < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small:
        if value % prime == 0:
            return value == prime
    odd = value - 1
    power = 0
    while odd % 2 == 0:
        odd //= 2
        power += 1
    # Deterministic for unsigned 64-bit integers; still a strong fixed-base test above it.
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, odd, value)
        if witness in (1, value - 1):
            continue
        for _ in range(power - 1):
            witness = witness * witness % value
            if witness == value - 1:
                break
        else:
            return False
    return True


def random_prime(rng: random.Random, bits: int) -> int:
    if bits < 2:
        raise ValueError("factor width must be at least 2 bits")
    while True:
        candidate = rng.randrange(1 << (bits - 1), 1 << bits) | 1
        if is_prime(candidate):
            return candidate


def records(widths: list[int], count: int, seed_base: int):
    if count < 1:
        raise ValueError("count must be positive")
    if len(widths) != len(set(widths)):
        raise ValueError("factor widths must be unique")
    for width in widths:
        rng = random.Random(seed_base + width)
        seen: set[int] = set()
        index = 0
        while index < count:
            left = random_prime(rng, width)
            right = random_prime(rng, width)
            target = left * right
            if target in seen:
                continue
            seen.add(target)
            public = {
                "id": f"fact-{width}-{index:04d}",
                "generator": "balanced-semiprime-v1",
                "factor_bits": width,
                "target": target,
                "seed": seed_base + width,
                "sequence_index": index,
            }
            oracle = {**public, "left_factor": left, "right_factor": right}
            yield public, oracle
            index += 1


Signal = bool | str


class Builder:
    def __init__(self):
        self.variables: list[str] = []
        self.known: set[str] = set()
        self.assignments: list[dict[str, Any]] = []
        self.counter = 0

    def variable(self, name: str) -> str:
        if name in self.known:
            raise ValueError(f"duplicate circuit variable {name!r}")
        self.known.add(name)
        self.variables.append(name)
        return name

    def fresh(self, label: str) -> str:
        self.counter += 1
        return self.variable(f"{label}${self.counter}")

    @staticmethod
    def expression(signal: Signal) -> dict[str, Any]:
        return const(signal) if isinstance(signal, bool) else var(signal)

    def gate(self, op: str, signals: list[Signal], label: str) -> Signal:
        if op == "And":
            if any(signal is False for signal in signals):
                return False
            live = [signal for signal in signals if signal is not True]
            if not live:
                return True
        elif op == "Or":
            if any(signal is True for signal in signals):
                return True
            live = [signal for signal in signals if signal is not False]
            if not live:
                return False
        elif op == "Xor":
            parity = sum(signal is True for signal in signals) % 2 == 1
            counts: dict[str, int] = {}
            order = []
            for signal in signals:
                if isinstance(signal, bool):
                    continue
                if signal not in counts:
                    order.append(signal)
                counts[signal] = counts.get(signal, 0) + 1
            live = [signal for signal in order if counts[signal] % 2]
            if parity:
                if not live:
                    return True
                live.append(True)
            elif not live:
                return False
        else:
            raise ValueError(f"unsupported gate {op!r}")
        if len(live) == 1:
            return live[0]
        output = self.fresh(label)
        self.assignments.append(
            assignment(output, nary(op, [self.expression(signal) for signal in live]))
        )
        return output

    def invert(self, signal: Signal, label: str) -> Signal:
        if isinstance(signal, bool):
            return not signal
        output = self.fresh(label)
        self.assignments.append(assignment(output, unary("Not", var(signal))))
        return output

    def assign(self, output: str, signal: Signal) -> None:
        self.assignments.append(assignment(output, self.expression(signal)))

    def add(self, left: list[Signal], right: list[Signal], label: str) -> list[Signal]:
        width = max(len(left), len(right))
        left = [*left, *([False] * (width - len(left)))]
        right = [*right, *([False] * (width - len(right)))]
        carry: Signal = False
        result = []
        for index, (left_bit, right_bit) in enumerate(zip(left, right, strict=True)):
            result.append(
                self.gate("Xor", [left_bit, right_bit, carry], f"{label}:sum[{index}]")
            )
            carry = self.gate(
                "Or",
                [
                    self.gate("And", [left_bit, right_bit], f"{label}:xy[{index}]"),
                    self.gate("And", [left_bit, carry], f"{label}:xc[{index}]"),
                    self.gate("And", [right_bit, carry], f"{label}:yc[{index}]"),
                ],
                f"{label}:carry[{index}]",
            )
        result.append(carry)
        return result

    def subtract(
        self, left: list[Signal], right: list[Signal], label: str
    ) -> list[Signal]:
        """Unsigned ripple subtraction; callers guarantee left >= right."""
        width = max(len(left), len(right))
        left = [*left, *([False] * (width - len(left)))]
        right = [*right, *([False] * (width - len(right)))]
        borrow: Signal = False
        result = []
        for index, (left_bit, right_bit) in enumerate(zip(left, right, strict=True)):
            result.append(
                self.gate(
                    "Xor", [left_bit, right_bit, borrow], f"{label}:diff[{index}]"
                )
            )
            not_left = self.invert(left_bit, f"{label}:not-left[{index}]")
            borrow = self.gate(
                "Or",
                [
                    self.gate("And", [not_left, right_bit], f"{label}:ny[{index}]"),
                    self.gate("And", [not_left, borrow], f"{label}:nb[{index}]"),
                    self.gate("And", [right_bit, borrow], f"{label}:yb[{index}]"),
                ],
                f"{label}:borrow[{index}]",
            )
        return result

    def schoolbook(
        self, left: list[Signal], right: list[Signal], label: str
    ) -> list[Signal]:
        width = len(left) + len(right)
        result: list[Signal] = [False] * width
        for right_index, right_bit in enumerate(right):
            row = [False] * right_index
            row.extend(
                self.gate(
                    "And",
                    [left_bit, right_bit],
                    f"{label}:pp[{left_index},{right_index}]",
                )
                for left_index, left_bit in enumerate(left)
            )
            row.extend([False] * (width - len(row)))
            result = self.add(result, row, f"{label}:row[{right_index}]")[:width]
        return result

    @staticmethod
    def shifted(bits: list[Signal], amount: int, width: int) -> list[Signal]:
        return ([False] * amount + bits + [False] * width)[:width]

    def karatsuba(
        self,
        left: list[Signal],
        right: list[Signal],
        base_case: int,
        label: str,
    ) -> list[Signal]:
        output_width = len(left) + len(right)
        size = max(len(left), len(right))
        left = [*left, *([False] * (size - len(left)))]
        right = [*right, *([False] * (size - len(right)))]
        if size <= base_case:
            return self.schoolbook(left, right, f"{label}:leaf")[:output_width]

        split = size // 2
        left_low, left_high = left[:split], left[split:]
        right_low, right_high = right[:split], right[split:]
        low = self.karatsuba(left_low, right_low, base_case, f"{label}:low")
        high = self.karatsuba(left_high, right_high, base_case, f"{label}:high")
        left_sum = self.add(left_low, left_high, f"{label}:left-sum")
        right_sum = self.add(right_low, right_high, f"{label}:right-sum")
        sum_product = self.karatsuba(
            left_sum, right_sum, base_case, f"{label}:sum-product"
        )
        cross = self.subtract(sum_product, low, f"{label}:minus-low")
        cross = self.subtract(cross, high, f"{label}:minus-high")

        full_width = size * 2
        combined = self.add(
            self.shifted(low, 0, full_width),
            self.shifted(cross, split, full_width),
            f"{label}:combine-cross",
        )[:full_width]
        combined = self.add(
            combined,
            self.shifted(high, split * 2, full_width),
            f"{label}:combine-high",
        )[:full_width]
        return combined[:output_width]


def generate_multiplier(bits: int, architecture: str, base_case: int = 4) -> dict[str, Any]:
    if bits < 1:
        raise ValueError("bits must be positive")
    if architecture == "karatsuba" and base_case < 3:
        raise ValueError("Karatsuba base case must be at least 3 bits")
    builder = Builder()
    left = [builder.variable(f"a[{index}]") for index in range(bits)]
    right = [builder.variable(f"b[{index}]") for index in range(bits)]
    if architecture == "array-ripple":
        product = builder.schoolbook(left, right, "array")
    elif architecture == "karatsuba":
        product = builder.karatsuba(left, right, base_case, "karatsuba")
    else:
        raise ValueError(f"unsupported architecture {architecture!r}")
    outputs = [builder.variable(f"product[{index}]") for index in range(bits * 2)]
    for output, signal in zip(outputs, product, strict=True):
        builder.assign(output, signal)
    result = {
        "variables": builder.variables,
        "circuit": {"assignments": builder.assignments},
        "metadata": {
            "format": "circuitsat-benchmark-v1",
            "generator": "boolean-inference-structural-multiplier-v1",
            "generator_sha256": sha256_file(Path(__file__)),
            "architecture": architecture,
            "factor_bits": bits,
            "karatsuba_base_case": base_case if architecture == "karatsuba" else None,
            "ports": {
                "a": {"direction": "input", "bits": left, "lsb_first": True},
                "b": {"direction": "input", "bits": right, "lsb_first": True},
                "product": {
                    "direction": "output",
                    "bits": outputs,
                    "lsb_first": True,
                },
            },
        },
    }
    validate_circuit(result)
    return result
