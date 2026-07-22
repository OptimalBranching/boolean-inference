"""Deterministic Tseitin encoding for the project's CircuitSAT JSON format."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .circuit import CircuitError, decode_expression, validate_circuit
except ImportError:  # direct script execution
    from circuit import CircuitError, decode_expression, validate_circuit  # type: ignore


Term = bool | int

# Bump this whenever clause generation changes.  Benchmark manifests record the
# value so logically equivalent CNFs are not accidentally treated as the same
# solver workload.
CNF_ENCODING = "circuit-tseitin-output-direct-v2"


@dataclass
class Cnf:
    variable_names: list[str]
    clauses: list[list[int]]

    def lines(self) -> Iterator[str]:
        for index, name in enumerate(self.variable_names, 1):
            yield f"c var {index} {name}\n"
        yield f"p cnf {len(self.variable_names)} {len(self.clauses)}\n"
        for clause in self.clauses:
            yield " ".join(map(str, clause)) + " 0\n"

    def dimacs(self) -> str:
        return "".join(self.lines())

    def write_dimacs(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            stream.writelines(self.lines())


class Encoder:
    def __init__(self, variables: list[str]):
        self.variable_names = list(variables)
        self.ids = {name: index for index, name in enumerate(variables, 1)}
        self.clauses: list[list[int]] = []

    def auxiliary(self, label: str) -> int:
        self.variable_names.append(f"__tseitin_{len(self.variable_names) + 1}_{label}")
        return len(self.variable_names)

    def add_clause(self, literals: list[int]) -> None:
        unique = list(dict.fromkeys(literals))
        if any(-literal in unique for literal in unique):
            return
        self.clauses.append(unique)

    def equivalence(self, output: int, term: Term) -> None:
        if isinstance(term, bool):
            self.add_clause([output if term else -output])
        elif output != term:
            self.add_clause([-output, term])
            self.add_clause([output, -term])

    def negation_equivalence(self, output: int, term: Term) -> None:
        if isinstance(term, bool):
            self.equivalence(output, not term)
        elif output != term:
            self.add_clause([-output, -term])
            self.add_clause([output, term])

    def and_equivalence(self, output: int, terms: list[Term]) -> None:
        if any(term is False for term in terms):
            self.equivalence(output, False)
            return
        live = [term for term in terms if term is not True]
        if not live:
            self.equivalence(output, True)
            return
        if len(live) == 1:
            self.equivalence(output, live[0])
            return
        for term in live:
            assert isinstance(term, int) and not isinstance(term, bool)
            self.add_clause([-output, term])
        self.add_clause([output, *(-term for term in live)])

    def or_equivalence(self, output: int, terms: list[Term]) -> None:
        if any(term is True for term in terms):
            self.equivalence(output, True)
            return
        live = [term for term in terms if term is not False]
        if not live:
            self.equivalence(output, False)
            return
        if len(live) == 1:
            self.equivalence(output, live[0])
            return
        for term in live:
            assert isinstance(term, int) and not isinstance(term, bool)
            self.add_clause([output, -term])
        self.add_clause([-output, *live])

    def xor_pair_equivalence(
        self, output: int, left: int, right: int, *, inverted: bool = False
    ) -> None:
        output_literal = -output if inverted else output
        self.add_clause([-left, -right, -output_literal])
        self.add_clause([left, right, -output_literal])
        self.add_clause([left, -right, output_literal])
        self.add_clause([-left, right, output_literal])

    def xor_equivalence(self, output: int, terms: list[Term]) -> None:
        parity = sum(term is True for term in terms) % 2 == 1
        counts: dict[int, int] = {}
        order = []
        for term in terms:
            if isinstance(term, bool):
                continue
            if term not in counts:
                order.append(term)
            counts[term] = counts.get(term, 0) + 1
        live = [term for term in order if counts[term] % 2]
        if not live:
            self.equivalence(output, parity)
            return
        if len(live) == 1:
            if parity:
                self.negation_equivalence(output, live[0])
            else:
                self.equivalence(output, live[0])
            return

        # Keep DIMACS width at most three.  A k-input parity needs k-2
        # auxiliaries; the assignment output itself is the final XOR gate.
        while len(live) > 2:
            left, right, *rest = live
            intermediate = self.auxiliary("xor")
            self.xor_pair_equivalence(intermediate, left, right)
            live = [intermediate, *rest]
        self.xor_pair_equivalence(output, live[0], live[1], inverted=parity)

    def negate(self, term: Term) -> Term:
        if isinstance(term, bool):
            return not term
        result = self.auxiliary("not")
        self.negation_equivalence(result, term)
        return result

    def and_gate(self, terms: list[Term]) -> Term:
        if any(term is False for term in terms):
            return False
        live = [term for term in terms if term is not True]
        if not live:
            return True
        if len(live) == 1:
            return live[0]
        result = self.auxiliary("and")
        self.and_equivalence(result, live)
        return result

    def or_gate(self, terms: list[Term]) -> Term:
        if any(term is True for term in terms):
            return True
        live = [term for term in terms if term is not False]
        if not live:
            return False
        if len(live) == 1:
            return live[0]
        result = self.auxiliary("or")
        self.or_equivalence(result, live)
        return result

    def xor_pair(self, left: Term, right: Term) -> Term:
        if isinstance(left, bool):
            return self.negate(right) if left else right
        if isinstance(right, bool):
            return self.negate(left) if right else left
        if left == right:
            return False
        result = self.auxiliary("xor")
        self.xor_pair_equivalence(result, left, right)
        return result

    def encode_expression(self, expr: dict[str, Any]) -> Term:
        op, arg = decode_expression(expr)
        if op == "Var":
            try:
                return self.ids[arg]
            except KeyError as exc:
                raise CircuitError(f"unknown variable {arg!r}") from exc
        if op == "Const":
            return arg
        if op == "Not":
            return self.negate(self.encode_expression(arg))
        terms = [self.encode_expression(child) for child in arg]
        if op == "And":
            return self.and_gate(terms)
        if op == "Or":
            return self.or_gate(terms)
        if op == "Xor":
            result: Term = False
            for term in terms:
                result = self.xor_pair(result, term)
            return result
        raise CircuitError(f"unsupported Boolean operation {op!r}")

    def encode_assignment(self, output: int, expr: dict[str, Any]) -> None:
        """Encode one assignment without duplicating its top-level gate.

        CircuitSAT already gives every assignment a named output wire.  The old
        encoder first created an auxiliary for the expression and then equated
        that auxiliary with the output, nearly doubling multiplier CNFs.  Use
        the output as the top-level Tseitin variable and reserve auxiliaries for
        genuinely nested expressions and long XOR chains.
        """
        op, arg = decode_expression(expr)
        if op == "Var":
            try:
                self.equivalence(output, self.ids[arg])
            except KeyError as exc:
                raise CircuitError(f"unknown variable {arg!r}") from exc
            return
        if op == "Const":
            self.equivalence(output, arg)
            return
        if op == "Not":
            self.negation_equivalence(output, self.encode_expression(arg))
            return
        terms = [self.encode_expression(child) for child in arg]
        if op == "And":
            self.and_equivalence(output, terms)
        elif op == "Or":
            self.or_equivalence(output, terms)
        elif op == "Xor":
            self.xor_equivalence(output, terms)
        else:
            raise CircuitError(f"unsupported Boolean operation {op!r}")

    def encode(self, data: dict[str, Any]) -> Cnf:
        for item in data["circuit"]["assignments"]:
            output = self.ids[item["outputs"][0]]
            self.encode_assignment(output, item["expr"])
        return Cnf(self.variable_names, self.clauses)


def encode_circuit(data: dict[str, Any]) -> Cnf:
    validate_circuit(data)
    return encode_validated_circuit(data)


def encode_validated_circuit(data: dict[str, Any]) -> Cnf:
    """Encode a CircuitSAT document already checked by ``validate_circuit``."""
    return Encoder(data["variables"]).encode(data)
