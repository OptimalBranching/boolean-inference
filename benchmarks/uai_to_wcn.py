#!/usr/bin/env python3
"""Convert UAI `.uai` MARKOV instances to the canonical wcn-1 JSON (see
src/instance.rs). The one genuinely new converter the survey calls for.

WHY PYTHON (not src/bin/uai_to_wcn.rs): every other instance generator/converter
under benchmarks/ is Python, and wcn-1 is trivial JSON to emit — a Rust binary
would only duplicate the serde schema. The multi-valued-domain SKIP rule and
.uai.evid clamping read more clearly here. Schema conformance is NOT taken on
faith: converted files are validated by the Rust loader (`count --parse-only`),
which runs `Instance::validate`. Weights are preserved as EXACT decimal strings
(RationalWeight::parse handles them without f64 rounding); scientific-notation
potentials are expanded to an exact `num/den` string so no precision is lost.

.uai MARKOV format (whitespace-delimited token stream after the keyword):
    MARKOV
    <N>                       number of variables
    <card_0> ... <card_{N-1}> cardinalities
    <M>                       number of factors
    <arity> <v..>  x M        factor scopes (0-indexed vars)
    then per factor: <count> followed by <count> non-negative reals, ROW-MAJOR
    with the FIRST scope variable MOST significant (last var fastest-changing).

wcn-1 mapping (BINARY domains only): a factor over scope [v0..v_{k-1}] becomes a
tensor with vars = the scope in the same order and `rows` = every table entry as
[config, weight]. wcn config bit i = value of vars[i] (vars[0] = LSB), whereas
the .uai entry index has vars[0] as the MSB — so the config is the k-bit reversal
of the entry index. Multi-valued instances (any cardinality != 2) are SKIPPED
whole (survey gap #2: do NOT log-encode).

Evidence `.uai.evid`: `<count>` then `(var value)` pairs on one line; each pair
clamps a variable, realised as a unary `allow` tensor pinning that var.

Usage:
  uai_to_wcn.py <in.uai> [<out.json>]         # one instance (stdout if no out)
  uai_to_wcn.py --evid <in.uai.evid> <in.uai> <out.json>
  uai_to_wcn.py --dir <src_dir> <dst_dir>     # batch; prints binary/skipped tally
"""
import argparse
import json
import os
import sys
from decimal import Decimal


class MultiValuedSkip(Exception):
    """Raised for an instance with a non-binary cardinality (skipped, noted)."""


def _tokens(text):
    return text.split()


def parse_uai(text):
    """Return (n_vars, cards, factors) where factors = list of (scope, table);
    `table` is the raw list of weight STRINGS in .uai row-major order."""
    toks = _tokens(text)
    if not toks:
        raise ValueError("empty .uai file")
    i = 0
    net = toks[i]; i += 1
    if net not in ("MARKOV", "BAYES"):
        raise ValueError(f"expected MARKOV/BAYES preamble, got {net!r}")
    n = int(toks[i]); i += 1
    cards = [int(toks[i + j]) for j in range(n)]; i += n
    m = int(toks[i]); i += 1
    scopes = []
    for _ in range(m):
        arity = int(toks[i]); i += 1
        scope = [int(toks[i + j]) for j in range(arity)]; i += arity
        scopes.append(scope)
    factors = []
    for scope in scopes:
        cnt = int(toks[i]); i += 1
        table = toks[i:i + cnt]; i += cnt
        if len(table) != cnt:
            raise ValueError("truncated factor table")
        factors.append((scope, table))
    return n, cards, factors


def parse_evid(text):
    """Parse a .uai.evid line into {var: value}. Empty / `0` means no evidence."""
    toks = _tokens(text)
    if not toks:
        return {}
    cnt = int(toks[0])
    ev = {}
    for j in range(cnt):
        var = int(toks[1 + 2 * j])
        val = int(toks[2 + 2 * j])
        ev[var] = val
    return ev


def _bit_reverse(t, k):
    """Reverse the low k bits of t: maps a .uai entry index (vars[0] MSB) to a
    wcn config (vars[0] LSB)."""
    c = 0
    for b in range(k):
        if (t >> b) & 1:
            c |= 1 << (k - 1 - b)
    return c


def _exact_weight_string(tok):
    """Preserve the potential EXACTLY as a RationalWeight-parseable string.
    Plain integers/decimals/ratios pass through verbatim; scientific notation is
    expanded to `num/den` via Decimal so no f64 rounding occurs."""
    s = tok.strip()
    if "/" in s:
        return s
    if "e" in s.lower():
        num, den = Decimal(s).as_integer_ratio()
        return f"{num}/{den}" if den != 1 else str(num)
    return s  # integer or decimal — RationalWeight::parse reads it exactly


def to_wcn(n_vars, cards, factors, evidence=None):
    """Build the wcn-1 dict. Raises MultiValuedSkip on any non-binary domain."""
    if any(c != 2 for c in cards):
        bad = sorted({c for c in cards if c != 2})
        raise MultiValuedSkip(f"non-binary cardinalities {bad}")
    tensors = []
    for scope, table in factors:
        k = len(scope)
        if k == 0:
            continue  # constant factor: a global scalar, folds into Z uniformly
        if k > 63:
            raise MultiValuedSkip(f"factor arity {k} exceeds 63")
        rows = []
        for t, w in enumerate(table):
            rows.append([_bit_reverse(t, k), _exact_weight_string(w)])
        rows.sort(key=lambda r: r[0])
        tensors.append({"vars": list(scope), "rows": rows})
    if evidence:
        for var, val in sorted(evidence.items()):
            if var >= n_vars:
                raise ValueError(f"evidence var {var} >= n_vars {n_vars}")
            if val not in (0, 1):
                raise MultiValuedSkip(f"evidence value {val} not binary")
            # Clamp: unary allow-tensor keeping only configs with var == val.
            tensors.append({"vars": [var], "allow": [val]})
    return {
        "format": "wcn-1",
        "n_vars": n_vars,
        "tensors": tensors,
        "meta": {"family": "uai-pr", "source": "uai", "query": "PR"},
    }


def convert_file(uai_path, evid_path=None):
    with open(uai_path) as f:
        n, cards, factors = parse_uai(f.read())
    evidence = None
    if evid_path and os.path.exists(evid_path):
        with open(evid_path) as f:
            evidence = parse_evid(f.read())
    return to_wcn(n, cards, factors, evidence)


def _default_evid(uai_path):
    cand = uai_path + ".evid"
    return cand if os.path.exists(cand) else None


def run_dir(src, dst):
    os.makedirs(dst, exist_ok=True)
    binary = skipped = errors = 0
    skip_notes = []
    for dp, _dn, fns in os.walk(src):
        for fn in fns:
            if not fn.endswith(".uai") or fn.endswith(".uai.evid"):
                continue
            uai_path = os.path.join(dp, fn)
            rel = os.path.relpath(uai_path, src)
            out = os.path.join(dst, rel[:-4] + ".json")
            try:
                inst = convert_file(uai_path, _default_evid(uai_path))
            except MultiValuedSkip as e:
                skipped += 1
                skip_notes.append(f"{rel}: {e}")
                continue
            except Exception as e:  # noqa: BLE001 - report and continue the batch
                errors += 1
                skip_notes.append(f"{rel}: ERROR {e}")
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                json.dump(inst, f, separators=(",", ":"))
                f.write("\n")
            binary += 1
    print(f"converted (binary-domain): {binary}")
    print(f"skipped (multi-valued):    {skipped}")
    print(f"errors:                    {errors}")
    for note in skip_notes[:40]:
        print(f"  - {note}")
    if len(skip_notes) > 40:
        print(f"  … and {len(skip_notes) - 40} more")
    return {"binary": binary, "skipped": skipped, "errors": errors}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", nargs=2, metavar=("SRC", "DST"),
                    help="batch-convert a directory tree; print binary/skipped tally")
    ap.add_argument("--evid", metavar="EVID", help="evidence file (.uai.evid)")
    ap.add_argument("uai", nargs="?", help="input .uai")
    ap.add_argument("out", nargs="?", help="output .json (stdout if omitted)")
    args = ap.parse_args()

    if args.dir:
        run_dir(*args.dir)
        return 0
    if not args.uai:
        ap.error("need <in.uai> or --dir")
    evid = args.evid or _default_evid(args.uai)
    try:
        inst = convert_file(args.uai, evid)
    except MultiValuedSkip as e:
        print(f"SKIP (multi-valued): {e}", file=sys.stderr)
        return 3
    text = json.dumps(inst, indent=2) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
