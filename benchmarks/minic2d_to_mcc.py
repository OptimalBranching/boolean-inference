#!/usr/bin/env python3
"""Rewrite MiniC2D-style weighted CNF (`c weights w1+ w1- w2+ w2- ...`) into the
MCC 2021+ format (`c p weight <lit> <w> 0`) that `to_wcn` accepts.

Why this exists (ADDMC suite, survey §7 + §10.4): the ADDMC benchmarks.zip ships
every instance in two weight flavors. The Cachet flavor (`w <var> <p>`) is a
TRAP: for the bayes half `w v 0.05` means the pair (0.05, 0.95), but for the
pseudoweighted half `w v 0.75` means the pair (1.5, 0.5) — ADDMC's nonstandard
doubling. Our DIMACS parser refuses bare `w` lines for exactly this reason
(silent convention mixing corrupts weighted counts). The MiniC2D flavor lists
BOTH polarity weights explicitly, so converting from it involves zero convention
guessing; this script is a mechanical relabeling, verified value-for-value.

The `c weights` line has 2*n_vars values: w(+1) w(-1) w(+2) w(-2) ...
Weights are copied verbatim as decimal strings (RationalWeight parses exactly).
Unit pairs (1, 1) are still emitted — explicit beats implicit for a bank.

Usage:
  minic2d_to_mcc.py <in.cnf> [<out.cnf>]      # stdout if no out
  minic2d_to_mcc.py --dir <src_dir> <dst_dir> # batch (mirrors the tree)
"""
import argparse
import os
import sys


def convert_text(text, path="<input>"):
    lines = text.splitlines()
    # Pass 1: collect the `c weights` values and n_vars (the `c weights` line
    # may appear before OR after the p-line depending on the emitting tool).
    weights = None
    n_vars = None
    for line in lines:
        toks = line.split()
        if toks[:2] == ["c", "weights"]:
            if weights is not None:
                raise ValueError(f"{path}: multiple `c weights` lines")
            weights = toks[2:]
        elif toks[:2] == ["p", "cnf"] and len(toks) >= 4:
            n_vars = int(toks[2])
    if n_vars is None:
        raise ValueError(f"{path}: no `p cnf` line")
    if weights is None:
        raise ValueError(f"{path}: no `c weights` line")
    if len(weights) != 2 * n_vars:
        raise ValueError(f"{path}: `c weights` has {len(weights)} values, "
                         f"expected 2*{n_vars}")
    # Pass 2: emit, dropping the `c weights` line and inserting the explicit
    # MCC pairs right after the p-line.
    out = []
    for line in lines:
        toks = line.split()
        if toks[:2] == ["c", "weights"]:
            continue
        out.append(line)
        if toks[:2] == ["p", "cnf"]:
            for v in range(1, n_vars + 1):
                wpos, wneg = weights[2 * (v - 1)], weights[2 * (v - 1) + 1]
                out.append(f"c p weight {v} {wpos} 0")
                out.append(f"c p weight -{v} {wneg} 0")
    return "\n".join(out) + "\n"


def convert_file(src, dst=None):
    with open(src, errors="ignore") as f:
        text = convert_text(f.read(), src)
    if dst:
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        with open(dst, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


def run_dir(src, dst):
    ok = failed = 0
    for dp, _dn, fns in os.walk(src):
        for fn in fns:
            if not fn.lower().endswith(".cnf"):
                continue
            s = os.path.join(dp, fn)
            d = os.path.join(dst, os.path.relpath(s, src))
            try:
                convert_file(s, d)
                ok += 1
            except ValueError as e:
                failed += 1
                print(f"  FAIL {e}", file=sys.stderr)
    print(f"converted: {ok}, failed: {failed}")
    return 0 if failed == 0 else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", nargs=2, metavar=("SRC", "DST"))
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    args = ap.parse_args()
    if args.dir:
        return run_dir(*args.dir)
    if not args.src:
        ap.error("need <in.cnf> or --dir")
    convert_file(args.src, args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
