#!/usr/bin/env python3
"""Phase 0 per-instance driver (array-job friendly): sample every internal node
of ONE factoring instance, conquer each node's residual with kissat, and record
each candidate cutoff signal's Spearman correlation against realized difficulty.

One array task = one (size, index). Reads `data/factoring_<n>x<n>.jsonl` (10
CircuitSAT instances/line, from gen_factoring.py), extracts instance `index`,
and:
  1. export_dimacs <inst> <n> 0 > inst.cnf   (VE budget 0 => raw numbering that
     lines up with gen_cubes; VE!=0 would renumber and break cube alignment).
  2. gen_cubes <inst> inst.cubes [--max-nodes N]   (exhaustive node sampling).
  3. conquer_and_correlate(inst.cnf, inst.cubes, kissat, cube_timeout).
  4. write one JSON result to <out>/f<size>_i<index>.json with per-signal rhos.

No theta anywhere (the arbitrary cutoff was discarded); this measures which
signal a resource-aware cutoff should key on. Aggregate the JSONs with
phase0_aggregate.py. Records-only: it measures, it does not interpret.

Usage:
  phase0_run.py --size 20 --index 0 --bin target/release/examples \
      --data benchmarks/data --work $SCRATCH --out results \
      --kissat ~/kissat/build/kissat --cube-timeout 60 [--max-nodes N]
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conquer_cubes import conquer_and_correlate, signal_rhos, SIGNAL_COLS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--bin", required=True, help="dir with gen_cubes/export_dimacs")
    ap.add_argument("--data", required=True, help="dir with factoring_<n>x<n>.jsonl")
    ap.add_argument("--work", required=True, help="scratch dir for per-instance files")
    ap.add_argument("--out", required=True, help="results dir for the JSON row")
    ap.add_argument("--kissat", default="kissat")
    ap.add_argument("--cube-timeout", type=float, default=None,
                    help="per-cube kissat timeout (s); censors pathological cubes")
    ap.add_argument("--max-nodes", type=int, default=None,
                    help="cap emitted samples (backstop); default exhaustive")
    args = ap.parse_args()

    n, k = args.size, args.index
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    jsonl = os.path.join(args.data, f"factoring_{n}x{n}.jsonl")
    with open(jsonl) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    if k >= len(lines):
        print(f"index {k} out of range ({len(lines)} instances)", file=sys.stderr)
        sys.exit(2)

    inst_json = os.path.join(args.work, f"inst_{n}_{k}.json")
    with open(inst_json, "w") as f:
        f.write(lines[k])
    cnf = os.path.join(args.work, f"inst_{n}_{k}.cnf")
    cubes = os.path.join(args.work, f"inst_{n}_{k}.cubes")

    export = os.path.join(args.bin, "export_dimacs")
    gen = os.path.join(args.bin, "gen_cubes")

    # 1. raw CNF (VE budget 0): numbering matches gen_cubes.
    with open(cnf, "w") as fout:
        subprocess.run([export, inst_json, str(n), "0"], stdout=fout, check=True)

    # 2. exhaustive node sampling (+ .meta signals).
    cmd = [gen, inst_json, cubes]
    if args.max_nodes is not None:
        cmd += ["--max-nodes", str(args.max_nodes)]
    gp = subprocess.run(cmd, capture_output=True, text=True)
    sample_line = gp.stderr.strip().splitlines()[-1] if gp.stderr.strip() else ""

    # 3. conquer every sampled node + correlate.
    r = conquer_and_correlate(
        cnf, cubes, kissat=args.kissat, cube_timeout=args.cube_timeout
    )
    rhos = signal_rhos(r["signals"], r["times"], r["confs"])

    total_conflicts = sum(r["confs"])
    max_time = max(r["times"]) if r["times"] else 0.0
    rec = {
        "size": n,
        "index": k,
        "nvars": r["nvars"],
        "nclauses": r["nclauses"],
        "n_open_cubes": r["ncubes"],
        "n_conquered": len(r["times"]),
        "refuted": r["refuted"],
        "timeouts": r["timeouts"],
        "total_conflicts": total_conflicts,
        "max_cube_time_s": max_time,
        "sample_line": sample_line,
        # per-signal: [n, rho_vs_time, rho_vs_conflicts]
        "rhos": {name: list(rhos[name]) for name in SIGNAL_COLS},
    }
    out_path = os.path.join(args.out, f"f{n}_i{k}.json")
    with open(out_path, "w") as f:
        json.dump(rec, f)
    # one-line stdout for the slurm log
    print(
        f"f{n} i{k}: {r['ncubes']} samples, {len(r['times'])} conquered, "
        f"{r['refuted']} refuted, {r['timeouts']} timeouts, "
        f"total_conflicts={total_conflicts}, max_cube={max_time:.3f}s -> {out_path}"
    )


if __name__ == "__main__":
    main()
