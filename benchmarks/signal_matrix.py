#!/usr/bin/env python3
"""Phase 0 orchestration: build the SIGNAL x INSTANCE-FAMILY correlation matrix
that picks a resource-aware cutoff signal (see `docs/research/cutoff_plan.md`).

For each instance in a manifest, cube it with our `gen_cubes` (writing a `.meta`
sidecar of candidate signals), conquer every cube with kissat, and correlate
each signal against realized per-cube difficulty. Aggregate the per-signal
Spearman rho by family, so the winning signal is the one that predicts difficulty
strongly AND stably across families — not one tuned to a single family.

There is no cutoff here: `gen_cubes` samples EVERY internal node (the arbitrary
theta heuristic was discarded). This script does not evaluate a cutoff, it
measures which signal a resource-aware cutoff should key on. Runs should be
recorded via runscribe.

Manifest (JSON):
  {
    "gen_cubes": "target/release/examples/gen_cubes",   # optional; sensible default
    "kissat":    "cnc-tools/bin/kissat",                 # optional
    "max_nodes": null,                                   # optional sample cap per instance
    "limit":     null,                                   # optional conquer cap per instance
    "entries": [
      {"family": "factoring", "cube_input": "x.json", "base_cnf": "x.cnf"},
      {"family": "xor",       "cube_input": "y.csp",  "base_cnf": "y.cnf", "max_nodes": 500}
    ]
  }

Usage:
  signal_matrix.py <manifest.json> [--out-csv matrix.csv] [--tmp-dir DIR]
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conquer_cubes import SIGNAL_COLS, conquer_and_correlate, signal_rhos  # noqa: E402


def mean(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


def run_instance(entry, gen_cubes, kissat, default_max_nodes, limit, tmp_dir):
    """Sample + conquer one instance; return signal -> rho_vs_time (or None on skip)."""
    cube_input = entry["cube_input"]
    base_cnf = entry["base_cnf"]
    max_nodes = entry.get("max_nodes", default_max_nodes)
    for p in (cube_input, base_cnf):
        if not os.path.exists(p):
            print(f"  skip (missing {p})", file=sys.stderr)
            return None

    stem = os.path.splitext(os.path.basename(cube_input))[0]
    cubes_path = os.path.join(tmp_dir, f"{stem}.cubes")
    cmd = [gen_cubes, cube_input, cubes_path]
    if max_nodes is not None:
        cmd += ["--max-nodes", str(max_nodes)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(cubes_path):
        print(f"  gen_cubes failed on {cube_input}: {proc.stderr.strip()}", file=sys.stderr)
        return None

    r = conquer_and_correlate(base_cnf, cubes_path, kissat=kissat, limit=limit)
    if len(r["times"]) <= 2:
        print(f"  {stem}: too few conquered cubes ({len(r['times'])})", file=sys.stderr)
        return None
    rhos = signal_rhos(r["signals"], r["times"], r["confs"])
    print(f"  {stem}: {len(r['times'])} cubes conquered, {r['refuted']} refuted")
    return {name: rhos[name][1] for name in SIGNAL_COLS}  # rho vs time


def main():
    a = sys.argv
    if len(a) < 2:
        print(__doc__)
        sys.exit(2)
    manifest = json.load(open(a[1]))
    out_csv = a[a.index("--out-csv") + 1] if "--out-csv" in a else None
    tmp_dir = a[a.index("--tmp-dir") + 1] if "--tmp-dir" in a else None

    gen_cubes = manifest.get("gen_cubes", "target/release/examples/gen_cubes")
    kissat = manifest.get("kissat", "kissat")
    default_max_nodes = manifest.get("max_nodes")
    limit = manifest.get("limit")

    made_tmp = tmp_dir is None
    if made_tmp:
        tmp_dir = tempfile.mkdtemp(prefix="signal_matrix_")
    os.makedirs(tmp_dir, exist_ok=True)

    # family -> list of {signal: rho} across its instances
    by_family = {}
    for entry in manifest["entries"]:
        fam = entry.get("family", "default")
        print(f"[{fam}] {entry['cube_input']}")
        res = run_instance(
            entry, gen_cubes, kissat, default_max_nodes, limit, tmp_dir
        )
        if res is not None:
            by_family.setdefault(fam, []).append(res)

    if not by_family:
        print("no usable instances", file=sys.stderr)
        sys.exit(1)

    families = sorted(by_family)
    # family-averaged rho per signal
    matrix = {
        name: {fam: mean([r[name] for r in runs]) for fam, runs in by_family.items()}
        for name in SIGNAL_COLS
    }

    # Print family x signal matrix (rho vs time).
    print("\n=== signal x family Spearman rho (vs conquer time), family-averaged ===")
    head = f"{'signal':16s}" + "".join(f"{f[:10]:>11s}" for f in families) + f"{'min|rho|':>10s}"
    print(head)
    ranked = []
    for name in SIGNAL_COLS:
        cells = matrix[name]
        finite = [abs(cells[f]) for f in families if cells[f] == cells[f]]
        worst = min(finite) if finite else float("nan")
        ranked.append((name, worst))
        row = f"{name:16s}" + "".join(
            f"{cells[f]:11.3f}" if cells[f] == cells[f] else f"{'--':>11s}"
            for f in families
        )
        row += f"{worst:10.3f}" if worst == worst else f"{'--':>10s}"
        print(row)

    # The cutoff candidate: strongest WORST-CASE correlation across families.
    ranked = [r for r in ranked if r[1] == r[1]]
    if ranked:
        best = max(ranked, key=lambda r: r[1])
        print(
            f"\n-> most stable across families (max of min|rho|): "
            f"{best[0]} (min|rho|={best[1]:.3f})"
        )

    if out_csv:
        with open(out_csv, "w") as f:
            f.write("signal," + ",".join(families) + ",min_abs_rho\n")
            for name in SIGNAL_COLS:
                cells = matrix[name]
                vals = [f"{cells[f]}" for f in families]
                finite = [abs(cells[f]) for f in families if cells[f] == cells[f]]
                worst = min(finite) if finite else float("nan")
                f.write(",".join([name] + vals + [str(worst)]) + "\n")
        print(f"\nwrote {out_csv}")

    if made_tmp:
        print(f"(cube/meta artifacts under {tmp_dir})")


if __name__ == "__main__":
    main()
