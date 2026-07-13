#!/usr/bin/env python3
"""Generate the frozen instance bank (benchmarks/instances.json) from a declared
sweep grid.

The bank is not hand-listed: a serious benchmark needs seed sweeps (so results
carry a median + variance, not a lucky single seed) and size ladders (so scaling
is visible). This script defines that grid for the MS1 base families
(factoring / Tseitin / coloring), runs each generator, computes the sha256 of
its artifacts, and writes instances.json. instances_check.py then re-derives and
verifies every hash independently.

Extension: XOR / QCP phase-transition sweeps and additional negative controls
land in issue #32; canonical pinned families (SAT-comp / CSPLib / XCSP3) in #33.
Rerun this script after editing the SWEEP to refresh the manifest, then commit.

Usage:
  instances_gen.py            # write benchmarks/instances.json
  instances_gen.py --dry-run  # print the entry count and family breakdown only
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "benchmarks", "instances.json")

# --- sweep grid -------------------------------------------------------------
# Seed counts: 20 per cell for the statistical families, 10 for grid (structure
# is fixed by shape, only edge charges vary). Size ladders span easy->hard.
FACTORING_SIZES = [16, 20, 24, 28]      # x 10 committed records each  = 40
FACTORING_RECORDS = 10
TSEITIN_REGULAR_V = [60, 70, 80, 90, 100]   # x 20 seeds                = 100
TSEITIN_REGULAR_SEEDS = 20
TSEITIN_GRID_ROWS = [2, 3, 4, 5]            # x 40 cols x 10 seeds       = 40
TSEITIN_GRID_COLS = 40
TSEITIN_GRID_SEEDS = 10
COLORING_N = [40, 60, 80]                   # banded + random, x 20 seeds = 120
COLORING_K = 4
COLORING_W = 4
COLORING_SEEDS = 20

MECHANISMS = {
    "arithmetic-circuit": "dense multiplier circuit; difficulty cliff vs refutation onset misalignment",
    "parity-treewidth": "Tseitin parity; regular-resolution length 2^Theta(tw); bounded-tw solvable, expander hard",
    "low-treewidth": "region grown around a vertex stays local on banded graphs; high on random",
}

# Representative predicted-win entry each Tseitin expander (predicted-loss) points
# at as its negative control target.
TSEITIN_WIN_ANCHOR = "tseitin-grid-r4c40-s1"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_generated(cmd, artifacts):
    """Run a generator cmd with {name} placeholders bound to temp files; return
    {name: sha256}."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = {name: os.path.join(tmp, f"a.{name}") for name in artifacts}
        proc = subprocess.run(cmd.format(**paths), shell=True, cwd=ROOT,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"generator failed ({proc.returncode}): {cmd}\n{proc.stderr}")
        return {name: sha256_file(paths[name]) for name in artifacts}


def build_entries():
    entries = []

    # factoring: hash the committed jsonl once per size; one entry per record.
    for n in FACTORING_SIZES:
        rel = f"benchmarks/data/factoring_{n}x{n}.jsonl"
        digest = sha256_file(os.path.join(ROOT, rel))
        with open(os.path.join(ROOT, rel)) as f:
            targets = [json.loads(l)["source"]["data"]["target"] for l in f]
        for idx in range(FACTORING_RECORDS):
            entries.append({
                "id": f"factoring-{n}x{n}-{idx}",
                "family": "factoring", "mechanism": "arithmetic-circuit",
                "predicted_win": True, "negative_control_of": None, "canonical_source": None,
                "kind": "file", "file": rel, "record_index": idx, "target": targets[idx],
                "sha256": {"file": digest}, "baseline_s": None,
            })

    # Tseitin grid: bounded-treewidth, predicted win (also the neg-control targets).
    for rows in TSEITIN_GRID_ROWS:
        for s in range(1, TSEITIN_GRID_SEEDS + 1):
            cmd = (f"python3 benchmarks/csp/gen_tseitin.py --family grid "
                   f"--rows {rows} --cols {TSEITIN_GRID_COLS} --seed {s} --csp {{csp}} --cnf {{cnf}}")
            entries.append({
                "id": f"tseitin-grid-r{rows}c{TSEITIN_GRID_COLS}-s{s}",
                "family": "tseitin", "mechanism": "parity-treewidth",
                "predicted_win": True, "negative_control_of": None, "canonical_source": None,
                "kind": "generated", "cmd": cmd, "seed": s,
                "sha256": run_generated(cmd, ["cnf", "csp"]), "baseline_s": None,
            })

    # Tseitin expander (3-regular): resolution-hard, predicted loss -> neg control.
    for v in TSEITIN_REGULAR_V:
        for s in range(1, TSEITIN_REGULAR_SEEDS + 1):
            cmd = (f"python3 benchmarks/csp/gen_tseitin.py --family regular "
                   f"--v {v} --d 3 --seed {s} --csp {{csp}} --cnf {{cnf}}")
            entries.append({
                "id": f"tseitin-regular-v{v}-s{s}",
                "family": "tseitin", "mechanism": "parity-treewidth",
                "predicted_win": False, "negative_control_of": TSEITIN_WIN_ANCHOR,
                "canonical_source": None,
                "kind": "generated", "cmd": cmd, "seed": s,
                "sha256": run_generated(cmd, ["cnf", "csp"]), "baseline_s": None,
            })

    # coloring: banded (predicted win) and random (matched neg control, same n+seed).
    for n in COLORING_N:
        for s in range(1, COLORING_SEEDS + 1):
            bcmd = (f"python3 benchmarks/csp/gen_coloring.py --family banded "
                    f"--n {n} --k {COLORING_K} --w {COLORING_W} --seed {s} > {{cnf}}")
            entries.append({
                "id": f"coloring-banded-n{n}k{COLORING_K}w{COLORING_W}-s{s}",
                "family": "coloring", "mechanism": "low-treewidth",
                "predicted_win": True, "negative_control_of": None, "canonical_source": None,
                "kind": "generated", "cmd": bcmd, "seed": s,
                "sha256": run_generated(bcmd, ["cnf"]), "baseline_s": None,
            })
            rcmd = (f"python3 benchmarks/csp/gen_coloring.py --family random "
                    f"--n {n} --k {COLORING_K} --seed {s} > {{cnf}}")
            entries.append({
                "id": f"coloring-random-n{n}k{COLORING_K}-s{s}",
                "family": "coloring", "mechanism": "low-treewidth",
                "predicted_win": False,
                "negative_control_of": f"coloring-banded-n{n}k{COLORING_K}w{COLORING_W}-s{s}",
                "canonical_source": None,
                "kind": "generated", "cmd": rcmd, "seed": s,
                "sha256": run_generated(rcmd, ["cnf"]), "baseline_s": None,
            })

    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print counts only, do not write")
    args = ap.parse_args()

    entries = build_entries()
    breakdown = Counter(e["family"] for e in entries)
    controls = sum(1 for e in entries if e["negative_control_of"])
    print(f"{len(entries)} entries: " + ", ".join(f"{k}={v}" for k, v in sorted(breakdown.items())))
    print(f"  predicted-loss negative controls: {controls}")

    if args.dry_run:
        return 0

    manifest = {
        "version": 1,
        "note": ("Frozen instance bank cited by all MS1-MS3 experiments. GENERATED by "
                 "benchmarks/instances_gen.py from a declared sweep grid (seed sweeps + size "
                 "ladders). Do not hand-edit; edit the SWEEP in instances_gen.py and rerun. "
                 "Verify with instances_check.py. See docs/design/predictive-cnc.md (F2) and "
                 "docs/research/2026-07-13-benchmark-representativeness-survey.md."),
        "mechanisms": MECHANISMS,
        "entries": entries,
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
