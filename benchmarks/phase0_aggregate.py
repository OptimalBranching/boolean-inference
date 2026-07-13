#!/usr/bin/env python3
"""Aggregate phase0_run.py per-instance JSONs into the SIGNAL x SIZE matrix.

Reads <results>/f*_i*.json and reports, per candidate cutoff signal, the median
Spearman rho (vs conquer time, and vs conflicts) across instances at each
factoring size. Also prints a per-size difficulty summary so you can see where
kissat conflicts turn on (below that, the difficulty label is preprocessing
noise and the ranking is not yet trustworthy). The winning signal is the one
with the strongest, most stable correlation across sizes where conflicts>0.

Records-only downstream: this interprets the measurement, it does not re-run it.

Usage:
  phase0_aggregate.py <results_dir> [--out-csv matrix.csv] [--metric time|conflicts]
"""
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conquer_cubes import SIGNAL_COLS  # noqa: E402


def med(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return statistics.median(xs) if xs else float("nan")


def main():
    a = sys.argv
    if len(a) < 2:
        print(__doc__)
        sys.exit(2)
    results_dir = a[1]
    out_csv = a[a.index("--out-csv") + 1] if "--out-csv" in a else None
    metric = a[a.index("--metric") + 1] if "--metric" in a else "time"
    rho_idx = 1 if metric == "time" else 2  # [n, rho_time, rho_confl]

    recs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "f*_i*.json"))):
        with open(path) as f:
            recs.append(json.load(f))
    if not recs:
        print(f"no result JSONs in {results_dir}", file=sys.stderr)
        sys.exit(1)

    sizes = sorted({r["size"] for r in recs})

    # Per-size difficulty summary — is the measurement trustworthy yet?
    print("=== per-size difficulty (is conflicts>0 yet?) ===")
    print(f"  {'size':>5s} {'inst':>5s} {'med_conqd':>10s} {'med_confl':>10s} "
          f"{'med_maxcube_s':>13s} {'timeouts':>9s}")
    for n in sizes:
        rs = [r for r in recs if r["size"] == n]
        print(
            f"  f{n:<4d} {len(rs):5d} "
            f"{med([r['n_conquered'] for r in rs]):10.0f} "
            f"{med([r['total_conflicts'] for r in rs]):10.0f} "
            f"{med([r['max_cube_time_s'] for r in rs]):13.3f} "
            f"{sum(r['timeouts'] for r in rs):9d}"
        )

    # Signal x size matrix of median rho (vs `metric`).
    print(f"\n=== signal x size: median Spearman rho vs {metric} ===")
    head = f"{'signal':16s}" + "".join(f"{('f'+str(n)):>9s}" for n in sizes) + f"{'min|med|':>10s}"
    print(head)
    matrix = {}
    for name in SIGNAL_COLS:
        cells = {}
        for n in sizes:
            rs = [r for r in recs if r["size"] == n]
            cells[n] = med([r["rhos"][name][rho_idx] for r in rs])
        matrix[name] = cells
        finite = [abs(cells[n]) for n in sizes if cells[n] == cells[n]]
        worst = min(finite) if finite else float("nan")
        row = f"{name:16s}" + "".join(
            f"{cells[n]:9.3f}" if cells[n] == cells[n] else f"{'--':>9s}" for n in sizes
        )
        row += f"{worst:10.3f}" if worst == worst else f"{'--':>10s}"
        print(row)

    ranked = [
        (name, min([abs(matrix[name][n]) for n in sizes if matrix[name][n] == matrix[name][n]], default=float("nan")))
        for name in SIGNAL_COLS
    ]
    ranked = [r for r in ranked if r[1] == r[1]]
    if ranked:
        best = max(ranked, key=lambda r: r[1])
        print(f"\n-> most stable across sizes (max of min|median rho|): "
              f"{best[0]} (min|rho|={best[1]:.3f})")
    print("   (trust sizes where med_confl > 0; smaller sizes are preprocessing noise)")

    if out_csv:
        with open(out_csv, "w") as f:
            f.write("signal," + ",".join(f"f{n}" for n in sizes) + ",min_abs_median\n")
            for name in SIGNAL_COLS:
                cells = matrix[name]
                vals = [f"{cells[n]}" for n in sizes]
                finite = [abs(cells[n]) for n in sizes if cells[n] == cells[n]]
                worst = min(finite) if finite else float("nan")
                f.write(",".join([name] + vals + [str(worst)]) + "\n")
        print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
