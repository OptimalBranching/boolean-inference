#!/usr/bin/env python3
"""Three-arm counting ablation: PerConfig vs BlockMerge vs GammaCover (issue #35).

For each sampled bank instance and each VE budget, run all three region-branch
arms and record models / branching_nodes / visited / time. Two things matter:

  (1) CORRECTNESS GATE — all three arms MUST report the same count on every
      (instance, budget). A mismatch means a broken partition (γ overlap
      double-counts, gap undercounts); it is printed loudly and the row is
      marked, because it invalidates any speed reading.

  (2) ADVANTAGE — the γ arm can only differ from BlockMerge where the search
      actually BRANCHES (branching_nodes > 0). Rows with 0 branch nodes (the
      instance dissolved under VE) carry no signal and are excluded from the
      ratio. The headline is the geometric mean of
      branching_nodes(GammaCover) / branching_nodes(BlockMerge) over the
      informative rows — pre-registered bar (docs/design/gamma-cover.md §0):
      <= 0.8 on the cube-friendly regime.

Budget sweep is the point: too low and VE leaves a huge slow residual (timeout);
too high and VE consumes everything (0 branch nodes, nothing to compare). The
informative regime is in between and is per-family, so we sweep and report which
budgets branched.

Every solver call is wrapped in a timeout; an arm that times out or errors drops
the whole (instance, budget) row (can't compare a partial triple). `.xz`
instances are decompressed transparently (reuses bank_check helpers).

Usage:
  ablation.py --family uai-pr kr2024-mc --sample 20 --budgets 0,8,16 \
              --max-rows 128 --timeout 30 --jobs 4 --out ablation.jsonl
"""
import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank_common as bc  # noqa: E402
from bank_check import run_count, sample_instances  # noqa: E402

ARMS = ["perconfig", "blockmerge", "gammacover"]


def parse_fields(stdout):
    """Pull the count binary's `key=val` output line into a dict."""
    out = {}
    for tok in stdout.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def run_arm(path, budget, max_rows, arm, timeout):
    """-> dict(models, branch, visited, time) or None on error/timeout."""
    try:
        r = run_count(path, [str(budget), str(max_rows), arm], timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    f = parse_fields(r.stdout)
    if "models" not in f:
        return None
    return {
        "models": f.get("models"),
        "branch": int(f.get("branching_nodes", 0)),
        "visited": int(f.get("visited", 0)),
        "fallbacks": int(f.get("gamma_fallbacks", 0)),
        "time": float(f.get("time", "0").rstrip("s")),
    }


def run_cell(args):
    """One (instance, budget) triple across all three arms."""
    path, budget, max_rows, timeout = args
    res = {arm: run_arm(path, budget, max_rows, arm, timeout) for arm in ARMS}
    row = {
        "instance": os.path.basename(path),
        "budget": budget,
        "complete": all(res[a] is not None for a in ARMS),
    }
    if not row["complete"]:
        row["missing"] = [a for a in ARMS if res[a] is None]
        return row
    counts = {res[a]["models"] for a in ARMS}
    row["counts_agree"] = len(counts) == 1
    row["models"] = res["perconfig"]["models"]
    for a in ARMS:
        row[a] = {k: res[a][k] for k in ("branch", "visited", "time", "fallbacks")}
    return row


def geomean(xs):
    xs = [x for x in xs if x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", nargs="+", required=True)
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--budgets", default="0,8,16")
    ap.add_argument("--max-rows", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=None, help="append JSONL rows here")
    args = ap.parse_args()

    count_bin = os.path.join(bc.ROOT, "target", "release", "count")
    if not os.path.exists(count_bin):
        bc.die("count binary not built (cargo build --release)")
    budgets = [int(b) for b in args.budgets.split(",")]

    cells = []
    for fam in args.family:
        fdir = os.path.join(bc.BANK_DIR, fam)
        if not os.path.isdir(fdir):
            print(f"== {fam}: not materialized — skipped ==")
            continue
        files = sample_instances(
            fdir, (".cnf", ".dimacs", ".json", ".csp", ".cnf.xz"),
            args.sample, small_only=True)
        print(f"== {fam}: {len(files)} instances x {len(budgets)} budgets ==")
        for p in files:
            for b in budgets:
                cells.append((p, b, args.max_rows, args.timeout, fam))

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_cell, c[:4]): c for c in cells}
        for fut in as_completed(futs):
            fam = futs[fut][4]
            row = fut.result()
            row["family"] = fam
            rows.append(row)
            if row.get("complete") and not row.get("counts_agree", True):
                print(f"  !!! COUNT MISMATCH {fam}/{row['instance']} "
                      f"budget={row['budget']}: "
                      f"{ {a: row[a] for a in ARMS} }")

    if args.out:
        with open(args.out, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    # ---- summary per family ----
    print("\n==== SUMMARY ====")
    any_mismatch = False
    for fam in args.family:
        frows = [r for r in rows if r["family"] == fam]
        complete = [r for r in frows if r.get("complete")]
        mism = [r for r in complete if not r["counts_agree"]]
        any_mismatch = any_mismatch or bool(mism)
        # informative = both BM and GC branched
        info = [r for r in complete if r["counts_agree"]
                and r["blockmerge"]["branch"] > 0 and r["gammacover"]["branch"] > 0]
        ratios = [r["gammacover"]["branch"] / r["blockmerge"]["branch"] for r in info]
        gm = geomean(ratios)
        wins = sum(1 for x in ratios if x < 1.0)
        ties = sum(1 for x in ratios if x == 1.0)
        losses = sum(1 for x in ratios if x > 1.0)
        fb = sum(r["gammacover"]["fallbacks"] for r in complete)
        print(f"\n{fam}: {len(frows)} rows, {len(complete)} complete, "
              f"{len(mism)} MISMATCH, {len(info)} informative (both branched)")
        if gm is not None:
            print(f"  branch ratio GC/BM  geomean={gm:.3f}  "
                  f"(win<1: {wins}, tie: {ties}, loss>1: {losses})  "
                  f"pre-registered bar <= 0.8 -> "
                  f"{'PASS' if gm <= 0.8 else 'FAIL'}")
        else:
            print("  no informative rows (nothing branched at these budgets) — "
                  "sweep more/lower budgets or larger instances")
        if fb:
            print(f"  gamma_fallbacks total = {fb} (IP guard-outs to BlockMerge)")
    print(f"\ncorrectness: {'OK — all arms agree' if not any_mismatch else 'FAIL — MISMATCHES ABOVE'}")
    return 1 if any_mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
