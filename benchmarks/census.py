#!/usr/bin/env python3
"""Coverage census (paper pillar P3, the honesty statement): classify every
sampled bank instance into a five-way disposition, per family. This IS the
"what this engine is for" table — long-clause CNF is sharpSAT/Ganak turf by
construction; short-arity native tables are ours. The refused fraction is
reported, never hidden (docs/design/hpc-campaign.md P3).

Disposition of one instance (probed at a FIXED budget, default 12):
  refused-clause  — a clause exceeds MAX_CLAUSE_WIDTH (dense-table limit)
  refused-weight  — bare Cachet `w` lines (convention ambiguity; needs convert)
  branches        — counts to completion AND the search branched (>0 nodes):
                    the regime where the γ arm can matter
  dissolves       — counts to completion with 0 branch nodes (VE consumed it)
  timeout         — did not finish under the per-instance timeout
  error           — any other non-zero exit (a real bug; printed)

"branches" + "dissolves" = the SOLVED fraction; "branches" alone = the ablation's
usable denominator. The census also picks, per family, the branchiest budget in
a small sweep, feeding the ablation sweet-spot.

Usage:
  census.py --family uai-pr kr2024-mc --sample 40 --budget 12 --timeout 30 \
            --jobs 4 --out census.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank_common as bc  # noqa: E402
from bank_check import run_count, sample_instances  # noqa: E402

DISPOSITIONS = ["branches", "dissolves", "timeout",
                "refused-clause", "refused-weight", "error"]


def classify(args):
    path, budget, max_rows, timeout = args
    name = os.path.basename(path)
    try:
        r = run_count(path, [str(budget), str(max_rows), "perconfig"],
                      timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"instance": name, "disposition": "timeout"}
    if r.returncode == 0:
        branch = 0
        for tok in r.stdout.split():
            if tok.startswith("branching_nodes="):
                branch = int(tok.split("=", 1)[1])
        return {"instance": name,
                "disposition": "branches" if branch > 0 else "dissolves",
                "branch": branch}
    err = (r.stderr or "").strip()
    if "exceeds the dense-table limit" in err:
        return {"instance": name, "disposition": "refused-clause"}
    if "Cachet" in err:
        return {"instance": name, "disposition": "refused-weight"}
    return {"instance": name, "disposition": "error", "err": err[:120]}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", nargs="+", required=True)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--max-rows", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.exists(os.path.join(bc.ROOT, "target", "release", "count")):
        bc.die("count binary not built (cargo build --release)")

    all_rows = []
    for fam in args.family:
        fdir = os.path.join(bc.BANK_DIR, fam)
        if not os.path.isdir(fdir):
            print(f"== {fam}: not materialized — skipped ==")
            continue
        files = sample_instances(
            fdir, (".cnf", ".dimacs", ".json", ".csp", ".cnf.xz"),
            args.sample, small_only=True)
        cells = [(p, args.budget, args.max_rows, args.timeout) for p in files]
        counts = Counter()
        rows = []
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(classify, c) for c in cells]
            for fut in as_completed(futs):
                row = fut.result()
                row["family"] = fam
                counts[row["disposition"]] += 1
                rows.append(row)
                if row["disposition"] == "error":
                    print(f"  ERROR {fam}/{row['instance']}: {row.get('err')}")
        all_rows += rows
        n = len(files)
        print(f"== {fam}: {n} instances @ budget {args.budget} ==")
        for d in DISPOSITIONS:
            if counts[d]:
                print(f"    {d:16s} {counts[d]:3d}  ({100*counts[d]//max(n,1)}%)")
        solved = counts["branches"] + counts["dissolves"]
        print(f"    -> solved {solved}/{n}, BRANCHES {counts['branches']}/{n} "
              f"(ablation-usable)")

    if args.out:
        with open(args.out, "a") as f:
            for row in all_rows:
                f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
