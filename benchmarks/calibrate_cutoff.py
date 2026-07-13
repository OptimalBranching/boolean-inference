#!/usr/bin/env python3
"""Sampling calibration of the cutoff threshold — the same protocol for BOTH
cubers, so a comparison isolates where-to-split (OB regions vs lookahead).

This is Zaikin's loop (JAIR 77, 2023, Alg. 4-5), the practice behind every
flagship CnC run: for each candidate threshold, cut, sample cubes, solve the
sample with kissat under a cap, and estimate the end-to-end cost

    est_wall = cubing_time + mean_cube_time * n_cubes / cores

picking the threshold that minimizes it. P95/P99/max of the sample are
recorded alongside — the uniformity claim is read off the same table.

Both cubers stop on the SAME measure (remaining free/unfixed variables):
  ours   gen_cubes <inst> out.icnf --cutoff unfixed:B --frontier-only
  march  march_cu <cnf> -o out.icnf -n B
Raw B values are NOT comparable across cubers (different propagation strength
under the counter), so each cuber is calibrated on its own grid and compared
at its own optimum and at matched cube counts.

Usage:
  calibrate_cutoff.py --cnf inst.cnf --cuber ours --inst inst.json \
      --thresholds 4400,4200,4000 [--sample 200] [--cube-timeout 60]
      [--cores 32] [--cubing-cap 900] [--max-cubes 200000]
      [--kissat PATH] [--march PATH] [--gen-cubes PATH] [--out prefix]
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conquer_cubes import base_cnf_text, read_cnf, read_cubes, kissat_solve, pct  # noqa: E402


def run_cuber(args, B, out_icnf):
    """Cut at threshold B; returns (wall_seconds, ncubes) or None on failure."""
    if args.cuber == "ours":
        cmd = [args.gen_cubes, args.inst, out_icnf,
               "--cutoff", f"unfixed:{B}", "--frontier-only",
               "--max-seconds", str(args.cubing_cap)]
    else:
        cmd = [args.march, args.cnf, "-o", out_icnf, "-n", str(B)]
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=args.cubing_cap * 1.5)
    except subprocess.TimeoutExpired:
        print(f"  B={B}: cubing exceeded {args.cubing_cap * 1.5:.0f}s, skipped",
              file=sys.stderr)
        return None
    dt = time.perf_counter() - t0
    if not os.path.exists(out_icnf):
        print(f"  B={B}: no cube file produced, skipped", file=sys.stderr)
        return None
    n = sum(1 for line in open(out_icnf) if line.startswith("a"))
    return dt, n


def sample_conquer(args, clauses, nvars, base_text, cubes, k):
    idx = sorted(random.sample(range(len(cubes)), min(k, len(cubes))))

    def one(i):
        dt, _dec, cf, res = kissat_solve(clauses, nvars, cubes[i], args.kissat,
                                         args.cube_timeout, base_text=base_text)
        return dt, cf, res

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        return list(ex.map(one, idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnf", required=True, help="base CNF (conquer input)")
    ap.add_argument("--cuber", choices=["ours", "march"], required=True)
    ap.add_argument("--inst", help="instance for gen_cubes (.json/.cnf/.csp); ours only")
    ap.add_argument("--thresholds", required=True,
                    help="comma-separated B grid, shallow (large) to deep (small)")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--cube-timeout", type=float, default=60.0)
    ap.add_argument("--cores", type=int, default=32, help="cores assumed in est_wall")
    ap.add_argument("--jobs", type=int, default=8, help="local sampling parallelism")
    ap.add_argument("--cubing-cap", type=float, default=900.0)
    ap.add_argument("--max-cubes", type=int, default=200000)
    ap.add_argument("--kissat", default="kissat")
    ap.add_argument("--march", default="march_cu")
    ap.add_argument("--gen-cubes", default="target/release/examples/gen_cubes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write <out>.calib.json")
    args = ap.parse_args()
    if args.cuber == "ours" and not args.inst:
        ap.error("--inst is required for --cuber ours")
    random.seed(args.seed)

    clauses, nvars = read_cnf(args.cnf)
    base_text = base_cnf_text(clauses)
    grid = [int(x) for x in args.thresholds.split(",")]
    rows = []
    print(f"cuber={args.cuber} sample={args.sample} cap={args.cube_timeout}s "
          f"cores={args.cores}")
    hdr = (f"{'B':>7s} {'cubing(s)':>10s} {'#cubes':>8s} {'mean(s)':>8s} "
           f"{'p95':>7s} {'p99':>7s} {'max':>7s} {'#to':>4s} "
           f"{'est_cpu(h)':>10s} {'est_wall(s)':>11s}")
    print(hdr)
    for B in grid:
        out_icnf = f"{args.out or 'calib'}_{args.cuber}_B{B}.icnf"
        r = run_cuber(args, B, out_icnf)
        if r is None:
            continue
        cubing_s, n = r
        if n == 0:
            print(f"{B:7d} {cubing_s:10.1f} {n:8d}  (no cubes — threshold above root?)")
            continue
        if n > args.max_cubes:
            print(f"{B:7d} {cubing_s:10.1f} {n:8d}  (over --max-cubes, not sampled)")
            rows.append({"B": B, "cubing_s": cubing_s, "ncubes": n, "skipped": True})
            continue
        cubes = read_cubes(out_icnf)
        recs = sample_conquer(args, clauses, nvars, base_text, cubes, args.sample)
        times = sorted(t for t, _c, _r in recs)
        n_to = sum(1 for _t, _c, r in recs if r == "timeout")
        mean = sum(times) / len(times)
        est_cpu = mean * n
        # Wall = cubing + max(work spread over cores, slowest sampled cube):
        # with few cubes the makespan term dominates, not work/cores.
        est_wall = cubing_s + max(est_cpu / args.cores, times[-1])
        print(f"{B:7d} {cubing_s:10.1f} {n:8d} {mean:8.3f} "
              f"{pct(times, 0.95):7.2f} {pct(times, 0.99):7.2f} {times[-1]:7.2f} "
              f"{n_to:4d} {est_cpu / 3600:10.2f} {est_wall:11.1f}")
        rows.append({"B": B, "cubing_s": cubing_s, "ncubes": n, "mean_s": mean,
                     "p95_s": pct(times, 0.95), "p99_s": pct(times, 0.99),
                     "max_s": times[-1], "timeouts": n_to,
                     "est_cpu_s": est_cpu, "est_wall_s": est_wall})

    # Censored levels (any sampled timeout) only LOWER-BOUND the cost — a mean
    # of capped values is fiction — so they are ineligible for "best".
    scored = [r for r in rows if "est_wall_s" in r and r["timeouts"] == 0]
    if scored:
        best = min(scored, key=lambda r: r["est_wall_s"])
        print(f"\nbest (uncensored levels only): B={best['B']} "
              f"est_wall={best['est_wall_s']:.1f}s "
              f"({best['ncubes']} cubes, mean {best['mean_s']:.3f}s, "
              f"p99 {best['p99_s']:.2f}s)")
    else:
        print("\nno uncensored level — extend the grid deeper or raise --cube-timeout")
    if args.out:
        with open(args.out + ".calib.json", "w") as fh:
            json.dump({"cuber": args.cuber, "cnf": args.cnf, "cores": args.cores,
                       "sample": args.sample, "cube_timeout": args.cube_timeout,
                       "rows": rows}, fh, indent=2)
        print(f"wrote {args.out}.calib.json")


if __name__ == "__main__":
    main()
