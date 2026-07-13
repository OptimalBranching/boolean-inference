#!/usr/bin/env python3
"""Production offline CnC conquer: solve every cube of an iCNF in parallel.

Standard two-phase semantics: the cube file is complete (a frontier covering
the whole space), workers pull cubes in file order, and on a SAT instance the
solve ENDS at the first SAT cube (--stop-on-sat); queued cubes are skipped,
in-flight ones drain. Reports wall time to first SAT, total wall, CPU sum, and
the per-cube distribution of completed conquers.

Usage:
  conquer_parallel.py <base.cnf> <cubes.icnf> --jobs 64 [--kissat PATH]
                      [--cube-timeout S] [--stop-on-sat] [--out csv]
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conquer_cubes import base_cnf_text, read_cnf, read_cubes, kissat_solve, pct  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("cubes")
    ap.add_argument("--kissat", default="kissat")
    ap.add_argument("--jobs", type=int, required=True)
    ap.add_argument("--cube-timeout", type=float, default=None)
    ap.add_argument("--stop-on-sat", action="store_true")
    ap.add_argument("--out", default=None, help="per-cube CSV")
    args = ap.parse_args()

    clauses, nvars = read_cnf(args.base)
    cubes = read_cubes(args.cubes)
    base_text = base_cnf_text(clauses)
    print(f"{len(cubes)} cubes; base {nvars} vars / {len(clauses)} clauses; "
          f"{args.jobs} workers", flush=True)

    stop = threading.Event()
    lock = threading.Lock()
    done = 0
    t0 = time.perf_counter()

    def one(i):
        nonlocal done
        if stop.is_set():
            return (i, 0.0, 0, 0, "skipped")
        dt, dec, cf, res = kissat_solve(
            clauses, nvars, cubes[i], args.kissat, args.cube_timeout,
            base_text=base_text)
        with lock:
            done += 1
            if done % 500 == 0:
                print(f"  ...{done} conquered ({time.perf_counter()-t0:.0f}s)",
                      file=sys.stderr, flush=True)
        return (i, dt, dec, cf, res)

    results = []
    sat_wall = None
    sat_idx = None
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(one, i) for i in range(len(cubes))]
        for fut in as_completed(futs):
            rec = fut.result()
            results.append(rec)
            if rec[4] == "sat" and sat_wall is None:
                sat_wall = time.perf_counter() - t0
                sat_idx = rec[0]
                if args.stop_on_sat:
                    stop.set()
    wall = time.perf_counter() - t0

    solved = [r for r in results if r[4] in ("sat", "unsat", "timeout")]
    times = sorted(r[1] for r in solved)
    n_to = sum(1 for r in solved if r[4] == "timeout")
    summary = {
        "ncubes": len(cubes),
        "solved": len(solved),
        "skipped": len(results) - len(solved),
        "timeouts": n_to,
        "sat_cube": sat_idx,
        "wall_to_sat_s": sat_wall,
        "wall_total_s": wall,
        "cpu_sum_s": sum(times),
        "cube_max_s": times[-1] if times else 0.0,
        "cube_p50_s": pct(times, 0.5),
        "cube_p95_s": pct(times, 0.95),
        "cube_p99_s": pct(times, 0.99),
    }
    print(json.dumps(summary, indent=2), flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("idx,time_s,decisions,conflicts,result\n")
            for i, dt, dec, cf, res in sorted(results):
                fh.write(f"{i},{dt},{dec},{cf},{res}\n")


if __name__ == "__main__":
    main()
