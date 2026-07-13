#!/usr/bin/env python3
"""Oracle cutoff frontier via tree DP, and regret of simple online rules.

Any cutoff rule ("stop branching when <signal> crosses <threshold>") selects a
FRONTIER in the branching tree: an antichain where descent stops and each
residual goes to the conquer engine. Given a SUBTREE-COMPLETE `gen_cubes` run —
exhaustive, or depth-capped via `--max-depth` so that every open node is either
fully expanded or flagged `boundary` — and a per-node conquer measurement, the
cheapest frontier within the sampled prefix is exact by a linear-time DP.
Boundary nodes are conquer-only, which is also how any ONLINE cutoff behaves
under the same prefix budget, so the restricted oracle is the fair yardstick at
scales where the full tree is unreachable. (A --max-nodes count cap truncates
mid-subtree and is NOT valid input here.)

The objective must charge the cuber for descending, or it degenerates: with
free expansion the all-terminal frontier costs 0 (the cuber "solves" the
instance alone) and the oracle says nothing. So the end-to-end objective is

    g(v) = min( t(v),  c + sum_{child} g(child) )
           conquer v   expand v (cubing cost) and recurse

with t(v) = measured kissat time on v's residual and c = the cuber's amortized
per-node expansion cost (sampling wall time / open nodes; terminal children are
folded in). The root's g is the ORACLE end-to-end sequential cost; it
upper-bounds every realizable cutoff, so a candidate rule is judged by regret
against it — not by rank correlation (Spearman), which is only a proxy.

Candidate rules are one-parameter first-crossing predicates on the `.meta`
signals. Each is swept over thresholds; a threshold's frontier is
{v : fires(v), no ancestor fires}, its expanded set {v : no ancestor fires,
v not fired}, and its cost c*|expanded| + sum_frontier t. Rules are ranked at
their own best threshold (best-case for the rule — regret is a lower bound).
Conflicts are reported at the same frontier as a spawn-noise-free difficulty
axis; makespan (max cube time, the parallel-CnC horizon) likewise.

Exhaustiveness matters: `gen_cubes` omits refuted/SAT terminals, so a missing
child is a cuber-resolved leaf only when no --max-nodes cap truncated the tree.
Tree reconstruction: cube files are DFS pre-order; a stack recovers parents by
decision-literal prefix, disambiguated by `sigma_all_parent` (sigma_all
strictly increases down the tree) because GreedyMerge rows may overlap.

Usage:
  frontier_dp.py <base.cnf> <cubes> --sampling-time S [--kissat PATH]
                 [--jobs N] [--cube-timeout S] [--cache CSV] [--out-prefix P]
"""
import argparse
import csv
import json
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conquer_cubes import (  # noqa: E402
    base_cnf_text, read_cnf, read_cubes, read_meta, kissat_solve,
)

# (signal, direction): "le" = emit at the first node on the path with
# signal <= B (size-like, shrinks with depth); "ge" = first node with
# signal >= B (depth-like / gamma-exhaustion; theta_proxy = the rejected
# sigma_dec*sigma_all cutoff as an honest historical control).
RULES = [
    ("unfixed_vars", "le"),
    ("active_tensors", "le"),
    ("hard_excess", "le"),
    ("sigma_dec", "ge"),
    ("node_gamma", "ge"),
    ("theta_proxy", "ge"),
]
N_THRESHOLDS = 128


def build_tree(cubes, meta):
    """parent[] and children[][] from DFS pre-order literals + meta sigma."""
    n = len(cubes)
    parent = [-1] * n
    children = [[] for _ in range(n)]
    stack = []
    for i in range(n):
        lits = cubes[i]
        sap = meta[i]["sigma_all_parent"]
        while stack:
            j = stack[-1]
            pl = cubes[j]
            if (
                len(pl) < len(lits)
                and lits[: len(pl)] == pl
                and meta[j]["sigma_all"] == sap
            ):
                break
            stack.pop()
        if stack:
            parent[i] = stack[-1]
            children[stack[-1]].append(i)
        elif i != 0:
            raise SystemExit(f"node {i}: no parent found (not an exhaustive run?)")
        stack.append(i)
    return parent, children


def conquer_all(clauses, nvars, cubes, kissat, jobs, cube_timeout, cache):
    """kissat on every node's residual (parallel). The CSV cache is STREAMED
    (appended as results land, indexed rows) so a killed/timed-out job resumes
    from where it stopped instead of losing the whole pass."""
    n = len(cubes)
    recs = [None] * n
    if cache and os.path.exists(cache):
        with open(cache) as fh:
            reader = csv.DictReader(fh)
            rows = reader if "idx" in (reader.fieldnames or ()) else ()
            for r in rows:
                i = int(r["idx"])
                if 0 <= i < n:
                    recs[i] = (float(r["time_s"]), int(r["decisions"]),
                               int(r["conflicts"]), r["result"])
        print(f"cache: {sum(r is not None for r in recs)}/{n} from {cache}",
              file=sys.stderr)

    todo = [i for i in range(n) if recs[i] is None]
    if not todo:
        return recs

    base_text = base_cnf_text(clauses)
    lock = threading.Lock()
    fh = open(cache, "a") if cache else None
    if fh and fh.tell() == 0:
        fh.write("idx,time_s,decisions,conflicts,result\n")
    done = 0

    def one(i):
        nonlocal done
        r = kissat_solve(clauses, nvars, cubes[i], kissat, cube_timeout,
                         base_text=base_text)
        with lock:
            recs[i] = r
            done += 1
            if fh:
                fh.write(f"{i},{r[0]},{r[1]},{r[2]},{r[3]}\n")
                if done % 500 == 0:
                    fh.flush()
            if done % 2000 == 0:
                print(f"  ...{done}/{len(todo)} conquered", file=sys.stderr)
        return r

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        list(ex.map(one, todo))
    if fh:
        fh.close()
    return recs


def oracle_dp(times, children, c, boundary):
    """g[] of the end-to-end DP, plus the optimal frontier (conquer choices).

    Boundary nodes (depth cap stopped expansion there) have no sampled
    children, so their only move is conquer: g = t. That makes the root's g
    the EXACT optimum over every frontier living inside the sampled prefix —
    the tractable stand-in for the full-tree oracle on large instances."""
    n = len(times)
    g = [0.0] * n
    for v in range(n - 1, -1, -1):  # pre-order: children have larger indices
        if boundary[v]:
            g[v] = times[v]
            continue
        g[v] = min(times[v], c + sum(g[ch] for ch in children[v]))
    frontier, expanded, stack = [], 0, [0]
    while stack:
        v = stack.pop()
        if boundary[v] or times[v] <= c + sum(g[ch] for ch in children[v]):
            frontier.append(v)
        else:
            expanded += 1
            stack.extend(children[v])
    return g, frontier, expanded


def frontier_stats(fr_idx, expanded_n, times, confs, c):
    ft = [times[v] for v in fr_idx]
    fc = [confs[v] for v in fr_idx]
    return {
        "ncubes": len(fr_idx),
        "expanded": expanded_n,
        "total_s": c * expanded_n + sum(ft),
        "makespan_s": max(ft) if ft else 0.0,
        "cubing_s": c * expanded_n,
        "max_conflicts": max(fc) if fc else 0,
        "sum_conflicts": sum(fc) if fc else 0,
    }


def sweep_rule(sig, direction, parent, times, confs, c, boundary):
    """Best threshold by end-to-end total; returns (B, stats) or None.

    A boundary node the rule hasn't fired on by then is conquered anyway (the
    online cuber has to stop somewhere the sample can still price), so rules
    are compared under the same prefix budget as the oracle. NaN signals never
    fire and are skipped in the ancestor fold."""
    n = len(sig)
    anc = [math.inf if direction == "le" else -math.inf] * n
    for v in range(1, n):
        p = parent[v]
        sp = sig[p]
        prev = anc[p]
        if sp != sp:  # NaN: ancestor never fires on this signal
            anc[v] = prev
        else:
            anc[v] = min(prev, sp) if direction == "le" else max(prev, sp)

    finite = sorted({s for s in sig if s == s and not math.isinf(s)})
    if not finite:
        return None
    step = max(1, len(finite) // N_THRESHOLDS)
    grid = finite[::step]
    if grid[-1] != finite[-1]:
        grid.append(finite[-1])  # extreme threshold = "conquer at the root"
    best = None
    for B in grid:
        fr, expanded = [], 0
        for v in range(n):
            quiet = anc[v] > B if direction == "le" else anc[v] < B
            if not quiet:
                continue
            s = sig[v]
            fires = (s == s) and ((s <= B) if direction == "le" else (s >= B))
            if fires or boundary[v]:
                fr.append(v)
            else:
                expanded += 1
        if not fr:
            continue
        st = frontier_stats(fr, expanded, times, confs, c)
        if best is None or st["total_s"] < best[1]["total_s"]:
            best = (B, st)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("cubes")
    ap.add_argument("--sampling-time", type=float, required=True,
                    help="gen_cubes wall seconds (stderr); amortized into c")
    ap.add_argument("--kissat", default="kissat")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--cube-timeout", type=float, default=30.0)
    ap.add_argument("--cache", default=None, help="conquer-record CSV cache")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    clauses, nvars = read_cnf(args.base)
    cubes = read_cubes(args.cubes)
    meta = read_meta(args.cubes + ".meta")
    if meta is None or len(meta) != len(cubes):
        raise SystemExit("need an aligned .meta sidecar (our gen_cubes output)")
    n = len(cubes)
    c = args.sampling_time / n
    print(f"{n} open nodes; base {nvars} vars / {len(clauses)} clauses; "
          f"cubing cost c={c * 1e3:.3f} ms/node")

    parent, children = build_tree(cubes, meta)
    boundary = [m["boundary"] > 0 for m in meta]
    roots = sum(1 for p in parent if p == -1)
    mono = all(meta[v]["sigma_all"] > meta[parent[v]]["sigma_all"] for v in range(1, n))
    print(f"tree: {roots} root(s), sigma_all monotone: {mono}, "
          f"{sum(boundary)} boundary nodes")

    recs = conquer_all(clauses, nvars, cubes, args.kissat, args.jobs,
                       args.cube_timeout, args.cache)
    times = [r[0] for r in recs]
    confs = [r[2] for r in recs]
    n_to = sum(1 for r in recs if r[3] == "timeout")
    n_sat = sum(1 for r in recs if r[3] == "sat")
    print(f"conquered: {n_sat} sat, {n_to} timeouts; "
          f"root kissat {times[0]:.3f}s / {confs[0]} conflicts")

    g, ofr, o_expanded = oracle_dp(times, children, c, boundary)
    ost = frontier_stats(sorted(ofr), o_expanded, times, confs, c)
    hdr = (f"{'':18s} {'total(s)':>9s} {'cubing(s)':>10s} {'makespan(s)':>12s} "
           f"{'#cubes':>7s} {'#expand':>8s} {'maxconf':>8s} {'sumconf':>9s} {'@B':>12s}")
    def row(name, st, B=""):
        return (f"{name:18s} {st['total_s']:9.4g} {st['cubing_s']:10.4g} "
                f"{st['makespan_s']:12.4g} {st['ncubes']:7d} {st['expanded']:8d} "
                f"{st['max_conflicts']:8d} {st['sum_conflicts']:9d} {B!s:>12s}")

    print(f"\nend-to-end sequential objective (conquer + cubing at c):")
    print(hdr)
    no_cubing = frontier_stats([0], 0, times, confs, c)
    print(row("no-cubing", no_cubing))
    print(row("ORACLE", ost))

    summary = {"n_nodes": n, "c_per_node_s": c, "timeouts": n_to, "sat_cubes": n_sat,
               "no_cubing": no_cubing, "oracle": ost, "rules": {}}
    for name, direction in RULES:
        if name == "theta_proxy":
            sig = [m["sigma_dec"] * m["sigma_all"] for m in meta]
        else:
            sig = [m[name] for m in meta]
        best = sweep_rule(sig, direction, parent, times, confs, c, boundary)
        if best is None:
            continue
        B, st = best
        st["regret"] = st["total_s"] / ost["total_s"] if ost["total_s"] else math.nan
        print(row(f"{name} ({direction})", st, B))
        summary["rules"][name] = {"direction": direction, "B": B, **st}

    print("\n(regret = rule total / oracle total, at the rule's own best threshold)")
    for name, r in summary["rules"].items():
        print(f"  {name:16s} regret x{r['regret']:.2f}")

    if args.out_prefix:
        with open(args.out_prefix + ".summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        with open(args.out_prefix + ".nodes.csv", "w") as fh:
            fh.write("idx,parent,depth_lits,time_s,decisions,conflicts,result,"
                     "unfixed_vars,active_tensors,hard_excess,sigma_dec,sigma_all,"
                     "node_gamma,oracle_g\n")
            for i, (dt, dec, cf, res) in enumerate(recs):
                m = meta[i]
                fh.write(f"{i},{parent[i]},{len(cubes[i])},{dt},{dec},{cf},{res},"
                         f"{m['unfixed_vars']},{m['active_tensors']},{m['hard_excess']},"
                         f"{m['sigma_dec']},{m['sigma_all']},{m['node_gamma']},{g[i]}\n")
        print(f"\nwrote {args.out_prefix}.summary.json / .nodes.csv")


if __name__ == "__main__":
    main()
