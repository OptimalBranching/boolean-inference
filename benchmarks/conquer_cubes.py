#!/usr/bin/env python3
"""Conquer a cube file with kissat and report the per-cube difficulty distribution.

Given a base DIMACS CNF and a cube file in march_cu's iCNF `a <lits> 0` format
(what both `gen_cubes` and `march_cu -o` produce, over the SAME variable
numbering), solve each cube = base CNF + its literals as unit clauses with
kissat, timing each. Reports the distribution whose LOW VARIANCE is the paper's
cube-uniformity claim, plus the product sigma_dec*sigma_all as one candidate
signal (sigma_dec = cube length; sigma_all = fixed vars after unit propagation).

If a `<cubes>.meta` sidecar exists (written by our `gen_cubes`, one row per
sampled node in cube-file order), its candidate CUTOFF SIGNALS are read too and
each is correlated (Spearman) against realized difficulty — this is the Phase 0
measurement that picks which signal a resource-aware cutoff should use, now that
the arbitrary theta cutoff is discarded (see `docs/research/cutoff_plan.md`).
Cubers without a sidecar (march_cu, Proofix) still work; their signal columns are
simply absent.

Usage:
  conquer_cubes.py <base.cnf> <cubes> [--out csv] [--limit N] [--kissat PATH]
                                      [--stop-on-sat]
"""
import math
import os
import subprocess
import sys
import time

# Candidate cutoff signals carried by the `.meta` sidecar, plus two derived on
# the fly. Order fixes the report/CSV column order.
META_COLS = [
    "sigma_dec",
    "sigma_all",
    "n_lits",
    "unfixed_vars",
    "active_tensors",
    "hard_excess",
    "sigma_all_parent",
    "parent_gamma",
    "node_gamma",
]
# Signals correlated against difficulty (meta-derived + the harness BCP proxy).
SIGNAL_COLS = [
    "proxy_sd_sa",     # sigma_dec * sigma_all, sized by the harness's own BCP
    "unfixed_vars",
    "active_tensors",
    "hard_excess",
    "full_yield",      # sigma_all / sigma_dec  (avg fixed per decision)
    "last_yield",      # sigma_all - sigma_all_parent (fixed by the last branch)
    "parent_gamma",
    "node_gamma",
]


def read_meta(path):
    """Read a `<cubes>.meta` sidecar into a list of dicts (cube-file order), or
    None if absent. Lines starting with '#' are headers/comments. The trailing
    `boundary` flag (depth-capped, unexpanded node) is optional — older meta
    files lack the column and get boundary=0."""
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # idx + the META_COLS fields (+ optional boundary).
            if len(parts) < 1 + len(META_COLS):
                continue
            vals = [float(x) for x in parts[1 : 1 + len(META_COLS)]]
            row = dict(zip(META_COLS, vals))
            row["boundary"] = (
                float(parts[1 + len(META_COLS)])
                if len(parts) > 1 + len(META_COLS)
                else 0.0
            )
            rows.append(row)
    return rows


def signals_from_meta(m):
    """Derive the correlated SIGNAL_COLS from one meta row (NaN-safe)."""
    sd, sa = m["sigma_dec"], m["sigma_all"]
    return {
        "unfixed_vars": m["unfixed_vars"],
        "active_tensors": m["active_tensors"],
        "hard_excess": m["hard_excess"],
        "full_yield": (sa / sd) if sd else float("nan"),
        "last_yield": sa - m["sigma_all_parent"],
        "parent_gamma": m["parent_gamma"],
        "node_gamma": m["node_gamma"],
    }


def read_cnf(path):
    clauses, nvars = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in "cp":
                if line.startswith("p cnf"):
                    nvars = int(line.split()[2])
                continue
            lits = [int(x) for x in line.split()]
            if lits and lits[-1] == 0:
                lits.pop()
            if lits:
                clauses.append(lits)
                nvars = max(nvars, max(abs(l) for l in lits))
    return clauses, nvars


def read_cubes(path):
    cubes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("a"):
                continue
            lits = [int(x) for x in line.split()[1:]]
            if lits and lits[-1] == 0:
                lits.pop()
            cubes.append(lits)
    return cubes


def bcp_fixed(clauses, nvars, assumed):
    """Unit-propagate `assumed` over `clauses`; return (n_fixed, conflict?).

    A light BCP purely to size sigma_all (the fixed-variable count the cutoff
    proxy uses) and to detect cubes propagation alone refutes. Not a solver.
    """
    val = {}  # var -> bool
    queue = list(assumed)
    # index clauses by literal for propagation
    watch = {}
    for ci, cl in enumerate(clauses):
        for l in cl:
            watch.setdefault(l, []).append(ci)

    def assign(lit):
        v, b = abs(lit), lit > 0
        if v in val:
            return val[v] == b  # False => conflict
        val[v] = b
        queue.append(lit)
        return True

    for lit in list(assumed):
        if not assign(lit):
            return len(val), True
    while queue:
        lit = queue.pop()
        # a clause is unit/violated only if the falsified literal is -lit
        for ci in watch.get(-lit, []):
            cl = clauses[ci]
            unassigned, sat = [], False
            for l in cl:
                v = abs(l)
                if v not in val:
                    unassigned.append(l)
                elif val[v] == (l > 0):
                    sat = True
                    break
            if sat:
                continue
            if not unassigned:
                return len(val), True
            if len(unassigned) == 1:
                if not assign(unassigned[0]):
                    return len(val), True
    return len(val), False


def base_cnf_text(clauses):
    """Clause body of the base CNF, pre-joined once so per-cube callers (44k+
    kissat launches in frontier_dp.py) don't rebuild it on every call."""
    return "".join(" ".join(map(str, cl)) + " 0\n" for cl in clauses)


def kissat_solve(clauses, nvars, cube, kissat, cube_timeout=None, base_text=None):
    """Solve base + cube-units with kissat. Returns (time, decisions, conflicts,
    result). A `cube_timeout` (seconds) censors a pathological cube: on timeout
    the wall is the timeout value and result is "timeout" — a real (maximal)
    difficulty data point, not a dropped one. `base_text` is an optional
    pre-joined clause body from `base_cnf_text` (pure speedup)."""
    body = [f"{l} 0" for l in cube]  # cube literals as units
    text = f"p cnf {nvars} {len(clauses) + len(cube)}\n"
    text += base_text if base_text is not None else base_cnf_text(clauses)
    text += "\n".join(body) + "\n"
    # Piped via stdin: at 40k+ conquers per instance a temp CNF per cube is
    # pure filesystem overhead (and deadly on GPFS scratch). NO -q: quiet mode
    # suppresses every `c` statistics line, which silently zeroed the parsed
    # conflicts/decisions for as long as this harness existed.
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            [kissat, "--relaxed", "/dev/stdin"],
            input=text,
            capture_output=True,
            text=True,
            timeout=cube_timeout,
        )
    except subprocess.TimeoutExpired:
        return float(cube_timeout), 0, 0, "timeout"
    dt = time.perf_counter() - t0
    decisions = conflicts = 0
    for line in p.stdout.splitlines():
        if line.startswith("c decisions:"):
            decisions = int(line.split()[2])
        elif line.startswith("c conflicts:"):
            conflicts = int(line.split()[2])
    result = {10: "sat", 20: "unsat"}.get(p.returncode, "unknown")
    return dt, decisions, conflicts, result


def pct(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (vx * vy) if vx and vy else float("nan")


def summarize(name, times, confs):
    def stats(v):
        n = len(v)
        mean = sum(v) / n
        var = sum((x - mean) ** 2 for x in v) / n
        cv = math.sqrt(var) / mean if mean else float("nan")
        return mean, pct(v, 0.5), pct(v, 0.95), pct(v, 0.99), max(v), cv

    print(f"\n=== {name}: {len(times)} cubes ===")
    print(f"  total conquer time: {sum(times):.3f}s   (max cube {max(times):.4f}s)")
    for label, v in (("time(s)", times), ("conflicts", confs)):
        mean, med, p95, p99, mx, cv = stats(v)
        print(
            f"  {label:9s} mean={mean:.4g} med={med:.4g} p95={p95:.4g} "
            f"p99={p99:.4g} max={mx:.4g}  CV={cv:.3f}"
        )
    print("  (CV = coefficient of variation = std/mean; LOWER = more uniform)")


def conquer_and_correlate(base, cubes_path, kissat="kissat", limit=None, cube_timeout=None):
    """Conquer every cube of `cubes_path` (against base CNF `base`) with kissat
    and gather per-signal series. Returns a dict with `times`, `confs`, the
    aligned `signals` (SIGNAL_COLS -> list over conquered cubes), `rows` (full
    per-cube records for CSV), `refuted`, and `timeouts` (censored cubes).
    `cube_timeout` (seconds) censors pathological cubes. Shared by `main()`
    (single run) and `signal_matrix.py` (cross-family sweep) so the conquer logic
    lives once."""
    clauses, nvars = read_cnf(base)
    cubes = read_cubes(cubes_path)
    meta = read_meta(cubes_path + ".meta")
    if meta is not None and len(meta) != len(cubes):
        print(
            f"  warning: {len(meta)} meta rows != {len(cubes)} cubes; "
            "signals disabled (order mismatch)",
            file=sys.stderr,
        )
        meta = None
    if limit:
        cubes = cubes[:limit]
        if meta is not None:
            meta = meta[:limit]

    rows = []
    times, confs = [], []
    signals = {name: [] for name in SIGNAL_COLS}
    refuted = 0
    timeouts = 0
    for i, cube in enumerate(cubes):
        s_all, conflict = bcp_fixed(clauses, nvars, cube)
        s_dec = len(cube)
        proxy = s_dec * s_all
        sig = {name: float("nan") for name in SIGNAL_COLS}
        sig["proxy_sd_sa"] = proxy
        if meta is not None:
            sig.update(signals_from_meta(meta[i]))
        if conflict:
            refuted += 1
            rows.append((i, s_dec, s_all, proxy, 0.0, 0, 0, "refuted-bcp", sig))
            continue
        dt, dec, cf, res = kissat_solve(clauses, nvars, cube, kissat, cube_timeout)
        if res == "timeout":
            timeouts += 1
        rows.append((i, s_dec, s_all, proxy, dt, dec, cf, res, sig))
        times.append(dt)
        confs.append(cf)
        for name in SIGNAL_COLS:
            signals[name].append(sig[name])
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(cubes)} cubes solved", file=sys.stderr)

    return {
        "nvars": nvars,
        "nclauses": len(clauses),
        "ncubes": len(cubes),
        "has_meta": meta is not None,
        "times": times,
        "confs": confs,
        "signals": signals,
        "rows": rows,
        "refuted": refuted,
        "timeouts": timeouts,
    }


def signal_rhos(signals, times, confs):
    """Per-signal (n, rho_time, rho_confl) over cubes where the signal is finite.
    Returns a dict signal -> (n, rho_time, rho_confl); rho is NaN if n <= 2."""
    result = {}
    for name in SIGNAL_COLS:
        xs, ts, cs = [], [], []
        for s, t, c in zip(signals[name], times, confs):
            if s == s and s not in (float("inf"), float("-inf")):
                xs.append(s)
                ts.append(t)
                cs.append(c)
        if len(xs) > 2:
            result[name] = (len(xs), spearman(xs, ts), spearman(xs, cs))
        else:
            result[name] = (len(xs), float("nan"), float("nan"))
    return result


def signal_table(signals, times, confs):
    """Print, for each candidate signal, its Spearman rho against conquer time
    and conflicts. The signal with the highest, most stable |rho| is the cutoff
    candidate."""
    rhos = signal_rhos(signals, times, confs)
    print("\n  cutoff-signal correlation (Spearman rho vs realized difficulty):")
    print(f"    {'signal':16s} {'n':>5s} {'rho(time)':>10s} {'rho(confl)':>11s}")
    for name in SIGNAL_COLS:
        n, rt, rc = rhos[name]
        if n > 2:
            print(f"    {name:16s} {n:5d} {rt:10.3f} {rc:11.3f}")
        else:
            print(f"    {name:16s} {n:5d} {'--':>10s} {'--':>11s}")
    finite = {k: v for k, v in rhos.items() if v[0] > 2 and v[1] == v[1]}
    if finite:
        best = max(finite.items(), key=lambda kv: abs(kv[1][1]))
        print(f"    -> strongest vs time: {best[0]} (rho={best[1][1]:.3f})")


def main():
    a = sys.argv
    base, cubes_path = a[1], a[2]
    out = a[a.index("--out") + 1] if "--out" in a else None
    limit = int(a[a.index("--limit") + 1]) if "--limit" in a else None
    kissat = a[a.index("--kissat") + 1] if "--kissat" in a else "kissat"

    r = conquer_and_correlate(base, cubes_path, kissat=kissat, limit=limit)
    print(
        f"base: {r['nvars']} vars, {r['nclauses']} clauses; {r['ncubes']} cubes"
        f"{' (+meta signals)' if r['has_meta'] else ''}"
    )

    summarize("conquer", r["times"], r["confs"])
    if r["refuted"]:
        print(f"  ({r['refuted']} cubes refuted by BCP alone — no kissat call)")
    if len(r["times"]) > 2:
        signal_table(r["signals"], r["times"], r["confs"])

    if out:
        with open(out, "w") as f:
            f.write(
                "idx,sigma_dec,sigma_all,time_s,decisions,conflicts,result,"
                + ",".join(SIGNAL_COLS)
                + "\n"
            )
            for row in r["rows"]:
                idx, s_dec, s_all, _proxy, dt, dec, cf, res, sig = row
                base_cols = [idx, s_dec, s_all, dt, dec, cf, res]
                sig_cols = [sig[name] for name in SIGNAL_COLS]
                f.write(",".join(map(str, base_cols + sig_cols)) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
