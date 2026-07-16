#!/usr/bin/env python3
"""Conquer a cube file with kissat and report the per-cube difficulty distribution.

Given a base DIMACS CNF and a cube file in march_cu's iCNF `a <lits> 0` format
(what both `gen_cubes` and `march_cu -o` produce, over the SAME variable
numbering), solve each cube = base CNF + its literals as unit clauses with
kissat, timing each. Reports the distribution whose LOW VARIANCE is the paper's
cube-uniformity claim, plus each cube's cutoff proxy sigma_dec*sigma_all
(sigma_dec = cube length; sigma_all = fixed vars after unit propagation) so the
proxy can be scattered against realized difficulty (experiment E1).

Usage:
  conquer_cubes.py <base.cnf> <cubes> [--out csv] [--limit N] [--kissat PATH]
                                      --contract-lock PATH [--stop-on-sat]
"""
import math
import re
import subprocess
import sys
import tempfile
import time


SHA256 = re.compile(r"^[0-9a-f]{64}$")


def read_contract_digest(path):
    """Read the frozen digest without adding a YAML dependency to this harness."""
    values = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("contract_digest:"):
                values.append(line.split(":", 1)[1].strip())
    if len(values) != 1 or SHA256.fullmatch(values[0]) is None:
        raise ValueError(f"{path}: expected one lowercase SHA-256 contract_digest")
    return values[0]


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


def kissat_solve(clauses, nvars, cube, kissat):
    body = [f"{l} 0" for l in cube]  # cube literals as units
    text = f"p cnf {nvars} {len(clauses) + len(cube)}\n"
    text += "".join(" ".join(map(str, cl)) + " 0\n" for cl in clauses)
    text += "\n".join(body) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".cnf", delete=True) as tf:
        tf.write(text)
        tf.flush()
        t0 = time.perf_counter()
        p = subprocess.run(
            [kissat, "-q", "--relaxed", tf.name],
            capture_output=True,
            text=True,
        )
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


def main():
    a = sys.argv
    if len(a) < 3 or "--contract-lock" not in a:
        raise SystemExit(
            "usage: conquer_cubes.py <base.cnf> <cubes> [--out csv] "
            "[--limit N] [--kissat PATH] --contract-lock PATH"
        )
    base, cubes_path = a[1], a[2]
    out = a[a.index("--out") + 1] if "--out" in a else None
    limit = int(a[a.index("--limit") + 1]) if "--limit" in a else None
    kissat = a[a.index("--kissat") + 1] if "--kissat" in a else "kissat"
    contract_digest = read_contract_digest(a[a.index("--contract-lock") + 1])

    clauses, nvars = read_cnf(base)
    cubes = read_cubes(cubes_path)
    if limit:
        cubes = cubes[:limit]
    print(
        f"base: {nvars} vars, {len(clauses)} clauses; {len(cubes)} cubes; "
        f"contract: {contract_digest}"
    )

    rows = []
    times, confs, proxies = [], [], []
    refuted = 0
    for i, cube in enumerate(cubes):
        s_all, conflict = bcp_fixed(clauses, nvars, cube)
        s_dec = len(cube)
        proxy = s_dec * s_all
        if conflict:
            refuted += 1
            rows.append((i, s_dec, s_all, proxy, 0.0, 0, 0, "refuted-bcp"))
            continue
        dt, dec, cf, res = kissat_solve(clauses, nvars, cube, kissat)
        rows.append((i, s_dec, s_all, proxy, dt, dec, cf, res))
        times.append(dt)
        confs.append(cf)
        proxies.append(proxy)
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(cubes)} cubes solved", file=sys.stderr)

    summarize("conquer", times, confs)
    if refuted:
        print(f"  ({refuted} cubes refuted by BCP alone — no kissat call)")
    if len(proxies) > 2:
        rho_t = spearman(proxies, times)
        rho_c = spearman(proxies, confs)
        print(
            f"  proxy sigma_dec*sigma_all -> difficulty: "
            f"Spearman rho vs time={rho_t:.3f}, vs conflicts={rho_c:.3f}"
        )

    if out:
        with open(out, "w") as f:
            f.write(
                "contract_digest,idx,sigma_dec,sigma_all,proxy,time_s,"
                "decisions,conflicts,result\n"
            )
            for r in rows:
                f.write(contract_digest + "," + ",".join(map(str, r)) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
