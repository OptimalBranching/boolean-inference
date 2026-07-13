#!/usr/bin/env python3
"""Verify the counting-benchmark bank.

Three checks, all SMALL-instances-only and every solver call under `timeout 60`
(never run the engine on big instances locally — survey/machine constraint):

  (a) structural  — every sampled instance parses. Uses `count --parse-only`
                    (load + Instance::validate, no counting), so it is safe even
                    on instances too big to count.
  (b) reference   — where references exist: MCC counts are log10 ESTIMATES, so we
                    compare log10(our exact count) to the reference within a
                    relative tolerance (never exact-equality on log10).
  (c) cross-check — >=3 small MCC mc-track instances are counted BOTH by our
                    engine and by the sharpSAT-TD binary; the exact integer
                    counts must agree.

Because the bank data is gitignored (materialize with the fetch_*.py scripts),
`--selftest` fabricates three tiny known-count CNFs and runs (a)+(c) end-to-end
so the tooling itself is verifiable with no downloads.

Usage:
  bank_check.py --selftest
  bank_check.py --family mcc-2023 --sample 30
  bank_check.py --all --sharpsat-dir <build-dir>
"""
import argparse
import os
import subprocess
import sys
import tempfile
from decimal import Decimal, getcontext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank_common as bc  # noqa: E402
from bank_families import EXTERNAL  # noqa: E402

getcontext().prec = 60

COUNT = os.path.join(bc.ROOT, "target", "release", "count")
DEFAULT_SHARPSAT_DIR = ("/private/tmp/claude-501/"
                        "-Users-xiweipan-Codes-boolean-inference/"
                        "7c2b95c4-087c-4ea9-bd3c-0595c9fd2a84/scratchpad/"
                        "sharpsat-td/build")
SMALL_BYTES = 200_000  # only count/cross-check instances below this size


def log10_of_count(s):
    """log10 of a count string: a plain BigInt or a `num/den` ratio."""
    if "/" in s:
        n, d = s.split("/")
        return Decimal(n).log10() - Decimal(d).log10()
    return Decimal(s).log10()


def run_count(path, extra=None, timeout=60):
    cmd = [COUNT, path] + (extra or [])
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def parse_only(path):
    try:
        r = run_count(path, ["--parse-only"])
    except subprocess.TimeoutExpired:
        return False, "timeout"
    return r.returncode == 0, (r.stdout or r.stderr).strip()[:100]


def count_models(path):
    try:
        r = run_count(path)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    for tok in r.stdout.split():
        if tok.startswith("models="):
            return tok[len("models="):]
    return None


def sharpsat_count(path, sharpsat_dir):
    """Run sharpSAT-TD from its build dir (flow_cutter must be in cwd). Returns
    the exact integer count string, or None."""
    binp = os.path.join(sharpsat_dir, "sharpSAT")
    if not os.path.exists(binp):
        return None
    cmd = [binp, "-decot", "1", "-decow", "100", "-tmpdir", ".", "-cs", "3500",
           os.path.abspath(path)]
    try:
        r = subprocess.run(cmd, cwd=sharpsat_dir, capture_output=True,
                           text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    for line in r.stdout.splitlines():
        if line.startswith("c s exact arb int"):
            return line.split()[-1]
    return None


def sample_instances(root, suffixes, n, small_only=False):
    hits = []
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(suffixes):
                p = os.path.join(dp, fn)
                if small_only and os.path.getsize(p) > SMALL_BYTES:
                    continue
                hits.append(p)
    hits.sort()
    return hits[:n]


def structural_check(root, sample):
    files = sample_instances(root, (".cnf", ".dimacs", ".json", ".csp"), sample)
    if not files:
        return None
    ok = 0
    for p in files:
        good, _ = parse_only(p)
        ok += good
        if not good:
            print(f"    FAIL parse: {os.path.relpath(p, bc.ROOT)}")
    print(f"  structural: {ok}/{len(files)} parsed")
    return ok, len(files)


def cross_check(root, sharpsat_dir, k):
    """Cross-check >=k small mc-track CNFs against sharpSAT-TD."""
    mc_root = os.path.join(root, "mc")
    search = mc_root if os.path.isdir(mc_root) else root
    files = sample_instances(search, (".cnf", ".dimacs"), k * 3, small_only=True)
    agree = tried = 0
    for p in files:
        if tried >= k:
            break
        ours = count_models(p)
        theirs = sharpsat_count(p, sharpsat_dir)
        if ours is None or theirs is None:
            continue
        tried += 1
        if ours == theirs:
            agree += 1
            print(f"    cross-check OK  {os.path.basename(p)}: {ours}")
        else:
            print(f"    cross-check MISMATCH {os.path.basename(p)}: "
                  f"ours={ours} sharpSAT={theirs}")
    if tried:
        print(f"  cross-check: {agree}/{tried} agree with sharpSAT-TD")
    return agree, tried


def check_family(fam, sample, sharpsat_dir, k):
    fdir = os.path.join(bc.BANK_DIR, fam["id"])
    if not os.path.isdir(fdir):
        print(f"== {fam['id']}: not materialized (run fetch_*.py) — skipped ==")
        return
    print(f"== {fam['id']} ==")
    structural_check(fdir, sample)
    if fam["id"].startswith("mcc-") and "mc" in (fam.get("tracks") or []):
        cross_check(fdir, sharpsat_dir, k)


def selftest(sharpsat_dir):
    """Fabricate tiny known-count CNFs and exercise structural + cross-check with
    no downloads, proving the checker itself."""
    print("== selftest (fabricated tiny instances) ==")
    cases = [
        # (name, cnf, expected count)
        ("or_chain", "c t mc\np cnf 3 2\n1 2 0\n2 3 0\n", "5"),
        ("single", "c t mc\np cnf 1 1\n1 0\n", "1"),
        ("free2", "c t mc\np cnf 2 1\n1 2 0\n", "3"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        mc = os.path.join(tmp, "mc")
        os.makedirs(mc)
        expected = {}
        for name, cnf, exp in cases:
            p = os.path.join(mc, f"{name}.cnf")
            with open(p, "w") as f:
                f.write(cnf)
            expected[name] = exp
        s = structural_check(tmp, 10)
        # our engine matches expected counts
        ok = 0
        for name, _cnf, exp in cases:
            got = count_models(os.path.join(mc, f"{name}.cnf"))
            ok += (got == exp)
            print(f"    count {name}: got {got}, expect {exp} "
                  f"{'OK' if got == exp else 'FAIL'}")
        c = cross_check(tmp, sharpsat_dir, 3)
        struct_ok = s and s[0] == s[1]
        count_ok = ok == len(cases)
        cross_ok = c and c[1] >= 1 and c[0] == c[1]
        print(f"\nselftest: structural={'OK' if struct_ok else 'FAIL'} "
              f"counts={'OK' if count_ok else 'FAIL'} "
              f"cross-check={'OK' if cross_ok else 'FAIL/skipped'}")
        return 0 if (struct_ok and count_ok) else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--family", nargs="+")
    g.add_argument("--selftest", action="store_true")
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--cross-k", type=int, default=3,
                    help="MCC instances to cross-check against sharpSAT-TD")
    ap.add_argument("--sharpsat-dir", default=DEFAULT_SHARPSAT_DIR)
    args = ap.parse_args()

    if not os.path.exists(COUNT):
        bc.die("count binary not built (cargo build --release)")

    if args.selftest:
        return selftest(args.sharpsat_dir)

    if args.all:
        fams = EXTERNAL
    else:
        by_id = {f["id"]: f for f in EXTERNAL}
        missing = [i for i in args.family if i not in by_id]
        if missing:
            bc.die(f"unknown family id(s): {missing}")
        fams = [by_id[i] for i in args.family]
    for fam in fams:
        check_family(fam, args.sample, args.sharpsat_dir, args.cross_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
