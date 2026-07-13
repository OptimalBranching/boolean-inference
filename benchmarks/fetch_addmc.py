#!/usr/bin/env python3
"""Fetch the ADDMC 1,914-instance weighted suite (the exact set every DP-side
baseline — ADDMC / DPMC / TensorOrder2 — publishes numbers on).

Source: the ADDMC v1.0.0 release asset `benchmarks.zip` (survey §7; the cited
Rochester URL is dead). CNF + literal weights; converts free via `to_wcn`
(Instance::from_dimacs_text). Idempotent; provenance records the archive sha256
and the instance count.

Usage:
  fetch_addmc.py                     # download + unpack
  fetch_addmc.py --materialize-wcn 20  # also convert 20 sampled instances
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank_common as bc  # noqa: E402
from bank_families import EXTERNAL  # noqa: E402

FAM = next(f for f in EXTERNAL if f["id"] == "addmc-1914")


def materialize_sample(raw, fdir, n):
    tw = os.path.join(bc.ROOT, "target", "release", "to_wcn")
    if not os.path.exists(tw):
        print("  [wcn] to_wcn not built; skipping materialization")
        return 0
    outdir = os.path.join(fdir, "wcn")
    os.makedirs(outdir, exist_ok=True)
    cnfs = []
    for dp, _dn, fns in os.walk(raw):
        for fn in fns:
            if fn.lower().endswith((".cnf", ".dimacs")):
                cnfs.append(os.path.join(dp, fn))
    made = 0
    for src in sorted(cnfs)[:n]:
        out = os.path.join(outdir, os.path.basename(src).rsplit(".", 1)[0] + ".json")
        r = subprocess.run([tw, src, out], capture_output=True, text=True)
        if r.returncode == 0:
            made += 1
        else:
            print(f"  [wcn] FAILED {os.path.basename(src)}: "
                  f"{r.stderr.strip()[:120]}")
    print(f"  [wcn] materialized {made} sample instances")
    return made


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--materialize-wcn", type=int, default=0, metavar="N")
    args = ap.parse_args()

    print(f"== {FAM['id']}: {FAM['title']} ==")
    ddir = bc.downloads_dir(FAM["id"])
    fdir = bc.bank_family_dir(FAM["id"])
    raw = os.path.join(fdir, "raw")
    archives = {}
    for url in FAM["urls"]:
        dest = os.path.join(ddir, url.split("/")[-1])
        try:
            archives[url.split("/")[-1]] = bc.download(url, dest)
            bc.unpack(dest, raw)
        except Exception as e:  # noqa: BLE001
            bc.die(str(e))

    n_cnf = bc.count_files(raw, (".cnf", ".dimacs"))
    print(f"  instances: {n_cnf}")
    made = materialize_sample(raw, fdir, args.materialize_wcn) \
        if args.materialize_wcn else 0
    bc.write_provenance(FAM["id"], archives, extra={
        "instance_count": n_cnf, "materialized_wcn_sample": made})
    return 0


if __name__ == "__main__":
    sys.exit(main())
