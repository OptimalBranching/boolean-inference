#!/usr/bin/env python3
"""Fetch MCC 2023 + 2024 competition instances, tracks 1 (mc) + 2 (wmc).

Idempotent: re-running skips archives whose sha256 already matches, and
re-classification is a cheap header read. Materialized data lands under
`benchmarks/bank/<family>/` (gitignored); provenance (archive sha256 + counts)
is written to `bank/<family>/provenance.json` for `bank_manifest_gen.py`.

Pipeline per year:
  1. discover the Zenodo record's files (API) and download them (resume/verify);
  2. extract tracks 1-2 ONLY. The 2024 archive is 60 GB uncompressed across all
     five tracks, so members are pre-filtered by path before extraction to
     respect the 12 GB bank cap; the 2023 archive is small enough to extract
     whole and then filter;
  3. classify each `.cnf` by its mandatory `c t mc|wmc|pmc|pwmc` header
     (layout-independent) into `mc/` and `wmc/`, discarding pmc/pwmc;
  4. optionally (--materialize-wcn) convert a SAMPLE to wcn-1 via `to_wcn`, to
     prove the conversion path without exploding disk.

Usage:
  fetch_mcc.py                       # fetch+classify both years
  fetch_mcc.py --years 2024          # one year
  fetch_mcc.py --materialize-wcn 20  # also convert 20 sampled instances/track
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank_common as bc  # noqa: E402
from bank_families import EXTERNAL  # noqa: E402

# Member paths worth extracting from a multi-track archive. Permissive: MCC
# archives have used mc/wmc/track1/track2/2024_MC-style names across years; the
# authoritative filter is still the `c t` header read in step 3.
TRACK_MEMBER_RE = re.compile(
    r"(track[_-]?1|track[_-]?2|(?:^|/)mc(?:$|[/_])|wmc|weighted|model.?count)",
    re.IGNORECASE)

# Zenodo file keys to SKIP at download time: projected tracks (pmc/pwmc,
# tracks 3-4) are deferred until M3 and never fetched here. `pmc` also matches
# `pwmc`; track ids cover naming variants. If this exclusion would drop every
# file (unexpected naming), everything is downloaded instead — the `c t` header
# classification remains the authoritative filter.
SKIP_KEY_RE = re.compile(r"(pmc|track[_-]?[3-9])", re.IGNORECASE)

TO_WCN = None  # resolved lazily to the built `to_wcn` binary


def mcc_families(years):
    want = {f"mcc-{y}" for y in years}
    return [f for f in EXTERNAL if f["id"] in want]


def _open_maybe_xz(path):
    """Text handle on a .cnf, transparently decompressing a .cnf.xz (MCC ships
    per-file XZ-compressed instances)."""
    if path.lower().endswith(".xz"):
        import lzma
        return lzma.open(path, "rt", errors="ignore")
    return open(path, "r", errors="ignore")


def classify_header(path, max_lines=200):
    """Return the MCC problem type ('mc'|'wmc'|'pmc'|'pwmc') from the `c t` line,
    or None. `c t` is mandatory in MCC 2021+; scan a bounded prefix."""
    try:
        with _open_maybe_xz(path) as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                s = line.strip()
                if s.startswith("c t "):
                    return s.split()[2].lower()
                # Stop scanning once real clauses begin.
                if s.startswith("p cnf") or (s and s[0].isdigit()):
                    # header may still appear after `p cnf`; keep scanning a bit
                    continue
    except (OSError, EOFError):
        return None
    return None


def selective_extract(archive, staging, big):
    """Extract into `staging`. For a big multi-track archive, list members and
    extract only track-1/2 candidates; otherwise extract whole."""
    os.makedirs(staging, exist_ok=True)
    if big:
        members = bc.list_archive_members(archive)
        keep = [m for m in members if TRACK_MEMBER_RE.search(m)]
        print(f"  [extract] {len(keep)}/{len(members)} members match tracks 1-2")
        if not keep:
            print("  [warn] no member matched the track filter; extracting whole "
                  "(header classification will still discard pmc/pwmc)")
            bc.unpack(archive, staging)
        else:
            bc.unpack(archive, staging, members=keep)
    else:
        bc.unpack(archive, staging)


def classify_tree(staging, family_dir, tracks):
    """Move every `.cnf` (or per-file-compressed `.cnf.xz`, decompressed on the
    way) under `staging` into `<family_dir>/<mc|wmc>/` per its header; discard
    other tracks. Returns {track: count}."""
    counts = {t: 0 for t in tracks}
    for t in tracks:
        os.makedirs(os.path.join(family_dir, t), exist_ok=True)
    for dp, _dn, fns in os.walk(staging):
        for fn in fns:
            if not fn.lower().endswith((".cnf", ".dimacs", ".cnf.xz",
                                        ".dimacs.xz")):
                continue
            src = os.path.join(dp, fn)
            typ = classify_header(src)
            if typ not in tracks:
                continue
            if fn.lower().endswith(".xz"):
                dst = os.path.join(family_dir, typ, fn[:-3])
                with open(dst, "wb") as out:
                    subprocess.run(["xz", "-dc", src], stdout=out, check=True)
                os.remove(src)
            else:
                dst = os.path.join(family_dir, typ, fn)
                shutil.move(src, dst)
            counts[typ] += 1
    return counts


def materialize_sample(family_dir, tracks, n):
    """Convert up to `n` instances/track to wcn-1 via `to_wcn` (proves the path;
    the bank keeps originals and converts on demand)."""
    global TO_WCN
    if TO_WCN is None:
        cand = os.path.join(bc.ROOT, "target", "release", "to_wcn")
        TO_WCN = cand if os.path.exists(cand) else None
    if not TO_WCN:
        print("  [wcn] to_wcn binary not built (cargo build --release); skipping")
        return
    made = 0
    for t in tracks:
        srcdir = os.path.join(family_dir, t)
        outdir = os.path.join(family_dir, "wcn", t)
        os.makedirs(outdir, exist_ok=True)
        files = sorted(f for f in os.listdir(srcdir)
                       if f.lower().endswith((".cnf", ".dimacs")))[:n]
        for fn in files:
            out = os.path.join(outdir, fn.rsplit(".", 1)[0] + ".json")
            r = subprocess.run([TO_WCN, os.path.join(srcdir, fn), out],
                               capture_output=True, text=True)
            if r.returncode == 0:
                made += 1
            else:
                print(f"  [wcn] FAILED {fn}: {r.stderr.strip()[:120]}")
    print(f"  [wcn] materialized {made} sample instances")


def fetch_family(fam, materialize_n):
    print(f"== {fam['id']}: {fam['title']} ==")
    ddir = bc.downloads_dir(fam["id"])
    fdir = bc.bank_family_dir(fam["id"])
    archives = {}
    files = bc.zenodo_files(fam["zenodo"])
    wanted = [f for f in files if not SKIP_KEY_RE.search(f["key"])]
    if not wanted:  # unexpected naming — fall back to everything
        wanted = files
    skipped = [f["key"] for f in files if f not in wanted]
    if skipped:
        print("  [skip tracks 3+] " + ", ".join(skipped))
    for fentry in wanted:
        key = fentry["key"]
        dest = os.path.join(ddir, key)
        sha = bc.download(fentry["url"], dest)
        archives[key] = sha
    # Extract + classify each downloaded archive.
    staging = os.path.join(fdir, "_staging")
    big = fam["id"] == "mcc-2024"
    for key in archives:
        arc = os.path.join(ddir, key)
        if not arc.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz",
                                     ".tar.xz", ".txz", ".xz")):
            continue
        selective_extract(arc, staging, big)
    counts = classify_tree(staging, fdir, fam["tracks"])
    shutil.rmtree(staging, ignore_errors=True)
    print(f"  classified: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    if materialize_n:
        materialize_sample(fdir, fam["tracks"], materialize_n)
    bc.write_provenance(fam["id"], archives, extra={
        "track_counts": counts,
        "materialized_wcn_sample": materialize_n or 0,
    })


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs="+", default=["2023", "2024"])
    ap.add_argument("--materialize-wcn", type=int, default=0, metavar="N",
                    help="also convert N sampled instances/track to wcn-1")
    args = ap.parse_args()

    fams = mcc_families(args.years)
    if not fams:
        bc.die(f"no MCC family for years {args.years}")
    for fam in fams:
        try:
            fetch_family(fam, args.materialize_wcn)
        except Exception as e:  # noqa: BLE001
            bc.die(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
