#!/usr/bin/env python3
"""Archive-now the rotting projected-counting links (survey deliverable 4).

These are stored but NOT converted: they need projected counting (M3), which is
not implemented — our parser correctly REJECTS `c p show`. The point is to pin a
copy while the links are alive (Klebanov's companion solver dir already 404'd).

Families (deferred=True in bank_families.py):
  * kr2024-projected — KR-2024 "Model Counting in the Wild" projected archive
    (117 QIF + functional-synthesis / reliability / NN-verification), 1.4 GB,
    from Zenodo 13284882 (only the projected zip).
  * klebanov-qif     — qif-cnf.tgz, projected #SAT leakage instances.
Also fetches kr2024-mc (the model-counting archive incl. the 411 crypto
instances), which is NOT deferred — it converts free via to_wcn.

Idempotent; provenance records archive sha256 + on-disk size.

Usage:
  fetch_archive.py                 # fetch all archive-now families
  fetch_archive.py --only kr2024-mc
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank_common as bc  # noqa: E402
from bank_families import EXTERNAL  # noqa: E402

ARCHIVE_IDS = ["kr2024-mc", "kr2024-projected", "klebanov-qif"]


def fetch_one(fam):
    print(f"== {fam['id']}: {fam['title']} ==")
    ddir = bc.downloads_dir(fam["id"])
    archives = {}
    only = set(fam.get("only_files") or [])
    if fam.get("zenodo"):
        files = bc.zenodo_files(fam["zenodo"])
        for fentry in files:
            if only and fentry["key"] not in only:
                continue
            dest = os.path.join(ddir, fentry["key"])
            archives[fentry["key"]] = bc.download(fentry["url"], dest)
    for url in fam.get("urls", []):
        name = url.split("/")[-1]
        dest = os.path.join(ddir, name)
        archives[name] = bc.download(url, dest)
    extra = {"deferred": fam.get("deferred", False),
             "conversion": fam["conversion"]}
    # kr2024-mc is not deferred: unpack so bank_check / materialization can use it.
    if not fam.get("deferred"):
        raw = os.path.join(bc.bank_family_dir(fam["id"]), "raw")
        for name in archives:
            try:
                bc.unpack(os.path.join(ddir, name), raw)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] unpack {name}: {e}")
        extra["instance_count"] = bc.count_files(raw, (".cnf", ".dimacs"))
    bc.write_provenance(fam["id"], archives, extra=extra)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=ARCHIVE_IDS)
    args = ap.parse_args()
    ids = args.only or ARCHIVE_IDS
    fams = [f for f in EXTERNAL if f["id"] in ids]
    for fam in fams:
        try:
            fetch_one(fam)
        except Exception as e:  # noqa: BLE001
            bc.die(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
