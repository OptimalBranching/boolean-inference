#!/usr/bin/env python3
"""Fetch the UAI PR (partition-function) corpus and convert its BINARY-domain
instances to wcn-1 via uai_to_wcn.py.

Sources (survey §2.3): the MIT dechterlab mirror tarball (2006/2008/2012/2014
.uai + .uai.evid) and the UCI-2022 Final + Tuning PR.zip sets. Multi-valued
instances (linkage/Promedus/protein, cardinality up to ~80) are SKIPPED, not
log-encoded (survey gap #2); the convertible/skipped tally is recorded in
provenance so the manifest reports exactly how much of the corpus is binary.

Idempotent: archives whose sha256 matches are skipped; conversion overwrites.

Usage:
  fetch_uai.py                 # download, unpack, convert binary instances
  fetch_uai.py --no-convert    # download + unpack only
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank_common as bc  # noqa: E402
import uai_to_wcn  # noqa: E402
from bank_families import EXTERNAL  # noqa: E402

FAM = next(f for f in EXTERNAL if f["id"] == "uai-pr")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-convert", action="store_true")
    args = ap.parse_args()

    print(f"== {FAM['id']}: {FAM['title']} ==")
    ddir = bc.downloads_dir(FAM["id"])
    fdir = bc.bank_family_dir(FAM["id"])
    raw = os.path.join(fdir, "raw")
    archives = {}
    for url in FAM["urls"]:
        name = url.split("/")[-1]
        if name == "master.tar.gz":  # disambiguate the two PR.zip archives
            name = "uai-competitions-master.tar.gz"
        if name == "PR.zip":
            name = ("PR-final.zip" if "FinalBenchmarks" in url
                    else "PR-tuning.zip")
        dest = os.path.join(ddir, name)
        try:
            archives[name] = bc.download(url, dest)
            bc.unpack(dest, raw)
        except Exception as e:  # noqa: BLE001
            bc.die(str(e))

    tally = {"binary": 0, "skipped": 0, "errors": 0}
    if not args.no_convert:
        wcn = os.path.join(fdir, "wcn")
        tally = uai_to_wcn.run_dir(raw, wcn)

    bc.write_provenance(FAM["id"], archives, extra={"convert_tally": tally})
    return 0


if __name__ == "__main__":
    sys.exit(main())
