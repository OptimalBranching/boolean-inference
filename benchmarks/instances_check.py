#!/usr/bin/env python3
"""Verify the frozen instance bank (benchmarks/instances.json).

Each entry is checked by re-deriving its artifacts and comparing sha256 against
the manifest:
  - kind=file      : hash the committed file as-is.
  - kind=generated : run `cmd` (with {cnf}/{csp} placeholders bound to temp
                     paths), then hash each produced artifact.

A frozen bank is only trustworthy if regeneration is byte-identical, so a single
mismatch fails the whole run and names the offending entry — that is exactly the
negative control: change any seed/hash in the manifest and this exits nonzero.

Usage:
  instances_check.py --all              verify every entry; print "OK <n>/<n>"
  instances_check.py --entry <id> ...   verify only the named entries
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "benchmarks", "instances.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_file_entry(entry):
    """Return list of (artifact, expected, got) mismatches for a kind=file entry."""
    path = os.path.join(ROOT, entry["file"])
    if not os.path.exists(path):
        return [("file", entry["sha256"].get("file", ""), "<missing file>")]
    got = sha256_file(path)
    exp = entry["sha256"]["file"]
    return [] if got == exp else [("file", exp, got)]


def check_generated_entry(entry):
    """Regenerate into a temp dir and diff every declared artifact hash."""
    mismatches = []
    with tempfile.TemporaryDirectory() as tmp:
        paths = {name: os.path.join(tmp, f"artifact.{name}") for name in entry["sha256"]}
        cmd = entry["cmd"].format(**paths)
        proc = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            return [("<cmd>", "exit 0", f"exit {proc.returncode}: {proc.stderr.strip()[:200]}")]
        for name, exp in entry["sha256"].items():
            if not os.path.exists(paths[name]):
                mismatches.append((name, exp, "<no artifact produced>"))
                continue
            got = sha256_file(paths[name])
            if got != exp:
                mismatches.append((name, exp, got))
    return mismatches


def check_entry(entry):
    kind = entry.get("kind")
    if kind == "file":
        return check_file_entry(entry)
    if kind == "generated":
        return check_generated_entry(entry)
    return [("<kind>", "file|generated", repr(kind))]


def check_integrity(entries):
    """Cheap manifest-level checks independent of hashing."""
    problems = []
    ids = [e["id"] for e in entries]
    dupes = {i for i in ids if ids.count(i) > 1}
    for d in sorted(dupes):
        problems.append(f"duplicate id: {d}")
    idset = set(ids)
    for e in entries:
        ref = e.get("negative_control_of")
        if ref is not None and ref not in idset:
            problems.append(f"{e['id']}: negative_control_of points at unknown id '{ref}'")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="verify every entry")
    g.add_argument("--entry", nargs="+", metavar="ID", help="verify only these entry ids")
    args = ap.parse_args()

    with open(MANIFEST) as f:
        manifest = json.load(f)
    entries = manifest["entries"]

    if args.entry:
        by_id = {e["id"]: e for e in entries}
        missing = [i for i in args.entry if i not in by_id]
        if missing:
            print(f"FAIL: no such entry id(s): {', '.join(missing)}", file=sys.stderr)
            return 2
        entries = [by_id[i] for i in args.entry]

    problems = check_integrity(entries)
    for p in problems:
        print(f"FAIL integrity: {p}", file=sys.stderr)

    passed = 0
    for entry in entries:
        mism = check_entry(entry)
        if mism:
            for artifact, exp, got in mism:
                print(f"FAIL {entry['id']} [{artifact}]: expected {exp[:16]}… got {got[:64]}", file=sys.stderr)
        else:
            passed += 1

    total = len(entries)
    if passed == total and not problems:
        print(f"OK {passed}/{total}")
        return 0
    print(f"FAILED {passed}/{total} passed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
