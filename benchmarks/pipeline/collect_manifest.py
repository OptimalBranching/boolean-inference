#!/usr/bin/env python3
"""Collect sidecar metadata, verify artifact hashes, and write a sorted manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .circuit import CircuitError, load_json, sha256_file, write_jsonl
except ImportError:  # direct script execution
    from circuit import CircuitError, load_json, sha256_file, write_jsonl  # type: ignore


def collect(root: Path) -> list[dict]:
    records = []
    ids: set[str] = set()
    for metadata_path in sorted(root.rglob("*.meta.json")):
        record = load_json(metadata_path)
        instance_id = record.get("id")
        if not isinstance(instance_id, str) or not instance_id:
            raise CircuitError(f"{metadata_path}: missing instance id")
        if instance_id in ids:
            raise CircuitError(f"duplicate instance id {instance_id!r}")
        ids.add(instance_id)
        for key in ("circuitsat", "cnf"):
            relative = record.get(key)
            expected = record.get(f"{key}_sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise CircuitError(f"{metadata_path}: missing {key} artifact or digest")
            artifact = root / relative
            actual = sha256_file(artifact)
            if actual != expected:
                raise CircuitError(f"{metadata_path}: {key} digest mismatch")
        records.append(record)
    if not records:
        raise CircuitError(f"{root}: no *.meta.json records found")
    return sorted(records, key=lambda item: item["id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        records = collect(args.root)
    except (CircuitError, OSError) as exc:
        parser.error(str(exc))
    write_jsonl(args.out, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
