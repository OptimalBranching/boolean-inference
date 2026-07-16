#!/usr/bin/env python3
"""Download a public benchmark artifact with mandatory provenance metadata."""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

try:
    from .circuit import CircuitError, sha256_file, write_json
except ImportError:  # direct script execution
    from circuit import CircuitError, sha256_file, write_json  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--sha256", help="expected digest; required after the first acquisition"
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=args.out.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urllib.request.urlopen(args.url) as response:  # noqa: S310 - explicit CLI URL
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
        actual = sha256_file(temporary_path)
        if args.sha256 and actual != args.sha256:
            temporary_path.unlink(missing_ok=True)
            raise CircuitError(
                f"download digest mismatch: expected {args.sha256}, got {actual}"
            )
        temporary_path.replace(args.out)
        write_json(
            args.out.with_suffix(args.out.suffix + ".source.json"),
            {"url": args.url, "sha256": actual, "artifact": args.out.name},
        )
        print(actual)
    except (OSError, CircuitError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
