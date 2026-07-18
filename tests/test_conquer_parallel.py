import concurrent.futures
import json
import sys

from benchmarks.cnc import conquer_parallel


def test_parallel_conquer_streams_cubes_and_reports_conflict_distribution(
    tmp_path, monkeypatch
):
    cnf = tmp_path / "base.cnf"
    cnf.write_text("p cnf 2 1\n1 2 0\n")
    cubes = tmp_path / "cubes.icnf"
    cubes.write_text("".join(f"a {literal} 0\n" for literal in [1, -1, 2, -2, 1]))
    fake_kissat = tmp_path / "kissat"
    fake_kissat.write_text(
        "#!/bin/sh\n"
        "case \" $* \" in *\" --statistics \"*) ;; *) exit 2 ;; esac\n"
        "echo 'c decisions: 7'\n"
        "echo 'c conflicts: 3'\n"
        "exit 20\n"
    )
    fake_kissat.chmod(0o755)
    out_dir = tmp_path / "out"
    temp_dir = tmp_path / "tmp"

    monkeypatch.setattr(
        concurrent.futures,
        "ProcessPoolExecutor",
        concurrent.futures.ThreadPoolExecutor,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "conquer_parallel.py",
            str(cnf),
            "--arm",
            f"test={cubes}",
            "--kissat",
            str(fake_kissat),
            "--workers",
            "2",
            "--timeout-s",
            "5",
            "--tmp-dir",
            str(temp_dir),
            "--out-dir",
            str(out_dir),
        ],
    )
    conquer_parallel.main()

    summary = json.loads((out_dir / "summary.json").read_text())
    result = summary["arms"]["test"]
    assert result["cubes"] == 5
    assert result["completed"] == 5
    assert result["errors"] == 0
    assert result["timeouts"] == 0
    assert result["total_decisions"] == 35
    assert result["total_conflicts"] == 15
    assert result["conflicts_p50"] == 3
    assert result["conflicts_p99_over_p95"] == 1
    assert len((out_dir / "test.jsonl").read_text().splitlines()) == 5
