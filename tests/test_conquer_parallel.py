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
    assert result["complete"] is True
    assert result["result"] == "unsat"
    assert result["terminal_records"] == 5
    assert result["lpt_makespan_by_workers_s"]["2"] > 0
    raw = [json.loads(line) for line in (out_dir / "test.jsonl").read_text().splitlines()]
    assert all(row["released_monotonic_ns"] <= row["started_monotonic_ns"] for row in raw)
    assert all(row["started_monotonic_ns"] <= row["finished_monotonic_ns"] for row in raw)
    assert all(row["worker_pid"] > 0 for row in raw)
    expected_measured = (
        max(row["collected_monotonic_ns"] for row in raw)
        - min(row["released_monotonic_ns"] for row in raw)
    ) / 1e9
    assert result["measured_makespan_s"] == expected_measured
    assert result["observed_parallel_wall_s"] >= result["measured_makespan_s"]
    assert len((out_dir / "test.jsonl").read_text().splitlines()) == 5


def test_censored_cubes_remain_in_work_and_lpt_accounting():
    result = conquer_parallel.summarize(
        cubes=2,
        completed=1,
        timeouts=1,
        errors=0,
        sat=0,
        unsat=1,
        durations=[1.0, 5.0],
        cpu_durations=[0.5, 4.5],
        decisions=[7.0],
        conflicts=[3.0],
        workers=2,
        replay_workers=[2, 4],
        wall_s=5.2,
        measured_makespan_s=5.0,
    )
    assert result["result"] == "timeout"
    assert result["complete"] is False
    assert result["censored"] is True
    assert result["total_solver_s"] == 6.0
    assert result["total_cpu_s"] == 5.0
    assert result["lpt_is_lower_bound"] is True
    assert result["lpt_makespan_by_workers_s"] == {"2": 5.0, "4": 5.0}
    assert result["conflicts_max"] == 3.0
