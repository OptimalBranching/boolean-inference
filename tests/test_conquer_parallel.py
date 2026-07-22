import concurrent.futures

import pytest

from benchmarks.cnc import conquer_parallel
from benchmarks.cnc.solve import conquer_frontier


def test_parallel_conquer_reports_unsat_distribution(tmp_path, monkeypatch):
    cnf = tmp_path / "base.cnf"
    cnf.write_text("p cnf 2 1\n1 2 0\n", encoding="utf-8")
    cubes = tmp_path / "cubes.icnf"
    cubes.write_text("".join(f"a {literal} 0\n" for literal in [1, -1, 2, -2]))
    kissat = tmp_path / "kissat"
    kissat.write_text(
        "#!/bin/sh\n"
        "echo 'c decisions: 7'\n"
        "echo 'c conflicts: 3'\n"
        "exit 20\n",
        encoding="utf-8",
    )
    kissat.chmod(0o755)
    monkeypatch.setattr(
        concurrent.futures,
        "ProcessPoolExecutor",
        concurrent.futures.ThreadPoolExecutor,
    )

    result = conquer_frontier(
        cnf,
        cubes,
        kissat,
        workers=2,
        timeout_s=5,
        out_dir=tmp_path / "out",
        tmp_dir=tmp_path / "tmp",
    )

    assert result["result"] == "unsat"
    assert result["complete"] is True
    assert result["completed"] == 4
    assert result["total_decisions"] == 28
    assert result["total_conflicts"] == 12
    assert result["not_started"] == 0


def test_parallel_conquer_stops_submitting_after_sat(tmp_path, monkeypatch):
    cnf = tmp_path / "base.cnf"
    cnf.write_text("p cnf 1 0\n", encoding="utf-8")
    cubes = tmp_path / "cubes.icnf"
    cubes.write_text("a 1 0\n" + "a -1 0\n" * 9, encoding="utf-8")
    kissat = tmp_path / "kissat"
    kissat.write_text("#!/bin/sh\nexit 10\n", encoding="utf-8")
    kissat.chmod(0o755)
    monkeypatch.setattr(
        concurrent.futures,
        "ProcessPoolExecutor",
        concurrent.futures.ThreadPoolExecutor,
    )

    result = conquer_frontier(
        cnf,
        cubes,
        kissat,
        workers=1,
        timeout_s=5,
        out_dir=tmp_path / "out",
    )

    assert result["result"] == "sat"
    assert result["complete"] is True
    assert result["completed"] == 1
    assert result["not_started"] == 9


def test_frontier_literals_are_checked_against_cnf_variables(tmp_path):
    frontier = tmp_path / "bad.icnf"
    frontier.write_text("a 3 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="variable range"):
        list(conquer_parallel.read_cubes(frontier, variables=2))


def test_censored_cubes_remain_in_work_accounting():
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
        replay_workers=[2],
        wall_s=5.2,
        measured_makespan_s=5.0,
    )
    assert result["result"] == "timeout"
    assert result["complete"] is False
    assert result["censored"] is True
    assert result["total_solver_s"] == 6.0
    assert result["lpt_is_lower_bound"] is True
