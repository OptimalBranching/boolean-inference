from pathlib import Path

from benchmarks.cnc import cubing


def successful_cuber(command):
    frontier = Path(command[command.index("-o") + 1])
    frontier.parent.mkdir(parents=True, exist_ok=True)
    frontier.write_text("a 1 0\na -1 0\n", encoding="utf-8")


def fake_run_process(command, **kwargs):
    successful_cuber(command)
    return {"returncode": 0, "timed_out": False, "wall_s": 0.1}


def fake_conquer(*args, **kwargs):
    return {"result": "unsat", "complete": True, "cubes": 2}


def test_march_pipeline_uses_cnf_then_shared_conquer(tmp_path, monkeypatch):
    monkeypatch.setattr(cubing, "run_process", fake_run_process)
    monkeypatch.setattr(cubing, "conquer_frontier", fake_conquer)

    record = cubing.march_then_conquer(
        tmp_path / "instance.cnf",
        tmp_path / "march_cu",
        tmp_path / "kissat",
        workers=4,
        cube_timeout_s=5,
        cubing_timeout_s=5,
        remaining_vars=20,
        out_dir=tmp_path / "run",
    )

    assert record["mode"] == "march-cu"
    assert record["cubing"]["cubes"] == 2
    assert record["cubing"]["command"][1].endswith("instance.cnf")
    assert record["cubing"]["command"][-4:-2] == ["-n", "20"]


def test_frozen_frontier_goes_directly_to_shared_conquer(tmp_path, monkeypatch):
    monkeypatch.setattr(cubing, "conquer_frontier", fake_conquer)
    (tmp_path / "instance.cnf").write_text("p cnf 1 0\n", encoding="utf-8")
    (tmp_path / "frontier.icnf").write_text("a 1 0\na -1 0\n", encoding="utf-8")

    record = cubing.frozen_then_conquer(
        tmp_path / "instance.cnf",
        tmp_path / "frontier.icnf",
        tmp_path / "kissat",
        workers=4,
        cube_timeout_s=5,
        out_dir=tmp_path / "run",
    )

    assert record["mode"] == "frozen-frontier"
    assert record["cubes"] == 2
    assert record["frontier"].endswith("frontier.icnf")
