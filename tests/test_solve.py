import json

from benchmarks.cnc.solve import run_kissat


def test_run_kissat_records_direct_result(tmp_path):
    cnf = tmp_path / "instance.cnf"
    cnf.write_text("p cnf 1 1\n1 0\n", encoding="utf-8")
    kissat = tmp_path / "kissat"
    kissat.write_text(
        "#!/bin/sh\n"
        "echo 'c decisions: 7'\n"
        "echo 'c conflicts: 3'\n"
        "exit 10\n",
        encoding="utf-8",
    )
    kissat.chmod(0o755)

    record = run_kissat(cnf, kissat, timeout_s=5, out_dir=tmp_path / "run")

    assert record["result"] == "sat"
    assert record["decisions"] == 7
    assert record["conflicts"] == 3
    assert json.loads((tmp_path / "run/summary.json").read_text()) == record
