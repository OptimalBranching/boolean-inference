use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "boolean-inference-trace-{}-{nonce}",
        std::process::id()
    ))
}

#[test]
fn trace_flag_preserves_cubes_and_writes_original_variable_ids() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).expect("create temp directory");
    let input = dir.join("input.cnf");
    let plain = dir.join("plain.cubes");
    let traced = dir.join("traced.cubes");
    let trace = dir.join("nodes.jsonl");
    fs::write(&input, "p cnf 3 4\n1 2 0\n-1 -2 0\n2 3 0\n-2 -3 0\n").expect("write CNF");

    let binary = env!("CARGO_BIN_EXE_cnc_cuber");
    let plain_run = Command::new(binary)
        .args([
            input.as_os_str(),
            "-n".as_ref(),
            "3".as_ref(),
            "-o".as_ref(),
            plain.as_os_str(),
            "--max-rows".as_ref(),
            "32".as_ref(),
        ])
        .output()
        .expect("run plain cuber");
    assert!(plain_run.status.success(), "{:?}", plain_run);

    let traced_run = Command::new(binary)
        .args([
            input.as_os_str(),
            "-n".as_ref(),
            "3".as_ref(),
            "-o".as_ref(),
            traced.as_os_str(),
            "--max-rows".as_ref(),
            "32".as_ref(),
            "--trace".as_ref(),
            trace.as_os_str(),
        ])
        .output()
        .expect("run traced cuber");
    assert!(traced_run.status.success(), "{:?}", traced_run);

    assert_eq!(fs::read(&plain).unwrap(), fs::read(&traced).unwrap());
    let cube_count = fs::read_to_string(&traced).unwrap().lines().count();
    let records: Vec<serde_json::Value> = fs::read_to_string(&trace)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).expect("valid trace JSON"))
        .collect();
    assert!(!records.is_empty());
    assert_eq!(records[0]["node_id"], 0);
    assert!(records[0]["parent_id"].is_null());
    assert!(records
        .iter()
        .all(|record| { record["kind"] == "refuted" || record["refutation_reason"].is_null() }));
    assert_eq!(
        records
            .iter()
            .filter(|record| record["kind"] == "cutoff")
            .count(),
        cube_count
    );
    for record in &records {
        for variable in record["rule_variables"]
            .as_array()
            .expect("rule variable array")
        {
            assert!((1..=3).contains(&variable.as_u64().unwrap()));
        }
    }

    fs::remove_dir_all(dir).expect("remove temp directory");
}

#[test]
fn structure_blind_selector_is_auditable_binary_control() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).expect("create temp directory");
    let input = dir.join("input.cnf");
    let cubes = dir.join("blind.cubes");
    let trace = dir.join("blind.jsonl");
    fs::write(&input, "p cnf 3 4\n1 2 0\n-1 -2 0\n2 3 0\n-2 -3 0\n").expect("write CNF");

    let run = Command::new(env!("CARGO_BIN_EXE_cnc_cuber"))
        .args([
            input.as_os_str(),
            "-n".as_ref(),
            "3".as_ref(),
            "-o".as_ref(),
            cubes.as_os_str(),
            "--selector".as_ref(),
            "structure-blind".as_ref(),
            "--trace".as_ref(),
            trace.as_os_str(),
        ])
        .output()
        .expect("run structure-blind cuber");
    assert!(run.status.success(), "{:?}", run);
    let stderr = String::from_utf8(run.stderr).expect("UTF-8 stderr");
    assert!(stderr.contains("selector=structure-blind"), "{stderr}");

    let records: Vec<serde_json::Value> = fs::read_to_string(&trace)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).expect("valid trace JSON"))
        .collect();
    for branch in records.iter().filter(|record| record["kind"] == "branch") {
        assert_eq!(branch["rule_variables"].as_array().unwrap().len(), 1);
        let clauses = branch["rule_clauses"].as_array().unwrap();
        assert_eq!(clauses.len(), 2);
        assert_eq!(clauses[0]["mask"], 1);
        assert_eq!(clauses[0]["value"], 0);
        assert_eq!(clauses[1]["mask"], 1);
        assert_eq!(clauses[1]["value"], 1);
    }

    fs::remove_dir_all(dir).expect("remove temp directory");
}

#[test]
fn root_refutation_trace_records_a_semantic_closure_reason() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).expect("create temp directory");
    let input = dir.join("root-unsat.cnf");
    let cubes = dir.join("root-unsat.cubes");
    let trace = dir.join("root-unsat.jsonl");
    fs::write(&input, "p cnf 1 2\n1 0\n-1 0\n").expect("write CNF");

    let run = Command::new(env!("CARGO_BIN_EXE_cnc_cuber"))
        .args([
            input.as_os_str(),
            "-n".as_ref(),
            "1".as_ref(),
            "-o".as_ref(),
            cubes.as_os_str(),
            "--trace".as_ref(),
            trace.as_os_str(),
        ])
        .output()
        .expect("run root-UNSAT cuber");
    assert!(run.status.success(), "{:?}", run);
    assert!(fs::read_to_string(&cubes).unwrap().is_empty());
    let record: serde_json::Value =
        serde_json::from_str(fs::read_to_string(&trace).unwrap().trim()).unwrap();
    assert_eq!(record["kind"], "refuted");
    assert_eq!(
        record["refutation_reason"],
        "root-propagation-contradiction"
    );

    fs::remove_dir_all(dir).expect("remove temp directory");
}
