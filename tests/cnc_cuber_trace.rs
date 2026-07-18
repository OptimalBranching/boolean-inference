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
