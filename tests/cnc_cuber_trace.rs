use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let nonce = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "boolean-inference-trace-{}-{nonce}",
        std::process::id()
    ))
}

#[test]
fn experiment_configuration_must_be_explicit() {
    let dir = temp_dir();
    let input = dir.join("input.cnf");
    let output = dir.join("cubes.icnf");
    let binary = env!("CARGO_BIN_EXE_cnc_cuber");

    let missing_solver = Command::new(binary)
        .args([
            input.as_os_str(),
            "-n".as_ref(),
            "1".as_ref(),
            "-o".as_ref(),
            output.as_os_str(),
            "--measure".as_ref(),
            "vars".as_ref(),
        ])
        .output()
        .expect("run without branch solver");
    assert!(!missing_solver.status.success());
    assert!(
        String::from_utf8_lossy(&missing_solver.stderr).contains("missing --branch-solver"),
        "{missing_solver:?}"
    );

    let missing_measure = Command::new(binary)
        .args([
            input.as_os_str(),
            "-n".as_ref(),
            "1".as_ref(),
            "-o".as_ref(),
            output.as_os_str(),
            "--branch-solver".as_ref(),
            "greedy".as_ref(),
        ])
        .output()
        .expect("run without measure");
    assert!(!missing_measure.status.success());
    assert!(
        String::from_utf8_lossy(&missing_measure.stderr).contains("missing --measure"),
        "{missing_measure:?}"
    );
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
            "--branch-solver".as_ref(),
            "greedy".as_ref(),
            "--measure".as_ref(),
            "vars".as_ref(),
            "--max-rows".as_ref(),
            "1".as_ref(),
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
            "--branch-solver".as_ref(),
            "greedy".as_ref(),
            "--measure".as_ref(),
            "vars".as_ref(),
            "--max-rows".as_ref(),
            "1".as_ref(),
            "--trace".as_ref(),
            trace.as_os_str(),
            "--trace-replay".as_ref(),
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
    assert!(records
        .iter()
        .all(|record| record.get("schema_version").is_none()));
    assert!(records
        .iter()
        .all(|record| record["search_semantics"] == "sat-decision"));
    assert!(records.iter().all(|record| record["selector"] == "region"));
    assert!(records
        .iter()
        .all(|record| record["branch_solver"] == "greedy"));
    assert!(records.iter().all(|record| record["measure"] == "vars"));
    assert!(records.iter().all(|record| record["propagation"] == "ct"));
    assert!(records.iter().all(|record| record["cdcl_mode"] == "off"));
    assert!(records
        .iter()
        .all(|record| record["input_kind"] == "dimacs"));
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
    let branches: Vec<_> = records
        .iter()
        .filter(|record| record["kind"] == "branch")
        .collect();
    assert!(!branches.is_empty());
    for branch in branches {
        let diagnostics = branch["rule_diagnostics"]
            .as_object()
            .expect("region branches carry mechanism diagnostics");
        assert_eq!(diagnostics["rule_semantics"], "cover");
        assert_eq!(diagnostics["cover_verified"], true);
        assert!((1..=3).contains(&diagnostics["focus_variable"].as_u64().unwrap()));
        assert!(diagnostics["region_tensors"].as_u64().unwrap() >= 1);
        assert_eq!(
            diagnostics["region_variables"].as_u64().unwrap() as usize,
            branch["rule_variables"].as_array().unwrap().len()
        );
        assert!(diagnostics["boundary_variables"].as_u64().unwrap() >= 1);
        assert!(
            diagnostics["joined_rows"].as_u64().unwrap()
                >= diagnostics["feasible_rows"].as_u64().unwrap()
        );
        assert_eq!(diagnostics["closed"], false);
        assert_eq!(
            diagnostics["branching_vector"].as_array().unwrap().len(),
            branch["rule_clauses"].as_array().unwrap().len()
        );
        assert!(diagnostics["gamma"].is_number());
        let timing = diagnostics["timing_ns"]
            .as_object()
            .expect("stage timing object");
        assert!(timing["region_growth"].is_number());
        assert!(timing["feasibility_probe"].is_number());
        assert!(timing["rule_solver"].is_number());
        let replay = diagnostics["same_state_replay"]
            .as_object()
            .expect("explicit replay flag produces counterfactuals");
        assert_eq!(replay["binary"]["branches"], 2);
        assert_eq!(
            replay["binary"]["branching_vector"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(replay["naive"]["branches"], diagnostics["feasible_rows"]);
        assert_eq!(
            replay["naive"]["decision_literals"].as_u64().unwrap(),
            diagnostics["feasible_rows"].as_u64().unwrap()
                * diagnostics["region_variables"].as_u64().unwrap()
        );
        assert!(
            diagnostics["gamma"].as_f64().unwrap()
                <= replay["naive"]["gamma"].as_f64().unwrap() + 1e-12
        );
    }

    fs::remove_dir_all(dir).expect("remove temp directory");
}

#[test]
fn cdcl_propagation_matches_ct_on_a_clause_network() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).expect("create temp directory");
    let input = dir.join("input.cnf");
    let ct_cubes = dir.join("ct.cubes");
    let cdcl_cubes = dir.join("cdcl.cubes");
    let cdcl_trace = dir.join("cdcl.jsonl");
    fs::write(
        &input,
        "p cnf 6 8\n\
         1 2 0\n-1 -2 0\n2 3 0\n-2 -3 0\n\
         4 5 0\n-4 -5 0\n5 6 0\n-5 -6 0\n",
    )
    .expect("write CNF");

    let binary = env!("CARGO_BIN_EXE_cnc_cuber");
    let common = [
        "-n",
        "4",
        "--branch-solver",
        "greedy",
        "--measure",
        "vars",
        "--max-rows",
        "1",
    ];
    let ct = Command::new(binary)
        .arg(&input)
        .args(["-o"])
        .arg(&ct_cubes)
        .args(common)
        .output()
        .expect("run CT cuber");
    assert!(ct.status.success(), "{ct:?}");

    let cdcl = Command::new(binary)
        .arg(&input)
        .args(["-o"])
        .arg(&cdcl_cubes)
        .args(common)
        .args(["--propagation", "cdcl", "--trace"])
        .arg(&cdcl_trace)
        .output()
        .expect("run CDCL cuber");
    assert!(cdcl.status.success(), "{cdcl:?}");
    assert_eq!(fs::read(&ct_cubes).unwrap(), fs::read(&cdcl_cubes).unwrap());
    assert!(
        String::from_utf8_lossy(&cdcl.stderr).contains("propagation=cdcl"),
        "{cdcl:?}"
    );
    let records: Vec<serde_json::Value> = fs::read_to_string(&cdcl_trace)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).expect("valid trace JSON"))
        .collect();
    assert!(!records.is_empty());
    assert!(records.iter().all(|record| record["propagation"] == "cdcl"));
    assert!(records
        .iter()
        .all(|record| record["cdcl_mode"] == "branch-learning"));

    fs::remove_dir_all(dir).expect("remove temp directory");
}

#[test]
fn branch_learning_handles_a_non_bcp_unsat_formula_without_full_search() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).expect("create temp directory");
    let input = dir.join("learning-unsat.cnf");
    let cubes = dir.join("learning-unsat.cubes");
    let trace = dir.join("learning-unsat.jsonl");
    // PHP(4,3): root BCP stays open. Repeated branch propagation retains
    // conflict clauses, but the cuber-side CaDiCaL never starts a full solve.
    fs::write(
        &input,
        "p cnf 12 22\n\
         1 2 3 0\n4 5 6 0\n7 8 9 0\n10 11 12 0\n\
         -1 -4 0\n-1 -7 0\n-1 -10 0\n-4 -7 0\n-4 -10 0\n-7 -10 0\n\
         -2 -5 0\n-2 -8 0\n-2 -11 0\n-5 -8 0\n-5 -11 0\n-8 -11 0\n\
         -3 -6 0\n-3 -9 0\n-3 -12 0\n-6 -9 0\n-6 -12 0\n-9 -12 0\n",
    )
    .expect("write non-BCP UNSAT CNF");

    let run = Command::new(env!("CARGO_BIN_EXE_cnc_cuber"))
        .arg(&input)
        .args(["-n", "1", "-o"])
        .arg(&cubes)
        .args([
            "--branch-solver",
            "greedy",
            "--measure",
            "vars",
            "--propagation",
            "cdcl",
            "--trace",
        ])
        .arg(&trace)
        .output()
        .expect("run learning CDCL cuber");
    assert!(run.status.success(), "{run:?}");
    assert!(fs::read_to_string(&cubes).unwrap().is_empty());
    let records: Vec<serde_json::Value> = fs::read_to_string(&trace)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).expect("valid trace JSON"))
        .collect();
    assert!(!records.is_empty());
    assert!(records
        .iter()
        .all(|record| record["cdcl_mode"] == "branch-learning"));
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(stderr.contains("cdcl_mode=branch-learning"), "{stderr}");
    assert!(stderr.contains("full_search_calls=0"), "{stderr}");
    assert!(stderr.contains("learned_total="), "{stderr}");

    fs::remove_dir_all(dir).expect("remove temp directory");
}

#[test]
fn hybrid_uses_ct_candidates_and_committed_branch_cdcl_learning() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).expect("create temp directory");
    let input = dir.join("learning-unsat.cnf");
    let cubes = dir.join("hybrid.cubes");
    let trace = dir.join("hybrid.jsonl");
    fs::write(
        &input,
        "p cnf 12 22\n\
         1 2 3 0\n4 5 6 0\n7 8 9 0\n10 11 12 0\n\
         -1 -4 0\n-1 -7 0\n-1 -10 0\n-4 -7 0\n-4 -10 0\n-7 -10 0\n\
         -2 -5 0\n-2 -8 0\n-2 -11 0\n-5 -8 0\n-5 -11 0\n-8 -11 0\n\
         -3 -6 0\n-3 -9 0\n-3 -12 0\n-6 -9 0\n-6 -12 0\n-9 -12 0\n",
    )
    .expect("write non-BCP UNSAT CNF");

    let binary = env!("CARGO_BIN_EXE_cnc_cuber");
    let run = Command::new(binary)
        .arg(&input)
        .args(["-n", "1", "-o"])
        .arg(&cubes)
        .args([
            "--branch-solver",
            "greedy",
            "--measure",
            "vars",
            "--propagation",
            "hybrid",
            "--trace",
        ])
        .arg(&trace)
        .output()
        .expect("run hybrid cuber");
    assert!(run.status.success(), "{run:?}");
    assert!(fs::read_to_string(&cubes).unwrap().is_empty());
    let records: Vec<serde_json::Value> = fs::read_to_string(&trace)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).expect("valid trace JSON"))
        .collect();
    assert!(!records.is_empty());
    assert!(records
        .iter()
        .all(|record| record["propagation"] == "hybrid"));
    assert!(records
        .iter()
        .all(|record| record["cdcl_mode"] == "branch-learning"));
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(stderr.contains("propagation=hybrid"), "{stderr}");
    assert!(stderr.contains("full_search_calls=0"), "{stderr}");
    assert!(stderr.contains("learned_total="), "{stderr}");

    fs::remove_dir_all(dir).expect("remove temp directory");
}

#[test]
fn native_regions_can_use_a_matching_flattened_cnf_for_cdcl_propagation() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).expect("create temp directory");
    let input = dir.join("xor.json");
    let cnf = dir.join("xor.cnf");
    let ct_cubes = dir.join("ct.cubes");
    let cdcl_cubes = dir.join("cdcl.cubes");
    fs::write(
        &input,
        r#"{
          "variables": ["a", "b", "c"],
          "circuit": {"assignments": [
            {"outputs": ["c"], "expr": {"op": {"Xor": [
              {"op": {"Var": "a"}}, {"op": {"Var": "b"}}
            ]}}}
          ]}
        }"#,
    )
    .expect("write CircuitSAT");
    fs::write(
        &cnf,
        "p cnf 3 4\n-1 -2 -3 0\n1 2 -3 0\n1 -2 3 0\n-1 2 3 0\n",
    )
    .expect("write matching Tseitin CNF");

    let binary = env!("CARGO_BIN_EXE_cnc_cuber");
    let common = [
        "-n",
        "3",
        "--branch-solver",
        "greedy",
        "--measure",
        "vars",
        "--max-rows",
        "1",
    ];
    let ct = Command::new(binary)
        .arg(&input)
        .args(["-o"])
        .arg(&ct_cubes)
        .args(common)
        .output()
        .expect("run native CT cuber");
    assert!(ct.status.success(), "{ct:?}");

    let cdcl = Command::new(binary)
        .arg(&input)
        .args(["-o"])
        .arg(&cdcl_cubes)
        .args(common)
        .args(["--propagation", "cdcl", "--propagate-cnf"])
        .arg(&cnf)
        .output()
        .expect("run native/CDCL cuber");
    assert!(cdcl.status.success(), "{cdcl:?}");
    assert_eq!(fs::read(&ct_cubes).unwrap(), fs::read(&cdcl_cubes).unwrap());

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
            "--branch-solver".as_ref(),
            "greedy".as_ref(),
            "--measure".as_ref(),
            "vars".as_ref(),
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
        assert_eq!(branch["selector"], "structure-blind");
        assert!(branch["rule_diagnostics"].is_null());
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
fn naive_branch_solver_is_an_auditable_full_tree_control() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).expect("create temp directory");
    let input = dir.join("input.cnf");
    let cubes = dir.join("naive.cubes");
    let trace = dir.join("naive.jsonl");
    fs::write(&input, "p cnf 3 4\n1 2 0\n-1 -2 0\n2 3 0\n-2 -3 0\n").expect("write CNF");

    let run = Command::new(env!("CARGO_BIN_EXE_cnc_cuber"))
        .args([
            input.as_os_str(),
            "-n".as_ref(),
            "3".as_ref(),
            "-o".as_ref(),
            cubes.as_os_str(),
            "--max-rows".as_ref(),
            "1".as_ref(),
            "--branch-solver".as_ref(),
            "naive".as_ref(),
            "--measure".as_ref(),
            "vars".as_ref(),
            "--trace".as_ref(),
            trace.as_os_str(),
            "--trace-replay".as_ref(),
        ])
        .output()
        .expect("run NaiveBranch cuber");
    assert!(run.status.success(), "{:?}", run);
    let stderr = String::from_utf8(run.stderr).expect("UTF-8 stderr");
    assert!(stderr.contains("branch_solver=naive"), "{stderr}");

    let records: Vec<serde_json::Value> = fs::read_to_string(&trace)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).expect("valid trace JSON"))
        .collect();
    for branch in records.iter().filter(|record| record["kind"] == "branch") {
        assert_eq!(branch["branch_solver"], "naive");
        let diagnostics = &branch["rule_diagnostics"];
        if diagnostics["rule_semantics"] != "cover" {
            continue;
        }
        assert_eq!(diagnostics["cover_verified"], true);
        assert_eq!(
            branch["rule_clauses"].as_array().unwrap().len() as u64,
            diagnostics["same_state_replay"]["naive"]["branches"]
                .as_u64()
                .unwrap()
        );
        assert_eq!(
            diagnostics["branching_vector"],
            diagnostics["same_state_replay"]["naive"]["branching_vector"]
        );
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
            "--branch-solver".as_ref(),
            "greedy".as_ref(),
            "--measure".as_ref(),
            "vars".as_ref(),
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
    assert!(record.get("schema_version").is_none());
    assert!(record["rule_diagnostics"].is_null());
    assert_eq!(
        record["refutation_reason"],
        "root-propagation-contradiction"
    );

    fs::remove_dir_all(dir).expect("remove temp directory");
}

#[test]
fn native_circuit_trace_exposes_the_same_region_mechanism_without_cnf() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).expect("create temp directory");
    let input = dir.join("native.json");
    let cubes = dir.join("native.cubes");
    let trace = dir.join("native.jsonl");
    fs::write(
        &input,
        r#"{
          "variables": ["a", "b", "c", "d", "e"],
          "circuit": {"assignments": [
            {"outputs": ["c"], "expr": {"op": {"Xor": [
              {"op": {"Var": "a"}}, {"op": {"Var": "b"}}
            ]}}},
            {"outputs": ["d"], "expr": {"op": {"And": [
              {"op": {"Var": "c"}}, {"op": {"Var": "b"}}
            ]}}},
            {"outputs": ["e"], "expr": {"op": {"Or": [
              {"op": {"Var": "d"}}, {"op": {"Var": "a"}}
            ]}}}
          ]}
        }"#,
    )
    .expect("write native CircuitSAT input");

    let run = Command::new(env!("CARGO_BIN_EXE_cnc_cuber"))
        .args([
            input.as_os_str(),
            "-n".as_ref(),
            "5".as_ref(),
            "-o".as_ref(),
            cubes.as_os_str(),
            "--branch-solver".as_ref(),
            "greedy".as_ref(),
            "--measure".as_ref(),
            "vars".as_ref(),
            "--max-rows".as_ref(),
            "32".as_ref(),
            "--trace".as_ref(),
            trace.as_os_str(),
            "--trace-replay".as_ref(),
        ])
        .output()
        .expect("run native CircuitSAT cuber");
    assert!(run.status.success(), "{:?}", run);

    let records: Vec<serde_json::Value> = fs::read_to_string(&trace)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).expect("valid trace JSON"))
        .collect();
    let branch = records
        .iter()
        .find(|record| record["kind"] == "branch")
        .expect("native constraint network should branch");
    let diagnostics = &branch["rule_diagnostics"];
    assert_eq!(branch["input_kind"], "circuit-sat");
    assert!(diagnostics.is_object());
    assert_eq!(diagnostics["rule_semantics"], "closed-witness");
    assert_eq!(diagnostics["closed"], true);
    assert!(diagnostics["region_tensors"].as_u64().unwrap() >= 1);
    assert!(diagnostics["feasible_rows"].as_u64().unwrap() > 0);
    assert!(diagnostics["gamma"].is_number());
    let replay = diagnostics["same_state_replay"]
        .as_object()
        .expect("closed region replay");
    assert_eq!(replay["binary"]["branches"], 2);
    assert_eq!(replay["naive"]["branches"], diagnostics["feasible_rows"]);

    fs::remove_dir_all(dir).expect("remove temp directory");
}

#[test]
fn native_extensional_csp_trace_uses_relation_tensors_without_cnf() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).expect("create temp directory");
    let input = dir.join("native.csp");
    let cubes = dir.join("native.cubes");
    let trace = dir.join("native.jsonl");
    fs::write(
        &input,
        "6\n0 1 2 : 1 2 4 7\n1 2 3 : 0 3 5 6\n2 3 4 : 1 2 4 7\n3 4 5 : 0 3 5 6\n",
    )
    .expect("write native CSP input");

    let run = Command::new(env!("CARGO_BIN_EXE_cnc_cuber"))
        .args([
            input.as_os_str(),
            "-n".as_ref(),
            "6".as_ref(),
            "-o".as_ref(),
            cubes.as_os_str(),
            "--branch-solver".as_ref(),
            "greedy".as_ref(),
            "--measure".as_ref(),
            "vars".as_ref(),
            "--max-rows".as_ref(),
            "1".as_ref(),
            "--trace".as_ref(),
            trace.as_os_str(),
            "--trace-replay".as_ref(),
        ])
        .output()
        .expect("run native extensional CSP cuber");
    assert!(run.status.success(), "{:?}", run);

    let records: Vec<serde_json::Value> = fs::read_to_string(&trace)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).expect("valid trace JSON"))
        .collect();
    let branch = records
        .iter()
        .find(|record| record["kind"] == "branch")
        .expect("native relation network should branch");
    assert_eq!(branch["selector"], "region");
    assert_eq!(branch["input_kind"], "extensional-csp");
    assert!(
        branch["rule_diagnostics"]["region_tensors"]
            .as_u64()
            .unwrap()
            >= 1
    );
    assert_eq!(branch["rule_diagnostics"]["cover_verified"], true);

    fs::remove_dir_all(dir).expect("remove temp directory");
}

#[test]
fn tail_greedy_preserves_the_initial_weakest_child() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).expect("create temp directory");
    let input = dir.join("chain.csp");
    let cubes = dir.join("chain.cubes");
    let trace = dir.join("chain.jsonl");
    // Two equality relations. With max_rows=1 the first two-variable tensor is
    // the selected region and NaiveBranch provides its unmerged floor.
    fs::write(&input, "3\n0 1 : 0 3\n1 2 : 0 3\n").expect("write native CSP input");

    let run = Command::new(env!("CARGO_BIN_EXE_cnc_cuber"))
        .args([
            input.as_os_str(),
            "-n".as_ref(),
            "3".as_ref(),
            "-o".as_ref(),
            cubes.as_os_str(),
            "--max-rows".as_ref(),
            "1".as_ref(),
            "--branch-solver".as_ref(),
            "tail-greedy".as_ref(),
            "--measure".as_ref(),
            "vars".as_ref(),
            "--trace".as_ref(),
            trace.as_os_str(),
            "--trace-replay".as_ref(),
        ])
        .output()
        .expect("run tail-aware cuber");
    assert!(run.status.success(), "{:?}", run);
    let stderr = String::from_utf8(run.stderr).expect("UTF-8 stderr");
    assert!(stderr.contains("branch_solver=tail-greedy"), "{stderr}");

    let records: Vec<serde_json::Value> = fs::read_to_string(&trace)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).expect("valid trace JSON"))
        .collect();
    let branch = records
        .iter()
        .find(|record| record["kind"] == "branch")
        .expect("equality chain should branch");
    let diagnostics = &branch["rule_diagnostics"];
    assert_eq!(branch["branch_solver"], "tail-greedy");
    assert_eq!(diagnostics["region_variables"], 2);
    assert_eq!(diagnostics["boundary_variables"], 1);
    assert_eq!(diagnostics["feasible_rows"], 2);
    assert_eq!(diagnostics["branching_rows"], 2);
    assert_eq!(branch["rule_variables"].as_array().unwrap().len(), 2);
    assert_eq!(diagnostics["cover_verified"], true);

    let selected_min = diagnostics["branching_vector"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .fold(f64::INFINITY, f64::min);
    let initial_min = diagnostics["same_state_replay"]["naive"]["branching_vector"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .fold(f64::INFINITY, f64::min);
    assert!(selected_min + 1e-12 >= initial_min);

    fs::remove_dir_all(dir).expect("remove temp directory");
}
