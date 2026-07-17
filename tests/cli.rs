use std::process::{Command, Stdio};

const BIN: &str = env!("CARGO_BIN_EXE_boolean-inference");

fn run(stdin: &str) -> (String, String, Option<i32>) {
    use std::io::Write;
    let mut child = Command::new(BIN)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn cli");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    let out = child.wait_with_output().expect("wait cli");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code(),
    )
}

#[test]
fn cli_reports_sat_with_named_wire_values() {
    let json = r#"{
        "variables": ["a", "b", "c"],
        "circuit": { "assignments": [
            { "outputs": ["c"], "expr": { "op": { "And": [
                { "op": { "Var": "a" } }, { "op": { "Var": "b" } }
            ] } } }
        ] }
    }"#;
    let (stdout, stderr, code) = run(json);
    assert!(stderr.is_empty(), "stderr: {stderr}");
    assert!(stdout.contains("s SATISFIABLE"), "stdout: {stdout}");
    assert!(stdout.lines().any(|line| line.starts_with("v a=")));
    assert_eq!(code, Some(10));
}

#[test]
fn cli_reports_unsat() {
    let json = r#"{
        "variables": ["x"],
        "circuit": { "assignments": [
            { "outputs": ["x"], "expr": { "op": { "Const": true } } },
            { "outputs": ["x"], "expr": { "op": { "Const": false } } }
        ] }
    }"#;
    let (stdout, stderr, code) = run(json);
    assert!(stderr.is_empty(), "stderr: {stderr}");
    assert!(stdout.contains("s UNSATISFIABLE"), "stdout: {stdout}");
    assert_eq!(code, Some(20));
}

#[test]
fn cli_rejects_dimacs_input() {
    let (stdout, stderr, code) = run("p cnf 1 1\n1 0\n");
    assert!(stdout.is_empty(), "stdout: {stdout}");
    assert!(stderr.contains("invalid CircuitSAT"), "stderr: {stderr}");
    assert_eq!(code, Some(2));
}
