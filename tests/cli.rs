use std::process::Command;

// Cargo sets CARGO_BIN_EXE_<bin-name>; the binary name is the package name.
const BIN: &str = env!("CARGO_BIN_EXE_boolean-inference");

fn run(stdin: &str) -> (String, Option<i32>) {
    use std::io::Write;
    let mut child = Command::new(BIN)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
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
        out.status.code(),
    )
}

#[test]
fn cli_reports_sat() {
    let (stdout, code) = run("p cnf 3 3\n1 2 3 0\n-1 2 0\n-2 3 0\n");
    assert!(stdout.contains("s SATISFIABLE"), "stdout: {stdout}");
    assert!(stdout
        .lines()
        .any(|l| l.starts_with("v ") && l.trim_end().ends_with(" 0")));
    assert_eq!(code, Some(10));
}

#[test]
fn cli_reports_unsat() {
    let cnf = "p cnf 3 8\n\
        1 2 3 0\n1 2 -3 0\n1 -2 3 0\n1 -2 -3 0\n\
        -1 2 3 0\n-1 2 -3 0\n-1 -2 3 0\n-1 -2 -3 0\n";
    let (stdout, code) = run(cnf);
    assert!(stdout.contains("s UNSATISFIABLE"), "stdout: {stdout}");
    assert_eq!(code, Some(20));
}
