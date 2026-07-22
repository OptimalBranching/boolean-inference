#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use boolean_inference::conquer::{ConquerResult, StreamingConquer};

fn temp_dir() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "boolean-inference-streaming-{}-{nonce}",
        std::process::id()
    ))
}

fn fake_kissat(path: &Path, result: &str, code: i32) {
    fs::write(
        path,
        format!(
            "#!/bin/sh\n\
             [ \"$#\" -eq 1 ] && [ \"$1\" = --relaxed ] || exit 3\n\
             input=$(cat)\n\
             case \"$input\" in *\"p cnf 2 \"*) ;; *) exit 4 ;; esac\n\
             echo 'c output that streaming mode must discard'\n\
             echo 's {result}'\n\
             exit {code}\n"
        ),
    )
    .expect("write fake Kissat");
    let mut permissions = fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).unwrap();
}

fn run_streaming(dir: &Path, kissat: &Path) -> std::process::Output {
    let cnf = dir.join("input.cnf");
    fs::write(&cnf, "p cnf 2 2\n1 2 0\n-1 -2 0\n").expect("write CNF");
    Command::new(env!("CARGO_BIN_EXE_cnc_cuber"))
        .args([
            cnf.as_os_str(),
            "-n".as_ref(),
            "2".as_ref(),
            "--solve-cnf".as_ref(),
            cnf.as_os_str(),
            "--kissat".as_ref(),
            kissat.as_os_str(),
            "--workers".as_ref(),
            "2".as_ref(),
        ])
        .output()
        .expect("run streaming solver")
}

#[test]
fn streaming_mode_reports_unsat_after_all_open_cubes_close() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let kissat = dir.join("kissat-unsat");
    fake_kissat(&kissat, "UNSATISFIABLE", 20);

    let output = run_streaming(&dir, &kissat);

    assert_eq!(output.status.code(), Some(20), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stdout).contains("s UNSATISFIABLE"));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("submitted=1"), "{stderr}");
    assert!(stderr.contains("unsat=1"), "{stderr}");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn streaming_mode_stops_after_a_sat_cube() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let kissat = dir.join("kissat-sat");
    fake_kissat(&kissat, "SATISFIABLE", 10);

    let output = run_streaming(&dir, &kissat);

    assert_eq!(output.status.code(), Some(10), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("s SATISFIABLE"));
    assert!(!stdout.contains("output that streaming mode must discard"));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("sat=1"), "{stderr}");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn streaming_mode_kills_an_inflight_solver_after_sat() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let cnf = dir.join("input.cnf");
    fs::write(&cnf, "p cnf 2 0\n").unwrap();
    let started = dir.join("slow-started");
    let kissat = dir.join("kissat-race");
    fs::write(
        &kissat,
        format!(
            "#!/bin/sh\n\
             input=$(cat)\n\
             case \"$input\" in\n\
               *'-1 0'*) while [ ! -e '{}' ]; do sleep 0.01; done; echo 's SATISFIABLE'; exit 10 ;;\n\
               *) touch '{}'; exec sleep 30 ;;\n\
             esac\n",
            started.display(),
            started.display(),
        ),
    )
    .unwrap();
    let mut permissions = fs::metadata(&kissat).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&kissat, permissions).unwrap();

    let conquer = StreamingConquer::start(&cnf, &kissat, 2).unwrap();
    assert!(conquer.submit(vec![1]).unwrap());
    assert!(conquer.submit(vec![-1]).unwrap());
    let before = Instant::now();
    let summary = conquer.finish(true).unwrap();

    assert_eq!(summary.result, ConquerResult::Sat);
    assert_eq!(summary.sat, 1);
    assert!(started.exists(), "slow Kissat never started");
    assert!(
        before.elapsed() < Duration::from_secs(10),
        "in-flight Kissat was not killed: {:?}",
        before.elapsed()
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn streaming_mode_never_claims_unsat_after_a_worker_error() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let kissat = dir.join("kissat-error");
    fake_kissat(&kissat, "UNKNOWN", 1);

    let output = run_streaming(&dir, &kissat);

    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(!String::from_utf8_lossy(&output.stdout).contains("UNSATISFIABLE"));
    assert!(String::from_utf8_lossy(&output.stderr).contains("incomplete"));
    fs::remove_dir_all(dir).unwrap();
}
