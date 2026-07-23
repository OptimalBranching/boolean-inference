#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use boolean_inference::conquer::{ConquerResult, StreamingConquer};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    let sequence = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "boolean-inference-streaming-{}-{nonce}-{sequence}",
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
            "--branch-solver".as_ref(),
            "greedy".as_ref(),
            "--measure".as_ref(),
            "vars".as_ref(),
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
fn conquer_first_answer_interrupts_the_ct_cuber() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let cnf = dir.join("many-cubes.cnf");
    let mut formula = String::from("p cnf 12 12\n");
    for pair in 0..6 {
        let left = pair * 2 + 1;
        let right = left + 1;
        formula.push_str(&format!("{left} {right} 0\n-{left} -{right} 0\n"));
    }
    fs::write(&cnf, formula).unwrap();

    let kissat = dir.join("kissat-first-answer");
    fs::write(
        &kissat,
        "#!/bin/sh\n\
         [ \"$#\" -eq 1 ] && [ \"$1\" = --relaxed ] || exit 3\n\
         cat >/dev/null\n\
         echo 's SATISFIABLE'\n\
         exit 10\n",
    )
    .unwrap();
    let mut permissions = fs::metadata(&kissat).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&kissat, permissions).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_cnc_cuber"))
        .arg(&cnf)
        .args(["-n", "1", "--solve-cnf"])
        .arg(&cnf)
        .args([
            "--kissat",
            kissat.to_str().unwrap(),
            "--workers",
            "1",
            "--selector",
            "structure-blind",
            "--branch-solver",
            "greedy",
            "--measure",
            "vars",
        ])
        .output()
        .expect("run first-answer CnC");

    assert_eq!(output.status.code(), Some(10), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("status=SAT_EARLY cubes_submitted="),
        "{stderr}"
    );
    assert!(stderr.contains("sat=1"), "{stderr}");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cuber_cdcl_does_not_solve_sat_before_a_cutoff_cube_reaches_conquer() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let cnf = dir.join("sat.cnf");
    // PHP(3,4) is SAT without a root unit. The cuber-side CaDiCaL may only
    // propagate selected branches; a cutoff cube must reach the conquer solver.
    fs::write(
        &cnf,
        "p cnf 12 33\n\
         1 2 3 4 0\n5 6 7 8 0\n9 10 11 12 0\n\
         -1 -2 0\n-1 -3 0\n-1 -4 0\n-2 -3 0\n-2 -4 0\n-3 -4 0\n\
         -5 -6 0\n-5 -7 0\n-5 -8 0\n-6 -7 0\n-6 -8 0\n-7 -8 0\n\
         -9 -10 0\n-9 -11 0\n-9 -12 0\n-10 -11 0\n-10 -12 0\n-11 -12 0\n\
         -1 -5 0\n-1 -9 0\n-5 -9 0\n-2 -6 0\n-2 -10 0\n-6 -10 0\n\
         -3 -7 0\n-3 -11 0\n-7 -11 0\n-4 -8 0\n-4 -12 0\n-8 -12 0\n",
    )
    .unwrap();
    let kissat = dir.join("kissat-sat");
    fs::write(
        &kissat,
        "#!/bin/sh\n\
         [ \"$#\" -eq 1 ] && [ \"$1\" = --relaxed ] || exit 3\n\
         cat >/dev/null\n\
         echo 's SATISFIABLE'\n\
         exit 10\n",
    )
    .unwrap();
    let mut permissions = fs::metadata(&kissat).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&kissat, permissions).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_cnc_cuber"))
        .arg(&cnf)
        .args(["-n", "1", "--solve-cnf"])
        .arg(&cnf)
        .args([
            "--kissat",
            kissat.to_str().unwrap(),
            "--workers",
            "1",
            "--branch-solver",
            "tail-greedy",
            "--measure",
            "vars",
            "--propagation",
            "hybrid",
        ])
        .output()
        .expect("run branch-learning CnC solver");

    assert_eq!(output.status.code(), Some(10), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("s SATISFIABLE"), "{stdout}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("status=OK cubes=1")
            || (stderr.contains("status=SAT_EARLY cubes_submitted=")
                && !stderr.contains("status=SAT_EARLY cubes_submitted=0")),
        "{stderr}"
    );
    assert!(stderr.contains("full_search_calls=0"), "{stderr}");
    assert!(
        stderr.contains("streaming submitted=1 sat=1 unsat=0 errors=0"),
        "{stderr}"
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn externally_reported_sat_model_stops_workers_and_preserves_the_witness() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let cnf = dir.join("input.cnf");
    fs::write(&cnf, "p cnf 2 1\n1 2 0\n").unwrap();
    let conquer = StreamingConquer::start(&cnf, &dir.join("must-not-run"), 2).unwrap();
    let witness = "s SATISFIABLE\nv 1 -2 0\n".to_string();

    conquer.mark_sat_with_witness(witness.clone());
    let summary = conquer.finish(false).unwrap();

    assert_eq!(summary.result, ConquerResult::Sat);
    assert_eq!(summary.sat, 1);
    assert_eq!(summary.submitted, 0);
    assert_eq!(summary.witness.as_deref(), Some(witness.as_str()));
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
