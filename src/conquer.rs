//! Bounded streaming conquer pool for Cube-and-Conquer frontiers.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::termination::TerminationSignal;

#[derive(Debug, thiserror::Error)]
pub enum ConquerError {
    #[error("read CNF {path}: {source}")]
    ReadCnf {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("CNF has no valid 'p cnf <variables> <clauses>' header")]
    MissingHeader,
    #[error("start Kissat: {0}")]
    StartKissat(std::io::Error),
    #[error("write CNF to Kissat: {0}")]
    WriteKissat(std::io::Error),
    #[error("read Kissat output: {0}")]
    ReadKissat(std::io::Error),
    #[error("streaming conquer worker disconnected")]
    Disconnected,
    #[error("streaming conquer worker panicked")]
    WorkerPanicked,
}

#[derive(Clone)]
struct CnfTemplate {
    variables: usize,
    clauses: usize,
    body: Arc<Vec<u8>>,
}

impl CnfTemplate {
    fn read(path: &Path) -> Result<Self, ConquerError> {
        let text = fs::read_to_string(path).map_err(|source| ConquerError::ReadCnf {
            path: path.to_owned(),
            source,
        })?;
        let mut header = None;
        let mut body = Vec::with_capacity(text.len());
        for line in text.split_inclusive('\n') {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.first() == Some(&"p") {
                if fields.len() != 4 || fields[1] != "cnf" {
                    return Err(ConquerError::MissingHeader);
                }
                let variables = fields[2].parse().map_err(|_| ConquerError::MissingHeader)?;
                let clauses = fields[3].parse().map_err(|_| ConquerError::MissingHeader)?;
                if header.replace((variables, clauses)).is_some() {
                    return Err(ConquerError::MissingHeader);
                }
            } else {
                body.extend_from_slice(line.as_bytes());
            }
        }
        let (variables, clauses) = header.ok_or(ConquerError::MissingHeader)?;
        Ok(Self {
            variables,
            clauses,
            body: Arc::new(body),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConquerResult {
    Sat,
    Unsat,
    Incomplete,
}

#[derive(Debug)]
pub struct ConquerSummary {
    pub result: ConquerResult,
    pub submitted: usize,
    pub sat: usize,
    pub unsat: usize,
    pub errors: usize,
    pub witness: Option<String>,
}

struct Shared {
    stopped: TerminationSignal,
    submitted: AtomicUsize,
    sat: AtomicUsize,
    unsat: AtomicUsize,
    errors: AtomicUsize,
    witness: Mutex<Option<String>>,
}

impl Shared {
    fn new() -> Self {
        Self {
            stopped: TerminationSignal::new(),
            submitted: AtomicUsize::new(0),
            sat: AtomicUsize::new(0),
            unsat: AtomicUsize::new(0),
            errors: AtomicUsize::new(0),
            witness: Mutex::new(None),
        }
    }
}

/// A fixed-size Kissat pool fed directly by the cuber's leaf callback.
pub struct StreamingConquer {
    sender: Option<mpsc::SyncSender<Vec<i64>>>,
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl StreamingConquer {
    pub fn start(cnf: &Path, kissat: &Path, workers: usize) -> Result<Self, ConquerError> {
        assert!(workers > 0, "workers must be positive");
        let template = CnfTemplate::read(cnf)?;
        let shared = Arc::new(Shared::new());
        let (sender, receiver) = mpsc::sync_channel(workers);
        let receiver = Arc::new(Mutex::new(receiver));
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let receiver = Arc::clone(&receiver);
            let shared = Arc::clone(&shared);
            let template = template.clone();
            let kissat = kissat.to_owned();
            handles.push(thread::spawn(move || {
                worker_loop(receiver, shared, template, kissat)
            }));
        }
        Ok(Self {
            sender: Some(sender),
            shared,
            workers: handles,
        })
    }

    /// Submit one open cube. Returns `false` once another cube has proved SAT.
    pub fn submit(&self, cube: Vec<i64>) -> Result<bool, ConquerError> {
        if self.shared.stopped.is_requested() {
            return Ok(false);
        }
        let sent = self
            .sender
            .as_ref()
            .ok_or(ConquerError::Disconnected)?
            .send(cube);
        if sent.is_err() {
            return if self.shared.stopped.is_requested() {
                Ok(false)
            } else {
                Err(ConquerError::Disconnected)
            };
        }
        self.shared.submitted.fetch_add(1, Ordering::Relaxed);
        Ok(!self.shared.stopped.is_requested())
    }

    /// Clone the global first-answer signal so the cuber can stop at its own
    /// safe boundaries when a conquer worker wins.
    pub fn termination_signal(&self) -> TerminationSignal {
        self.shared.stopped.clone()
    }

    /// Record a satisfying leaf found by the cuber itself.
    pub fn mark_sat(&self) {
        self.shared.sat.fetch_add(1, Ordering::Relaxed);
        self.shared.stopped.request();
    }

    /// Stop all conquer workers after an integrated solver found a model.
    pub fn mark_sat_with_witness(&self, witness: String) {
        *self.shared.witness.lock().expect("witness lock") = Some(witness);
        self.mark_sat();
    }

    pub fn finish(mut self, cubing_complete: bool) -> Result<ConquerSummary, ConquerError> {
        self.close_and_join(false)?;
        let submitted = self.shared.submitted.load(Ordering::Relaxed);
        let sat = self.shared.sat.load(Ordering::Relaxed);
        let unsat = self.shared.unsat.load(Ordering::Relaxed);
        let errors = self.shared.errors.load(Ordering::Relaxed);
        let result = if sat > 0 {
            ConquerResult::Sat
        } else if cubing_complete && errors == 0 && unsat == submitted {
            ConquerResult::Unsat
        } else {
            ConquerResult::Incomplete
        };
        let witness = self.shared.witness.lock().expect("witness lock").take();
        Ok(ConquerSummary {
            result,
            submitted,
            sat,
            unsat,
            errors,
            witness,
        })
    }

    fn close_and_join(&mut self, request_stop: bool) -> Result<(), ConquerError> {
        if request_stop {
            self.shared.stopped.request();
        }
        self.sender.take();
        let mut worker_panicked = false;
        for worker in self.workers.drain(..) {
            worker_panicked |= worker.join().is_err();
        }
        if worker_panicked {
            Err(ConquerError::WorkerPanicked)
        } else {
            Ok(())
        }
    }
}

impl Drop for StreamingConquer {
    fn drop(&mut self) {
        // Error paths must not detach workers or leave their Kissat children
        // behind. Each worker observes this signal in its polling loop, kills
        // its active child, and is joined here before the pool disappears.
        let _ = self.close_and_join(true);
    }
}

fn worker_loop(
    receiver: Arc<Mutex<mpsc::Receiver<Vec<i64>>>>,
    shared: Arc<Shared>,
    template: CnfTemplate,
    kissat: PathBuf,
) {
    loop {
        if shared.stopped.is_requested() {
            break;
        }
        let cube = match receiver.lock().expect("cube receiver lock").recv() {
            Ok(cube) => cube,
            Err(_) => break,
        };
        if shared.stopped.is_requested() {
            break;
        }
        match solve_cube(&template, &kissat, &shared, &cube) {
            Ok(CubeResult::Sat(output)) => {
                shared.sat.fetch_add(1, Ordering::Relaxed);
                *shared.witness.lock().expect("witness lock") = Some(output);
                shared.stopped.request();
            }
            Ok(CubeResult::Unsat) => {
                shared.unsat.fetch_add(1, Ordering::Relaxed);
            }
            Ok(CubeResult::Cancelled) => {}
            Ok(CubeResult::Unknown) | Err(_) => {
                shared.errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

enum CubeResult {
    Sat(String),
    Unsat,
    Cancelled,
    Unknown,
}

fn solve_cube(
    template: &CnfTemplate,
    kissat: &Path,
    shared: &Shared,
    cube: &[i64],
) -> Result<CubeResult, ConquerError> {
    if cube
        .iter()
        .any(|literal| literal.unsigned_abs() as usize > template.variables)
    {
        return Ok(CubeResult::Unknown);
    }
    let mut child = Command::new(kissat)
        .arg("--relaxed")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(ConquerError::StartKissat)?;
    let stdout = child.stdout.take().expect("piped Kissat stdout");
    let reader = thread::spawn(move || read_solution(BufReader::new(stdout)));
    let write_result = (|| {
        let mut stdin = child.stdin.take().expect("piped Kissat stdin");
        writeln!(
            stdin,
            "p cnf {} {}",
            template.variables,
            template.clauses + cube.len()
        )
        .and_then(|_| stdin.write_all(&template.body))
        .map_err(ConquerError::WriteKissat)?;
        if !template.body.ends_with(b"\n") {
            stdin.write_all(b"\n").map_err(ConquerError::WriteKissat)?;
        }
        for literal in cube {
            writeln!(stdin, "{literal} 0").map_err(ConquerError::WriteKissat)?;
        }
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = child.kill();
        let _ = child.wait();
        let _ = reader.join();
        return Err(error);
    }

    let status = loop {
        if shared.stopped.is_requested() {
            let _ = child.kill();
            let _ = child.wait();
            reader.join().expect("Kissat output reader panicked")?;
            return Ok(CubeResult::Cancelled);
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                return Err(ConquerError::StartKissat(error));
            }
        }
        thread::sleep(std::time::Duration::from_millis(5));
    };
    let solution = reader.join().expect("Kissat output reader panicked")?;
    match status.code() {
        Some(10) => Ok(CubeResult::Sat(solution)),
        Some(20) => Ok(CubeResult::Unsat),
        _ => Ok(CubeResult::Unknown),
    }
}

fn read_solution(reader: impl BufRead) -> Result<String, ConquerError> {
    let mut solution = String::new();
    for line in reader.lines() {
        let line = line.map_err(ConquerError::ReadKissat)?;
        if line.starts_with("s ") || line.starts_with("v ") {
            solution.push_str(&line);
            solution.push('\n');
        }
    }
    Ok(solution)
}
