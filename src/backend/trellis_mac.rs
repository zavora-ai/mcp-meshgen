//! TRELLIS.2 on Apple Silicon through trellis-mac (design §4.2; R4–R7).
//!
//! Runs `worker/trellis_mac_worker.py` (embedded here, written to the cache dir) with trellis-mac's venv python, in its
//! own process group, and talks JSON lines to it. The worker exits after each job unless the job keeps the model loaded
//! (A6); a kept worker is stopped by `unload_models` or after `idle_unload_s` idle seconds.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio_util::sync::CancellationToken;

use super::process::{kill_tree, own_process_group};
use super::{
    Availability, BackendError, BackendJob, GeneratedMesh, MeshBackend, ProgressSink, Stage,
};
use crate::config::TrellisMacConfig;
use crate::registry::ModelSpec;

pub const WORKER_SOURCE: &str = include_str!("../../worker/trellis_mac_worker.py");
pub const PROTOCOL: u64 = 1;
/// Printed by Metal when the macOS GPU watchdog kills a long kernel (trellis-mac's known failure).
pub const WATCHDOG_MARKER: &str = "kIOGPUCommandBufferCallbackErrorImpactingInteractivity";
const WATCHDOG_HINT: &str = "the macOS GPU watchdog killed a long Metal kernel. Close other GPU-heavy apps and external \
displays and retry, or use quality 512 / fewer steps (see trellis-mac's GPU watchdog notes)";
const LOG_TAIL_LINES: usize = 40;
const HELLO_TIMEOUT: Duration = Duration::from_secs(120);
const EXIT_TIMEOUT: Duration = Duration::from_secs(60);

/// How the worker is started: the embedded Python worker, or a stand-in program (tests: `fake-worker`).
#[derive(Debug, Clone)]
pub enum Launcher {
    Python,
    Program {
        path: PathBuf,
        env: Vec<(String, String)>,
    },
}

pub struct TrellisMacBackend {
    config: TrellisMacConfig,
    cache_dir: PathBuf,
    idle_unload_s: u64,
    launcher: Launcher,
    slot: Arc<tokio::sync::Mutex<Option<Worker>>>,
    loaded: Arc<Mutex<Option<String>>>,
    idle_generation: Arc<AtomicU64>,
}

struct Worker {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Lines<BufReader<ChildStdout>>,
    log: Arc<Mutex<VecDeque<String>>>,
    watchdog: Arc<AtomicBool>,
    pipeline: String,
}

#[derive(Debug, Deserialize)]
struct Event {
    event: String,
    #[serde(default)]
    id: String,
    #[serde(default)]
    protocol: u64,
    #[serde(default)]
    stage: String,
    #[serde(default)]
    step: u32,
    #[serde(default)]
    total: u32,
    #[serde(default)]
    kind: String,
    #[serde(default)]
    message: String,
    #[serde(default)]
    hint: String,
    #[serde(default)]
    timings: BTreeMap<String, f64>,
    #[serde(default)]
    peak_rss_mb: Option<u64>,
    #[serde(default)]
    fixes: Vec<String>,
    #[serde(default)]
    warnings: Vec<String>,
    #[serde(default)]
    details: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    files: Vec<PathBuf>,
}

impl TrellisMacBackend {
    pub fn new(config: TrellisMacConfig, cache_dir: PathBuf, idle_unload_s: u64) -> Self {
        Self::with_launcher(config, cache_dir, idle_unload_s, Launcher::Python)
    }

    pub fn with_launcher(
        config: TrellisMacConfig,
        cache_dir: PathBuf,
        idle_unload_s: u64,
        launcher: Launcher,
    ) -> Self {
        Self {
            config,
            cache_dir,
            idle_unload_s,
            launcher,
            slot: Arc::new(tokio::sync::Mutex::new(None)),
            loaded: Arc::new(Mutex::new(None)),
            idle_generation: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Availability rules (§4.1), with the platform passed in so every branch is testable.
    pub fn check(&self, apple_silicon: bool) -> Availability {
        let unavailable = |reason: String| Availability::Unavailable { reason };
        if matches!(self.launcher, Launcher::Python) && !apple_silicon {
            return unavailable(
                "trellis-mac needs Apple Silicon (macOS on arm64); on Linux or Windows with an NVIDIA GPU, TRELLIS.2 runs \
                 natively (a CUDA backend is planned)"
                    .into(),
            );
        }
        let Some(dir) = &self.config.dir else {
            return unavailable(
                "trellis-mac isn't configured: clone https://github.com/shivampkumar/trellis-mac, run its setup.sh, then \
                 set [backends.trellis_mac] dir in config.toml to the checkout"
                    .into(),
            );
        };
        if !dir.join("TRELLIS.2").is_dir() {
            return unavailable(format!(
                "{} is not a set-up trellis-mac checkout (no TRELLIS.2 folder); run its setup.sh there",
                dir.display()
            ));
        }
        if let Launcher::Program { path, .. } = &self.launcher {
            if !path.is_file() {
                return unavailable(format!("worker program {} not found", path.display()));
            }
            return Availability::Ready;
        }
        match self.config.python_path() {
            Some(python) if python.is_file() => Availability::Ready,
            Some(python) => unavailable(format!(
                "no venv python at {}; run setup.sh in the trellis-mac checkout (it creates .venv), or set \
                 [backends.trellis_mac] python",
                python.display()
            )),
            None => unavailable("no python configured for trellis-mac".into()),
        }
    }

    /// Write the embedded worker to `<cache>/worker/trellis_mac_worker-<hash>.py` (once per version).
    pub fn worker_script(&self) -> Result<PathBuf, BackendError> {
        let hash = format!("{:x}", Sha256::digest(WORKER_SOURCE.as_bytes()));
        let dir = self.cache_dir.join("worker");
        let path = dir.join(format!("trellis_mac_worker-{}.py", &hash[..12]));
        if path.is_file() {
            return Ok(path);
        }
        let io = |e: std::io::Error, p: &Path| {
            BackendError::failed(
                "starting",
                "cache_write",
                format!("{}: {e}", p.display()),
                "check the cache dir is writable",
            )
        };
        std::fs::create_dir_all(&dir).map_err(|e| io(e, &dir))?;
        // Unique per writer: two threads (or servers) may write the same script at once.
        let tmp = dir.join(format!(
            ".trellis_mac_worker-{}.{:016x}.tmp",
            &hash[..12],
            rand::random::<u64>()
        ));
        std::fs::write(&tmp, WORKER_SOURCE).map_err(|e| io(e, &tmp))?;
        std::fs::rename(&tmp, &path).map_err(|e| io(e, &path))?;
        Ok(path)
    }

    async fn spawn(&self, job: &BackendJob, pipeline: &str) -> Result<Worker, BackendError> {
        let Some(dir) = self.config.dir.clone() else {
            return Err(BackendError::Unavailable {
                reason: "trellis-mac isn't configured".into(),
            });
        };
        let mut cmd = match &self.launcher {
            Launcher::Python => {
                let python =
                    self.config
                        .python_path()
                        .ok_or_else(|| BackendError::Unavailable {
                            reason: "no python configured".into(),
                        })?;
                let mut cmd = Command::new(python);
                cmd.arg("-u").arg(self.worker_script()?).arg(&dir);
                cmd
            }
            Launcher::Program { path, env } => {
                let mut cmd = Command::new(path);
                cmd.arg(&dir).envs(env.iter().map(|(k, v)| (k, v)));
                cmd
            }
        };
        cmd.current_dir(&dir)
            .env("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            .env("PYTHONUNBUFFERED", "1")
            .env("HF_HUB_OFFLINE", "1")
            .env("HF_HUB_CACHE", &job.hf_cache)
            .env("TQDM_DISABLE", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        own_process_group(&mut cmd);
        let mut child = cmd.spawn().map_err(|e| {
            BackendError::failed(
                "starting",
                "spawn_failed",
                format!("couldn't start the worker: {e}"),
                "check the venv python runs",
            )
        })?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let log = Arc::new(Mutex::new(VecDeque::with_capacity(LOG_TAIL_LINES)));
        let watchdog = Arc::new(AtomicBool::new(false));
        tokio::spawn(read_log(stderr, log.clone(), watchdog.clone()));
        let mut worker = Worker {
            child,
            stdin,
            lines: BufReader::new(stdout).lines(),
            log,
            watchdog,
            pipeline: pipeline.to_string(),
        };
        match tokio::time::timeout(HELLO_TIMEOUT, next_event(&mut worker.lines)).await {
            Ok(Some(ev)) if ev.event == "hello" && ev.protocol == PROTOCOL => Ok(worker),
            Ok(Some(ev)) if ev.event == "hello" => {
                kill_tree(&mut worker.child).await;
                Err(BackendError::failed(
                    "starting",
                    "worker_protocol",
                    format!(
                        "the worker speaks protocol {}, this server {PROTOCOL}",
                        ev.protocol
                    ),
                    "the worker file and server come from different versions; restart the server",
                ))
            }
            Ok(_) => Err(self.died(&mut worker, "starting").await),
            Err(_) => {
                kill_tree(&mut worker.child).await;
                Err(BackendError::failed(
                    "starting",
                    "worker_timeout",
                    format!(
                        "the worker didn't start within {} s",
                        HELLO_TIMEOUT.as_secs()
                    ),
                    "check the trellis-mac venv imports (run its generate.py once by hand)",
                ))
            }
        }
    }

    /// The worker's stdout closed without an answer: report how it ended, with its log.
    async fn died(&self, worker: &mut Worker, stage: &str) -> BackendError {
        let status = tokio::time::timeout(Duration::from_secs(5), worker.child.wait()).await;
        let how = match status {
            Ok(Ok(s)) => describe_exit(&s),
            _ => {
                kill_tree(&mut worker.child).await;
                "stopped answering".to_string()
            }
        };
        let (kind, hint) = if worker.watchdog.load(Ordering::SeqCst) {
            ("gpu_watchdog", WATCHDOG_HINT.to_string())
        } else if how.contains("SIGKILL") {
            ("worker_killed", "macOS may have killed it for memory; close heavy apps (Unreal, Blender, other models) and retry".to_string())
        } else {
            ("worker_exited", "see the worker log tail below".to_string())
        };
        BackendError::Failed {
            stage: stage.to_string(),
            kind: kind.to_string(),
            message: format!("the worker {how}"),
            hint,
            log_tail: tail(&worker.log),
        }
    }

    /// After a job without keep_loaded: close stdin and let the worker exit so its memory is back before Blender runs.
    async fn retire(mut worker: Worker) {
        worker.stdin.take();
        if tokio::time::timeout(EXIT_TIMEOUT, worker.child.wait())
            .await
            .is_err()
        {
            kill_tree(&mut worker.child).await;
        }
    }

    fn schedule_idle_unload(&self) {
        if self.idle_unload_s == 0 {
            return;
        }
        let generation = self.idle_generation.fetch_add(1, Ordering::SeqCst) + 1;
        let (slot, loaded, counter) = (
            self.slot.clone(),
            self.loaded.clone(),
            self.idle_generation.clone(),
        );
        let wait = Duration::from_secs(self.idle_unload_s);
        tokio::spawn(async move {
            tokio::time::sleep(wait).await;
            if counter.load(Ordering::SeqCst) != generation {
                return;
            }
            let mut guard = slot.lock().await;
            if counter.load(Ordering::SeqCst) == generation
                && let Some(worker) = guard.take()
            {
                loaded.lock().unwrap().take();
                stop(worker).await;
            }
        });
    }
}

#[async_trait::async_trait]
impl MeshBackend for TrellisMacBackend {
    fn id(&self) -> &'static str {
        "trellis_mac"
    }

    async fn availability(&self) -> Availability {
        self.check(cfg!(all(target_os = "macos", target_arch = "aarch64")))
    }

    async fn generate(
        &self,
        job: &BackendJob,
        model: &ModelSpec,
        progress: ProgressSink,
        cancel: CancellationToken,
    ) -> Result<GeneratedMesh, BackendError> {
        if let Availability::Unavailable { reason } = self.availability().await {
            return Err(BackendError::Unavailable { reason });
        }
        let pipeline = model
            .pipeline_id()
            .ok_or_else(|| BackendError::Unavailable {
                reason: format!(
                    "model `{}` names no pipeline repo (set `pipeline` in models.toml)",
                    model.id
                ),
            })?;
        // Invalidate any pending idle unload while this job runs.
        self.idle_generation.fetch_add(1, Ordering::SeqCst);
        let mut slot = self.slot.lock().await;
        let reuse = match slot.as_mut() {
            Some(w) => w.pipeline == pipeline && w.child.try_wait().ok().flatten().is_none(),
            None => false,
        };
        let mut worker = match (reuse, slot.take()) {
            (true, Some(w)) => w,
            (_, old) => {
                if let Some(old) = old {
                    stop(old).await;
                }
                self.loaded.lock().unwrap().take();
                progress.report(Stage::Starting);
                tokio::select! {
                    _ = cancel.cancelled() => return Err(BackendError::Cancelled),
                    w = self.spawn(job, pipeline) => w?,
                }
            }
        };

        let r = &job.request;
        let request = serde_json::json!({
            "op": "generate",
            "id": job.id,
            "image": r.image,
            "pretrained": pipeline,
            "seed": r.seed,
            "quality": r.quality,
            "texture_size": r.texture_size,
            "steps": r.steps,
            "decimation_target": r.target_triangles,
            "out_glb": job.out_glb,
            "out_obj": job.out_obj,
        });
        let sent = match worker.stdin.as_mut() {
            Some(stdin) => stdin
                .write_all(format!("{request}\n").as_bytes())
                .await
                .and(stdin.flush().await),
            None => Err(std::io::Error::other("worker stdin closed")),
        };
        if sent.is_err() {
            return Err(self.died(&mut worker, "starting").await);
        }
        if !r.keep_loaded {
            // EOF after this job: the worker exits once it has answered.
            worker.stdin.take();
        }

        let outcome = loop {
            let event = tokio::select! {
                _ = cancel.cancelled() => {
                    kill_tree(&mut worker.child).await;
                    self.loaded.lock().unwrap().take();
                    return Err(BackendError::Cancelled);
                }
                ev = next_event(&mut worker.lines) => ev,
            };
            let Some(ev) = event else {
                let stage = "worker";
                break Err(self.died(&mut worker, stage).await);
            };
            if !ev.id.is_empty() && ev.id != job.id {
                continue;
            }
            match ev.event.as_str() {
                "loading" => {
                    *self.loaded.lock().unwrap() = Some(model.id.clone());
                    progress.report(Stage::Loading);
                }
                "progress" => {
                    if let Some(stage) = map_stage(&ev.stage, ev.step, ev.total) {
                        progress.report(stage);
                    }
                }
                "result" => {
                    break Ok(GeneratedMesh {
                        timings: ev.timings,
                        peak_rss_mb: ev.peak_rss_mb,
                        fixes: ev.fixes,
                        warnings: ev.warnings,
                        extra_files: ev.files,
                        details: ev.details,
                    });
                }
                "error" => {
                    let watchdog = worker.watchdog.load(Ordering::SeqCst)
                        && matches!(
                            ev.kind.as_str(),
                            "empty_mesh" | "empty_structure" | "worker_exception"
                        );
                    break Err(BackendError::Failed {
                        stage: if ev.stage.is_empty() {
                            "worker".into()
                        } else {
                            ev.stage
                        },
                        kind: if watchdog {
                            "gpu_watchdog".into()
                        } else {
                            ev.kind
                        },
                        message: ev.message,
                        hint: if watchdog {
                            WATCHDOG_HINT.into()
                        } else {
                            ev.hint
                        },
                        log_tail: tail(&worker.log),
                    });
                }
                _ => {}
            }
        };

        if r.keep_loaded && outcome.is_ok() && worker.child.try_wait().ok().flatten().is_none() {
            *self.loaded.lock().unwrap() = Some(model.id.clone());
            *slot = Some(worker);
            drop(slot);
            self.schedule_idle_unload();
        } else {
            self.loaded.lock().unwrap().take();
            drop(slot);
            Self::retire(worker).await;
        }
        outcome
    }

    async fn unload(&self) {
        self.idle_generation.fetch_add(1, Ordering::SeqCst);
        let worker = self.slot.lock().await.take();
        self.loaded.lock().unwrap().take();
        if let Some(worker) = worker {
            stop(worker).await;
        }
    }

    fn loaded_model(&self) -> Option<String> {
        self.loaded.lock().unwrap().clone()
    }
}

/// Ask a kept worker to unload and exit; kill it if it doesn't within a few seconds.
async fn stop(mut worker: Worker) {
    if let Some(stdin) = worker.stdin.as_mut() {
        let _ = stdin.write_all(b"{\"op\":\"unload\"}\n").await;
        let _ = stdin.flush().await;
    }
    worker.stdin.take();
    if tokio::time::timeout(Duration::from_secs(15), worker.child.wait())
        .await
        .is_err()
    {
        kill_tree(&mut worker.child).await;
    }
}

/// Next protocol event; non-JSON lines on stdout are skipped. `None` at EOF.
async fn next_event(lines: &mut Lines<BufReader<ChildStdout>>) -> Option<Event> {
    loop {
        let line = lines.next_line().await.ok()??;
        if let Ok(ev) = serde_json::from_str::<Event>(line.trim()) {
            return Some(ev);
        }
    }
}

/// Keep the last lines of the worker's stderr (splitting tqdm's `\r` redraws) and flag the watchdog marker.
async fn read_log(
    mut stderr: tokio::process::ChildStderr,
    log: Arc<Mutex<VecDeque<String>>>,
    watchdog: Arc<AtomicBool>,
) {
    let mut buf = vec![0u8; 8192];
    let mut pending = String::new();
    loop {
        let n = match stderr.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        pending.push_str(&String::from_utf8_lossy(&buf[..n]));
        while let Some(pos) = pending.find(['\n', '\r']) {
            let line: String = pending.drain(..=pos).collect();
            push_line(&log, &watchdog, line.trim_end());
        }
        if pending.len() > 4096 {
            let line = std::mem::take(&mut pending);
            push_line(&log, &watchdog, &line);
        }
    }
    if !pending.is_empty() {
        push_line(&log, &watchdog, pending.trim_end());
    }
}

fn push_line(log: &Mutex<VecDeque<String>>, watchdog: &AtomicBool, line: &str) {
    if line.trim().is_empty() {
        return;
    }
    if line.contains(WATCHDOG_MARKER) {
        watchdog.store(true, Ordering::SeqCst);
    }
    let mut log = log.lock().unwrap();
    if log.len() == LOG_TAIL_LINES {
        log.pop_front();
    }
    log.push_back(line.chars().take(500).collect());
}

fn tail(log: &Mutex<VecDeque<String>>) -> String {
    log.lock()
        .unwrap()
        .iter()
        .cloned()
        .collect::<Vec<_>>()
        .join("\n")
}

fn map_stage(stage: &str, step: u32, total: u32) -> Option<Stage> {
    Some(match stage {
        "sparse_structure" => Stage::SparseStructure { step, total },
        "shape" => Stage::Shape { step, total },
        "texture" => Stage::Texture { step, total },
        "decode" => Stage::Decode,
        "bake" => Stage::Bake,
        "export" => Stage::Export,
        _ => return None,
    })
}

fn describe_exit(status: &std::process::ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            let name = match sig {
                9 => "SIGKILL",
                6 => "SIGABRT",
                11 => "SIGSEGV",
                15 => "SIGTERM",
                _ => "a signal",
            };
            return format!("was killed by {name} ({sig})");
        }
    }
    match status.code() {
        Some(code) => format!("exited with code {code} before answering"),
        None => "exited before answering".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend(dir: Option<PathBuf>, python: Option<PathBuf>) -> TrellisMacBackend {
        TrellisMacBackend::new(TrellisMacConfig { dir, python }, std::env::temp_dir(), 0)
    }

    #[test]
    fn availability_rules() {
        let reason = |a: Availability| a.reason().unwrap_or("ready").to_string();
        assert!(reason(backend(None, None).check(false)).contains("needs Apple Silicon"));
        assert!(reason(backend(None, None).check(true)).contains("isn't configured"));
        let dir = tempfile::tempdir().unwrap();
        let b = backend(Some(dir.path().into()), None);
        assert!(reason(b.check(true)).contains("no TRELLIS.2 folder"));
        std::fs::create_dir_all(dir.path().join("TRELLIS.2")).unwrap();
        assert!(reason(b.check(true)).contains("no venv python"));
        let python = crate::config::venv_python(dir.path());
        std::fs::create_dir_all(python.parent().unwrap()).unwrap();
        std::fs::write(&python, b"").unwrap();
        assert_eq!(b.check(true), Availability::Ready);
    }

    #[test]
    fn worker_script_is_written_once_per_version() {
        let cache = tempfile::tempdir().unwrap();
        let b = TrellisMacBackend::new(TrellisMacConfig::default(), cache.path().into(), 0);
        let path = b.worker_script().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), WORKER_SOURCE);
        assert!(
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("trellis_mac_worker-")
        );
        assert_eq!(b.worker_script().unwrap(), path);
    }

    #[test]
    fn stage_names_map() {
        assert_eq!(
            map_stage("shape", 2, 12),
            Some(Stage::Shape { step: 2, total: 12 })
        );
        assert_eq!(map_stage("bake", 0, 0), Some(Stage::Bake));
        assert_eq!(map_stage("preprocess", 0, 0), None);
    }

    #[test]
    fn log_keeps_a_bounded_tail_and_spots_the_watchdog() {
        let (log, flag) = (Mutex::new(VecDeque::new()), AtomicBool::new(false));
        for i in 0..100 {
            push_line(&log, &flag, &format!("line {i}"));
        }
        assert_eq!(log.lock().unwrap().len(), LOG_TAIL_LINES);
        assert!(tail(&log).ends_with("line 99"));
        assert!(!flag.load(Ordering::SeqCst));
        push_line(&log, &flag, &format!("Error: {WATCHDOG_MARKER} (00000004)"));
        assert!(flag.load(Ordering::SeqCst));
    }

    #[test]
    fn the_embedded_worker_speaks_this_protocol() {
        assert!(WORKER_SOURCE.contains(&format!("PROTOCOL = {PROTOCOL}")));
        assert!(WORKER_SOURCE.contains("mps_sparse_decoder_cpu"));
    }
}
