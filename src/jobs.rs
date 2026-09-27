//! Job manager (design §5–§7; R5, R6).
//!
//! One local lane: a bounded FIFO queue and one runner, so conversions never overlap. A job takes the cross-process
//! GPU lock, passes the memory preflight, reserves its file names, runs the backend (which writes `*.partial` files),
//! renames the partials, runs previews and the engine copy after the worker has exited, writes the sidecar, and only
//! then releases the lock. Cancel or failure deletes every partial. Each job has a `watch` channel with its latest
//! snapshot, so callers can wait, poll or stream progress; finished jobs are kept for `job_ttl`.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use tokio::sync::{Notify, watch};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::backend::{
    BackendError, BackendJob, BackendRegistry, GeneratedMesh, ProgressSink, Stage,
};
use crate::error::MeshGenError;
use crate::glb::MeshStats;
use crate::gpu_lock::{GpuLock, Holder};
use crate::heavy::{ProcessLister, advice, census};
use crate::input::InputReport;
use crate::memory::{MemoryProbe, preflight};
use crate::output::{self, MeshNames, MeshTarget};
use crate::post::{GameCopy, PostProcessor};
use crate::registry::{Licence, ModelSpec};
use crate::request::ResolvedRequest;

/// What the job manager needs from the rest of the server.
#[derive(Clone)]
pub struct JobDeps {
    pub backends: BackendRegistry,
    pub post: Option<Arc<dyn PostProcessor>>,
    pub gpu_lock: GpuLock,
    pub memory: Arc<dyn MemoryProbe>,
    pub processes: Arc<dyn ProcessLister>,
    pub heavy_processes: Vec<String>,
    pub memory_headroom_mb: u64,
    pub hf_cache: PathBuf,
    pub max_queue: usize,
    pub job_ttl: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Done | Self::Failed | Self::Cancelled)
    }
}

/// Provenance recorded in the result and the `<mesh>.json` sidecar (§5).
#[derive(Debug, Clone, Serialize)]
pub struct MeshRecord {
    pub model: String,
    pub backend: String,
    pub runs: String,
    /// The resolved request: seed, quality, texture size, steps, triangle budget, size, outputs.
    pub params: ResolvedRequest,
    pub input: InputReport,
    pub stats: MeshStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub game: Option<GameCopy>,
    /// Seconds per stage, plus `total` for the whole job.
    pub timings: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_rss_mb: Option<u64>,
    pub fixes: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    pub licences: Vec<Licence>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub backend_details: BTreeMap<String, serde_json::Value>,
    pub generator: String,
}

/// Files a finished job wrote.
#[derive(Debug, Clone, Default, Serialize)]
pub struct MeshFiles {
    pub glb: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub obj: Option<PathBuf>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub obj_extras: Vec<PathBuf>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub previews: Vec<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub game_glb: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub game_fbx: Option<PathBuf>,
    pub sidecar: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobResult {
    pub job_id: String,
    pub files: MeshFiles,
    #[serde(flatten)]
    pub record: MeshRecord,
    #[serde(skip)]
    pub preview_png: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobError {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobSnapshot {
    pub job_id: String,
    pub status: JobStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<Stage>,
    /// Human-readable progress, e.g. "shape 3/12".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<JobResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JobError>,
}

/// A validated job ready to run.
#[derive(Debug, Clone)]
pub struct JobRequest {
    pub model: ModelSpec,
    pub request: ResolvedRequest,
    pub target: MeshTarget,
    pub input: InputReport,
}

struct JobEntry {
    tx: watch::Sender<JobSnapshot>,
    cancel: CancellationToken,
    request: Mutex<Option<JobRequest>>,
    finished_at: Mutex<Option<Instant>>,
}

impl JobEntry {
    fn snapshot(&self) -> JobSnapshot {
        self.tx.borrow().clone()
    }

    fn set_stage(&self, stage: Stage) {
        self.tx.send_modify(|s| {
            s.progress_text = Some(stage.to_string());
            s.progress = Some(stage);
        });
    }

    fn finish(&self, status: JobStatus, result: Option<JobResult>, error: Option<JobError>) {
        self.tx.send_modify(|s| {
            s.status = status;
            s.result = result;
            s.error = error;
            s.progress = None;
            s.progress_text = None;
        });
        *self.finished_at.lock().unwrap() = Some(Instant::now());
    }

    fn fail(&self, err: &MeshGenError) {
        if matches!(err, MeshGenError::Cancelled) {
            self.finish(JobStatus::Cancelled, None, None);
        } else {
            self.finish(JobStatus::Failed, None, Some(to_job_error(err)));
        }
    }
}

struct Inner {
    jobs: Mutex<HashMap<String, Arc<JobEntry>>>,
    queue: Mutex<VecDeque<String>>,
    running: Mutex<Option<String>>,
    wake: Notify,
    deps: JobDeps,
}

#[derive(Clone)]
pub struct JobManager {
    inner: Arc<Inner>,
}

impl JobManager {
    /// Create the manager and spawn the lane's runner on the current tokio runtime.
    pub fn new(deps: JobDeps) -> Self {
        let inner = Arc::new(Inner {
            jobs: Mutex::new(HashMap::new()),
            queue: Mutex::new(VecDeque::new()),
            running: Mutex::new(None),
            wake: Notify::new(),
            deps,
        });
        tokio::spawn(runner(Arc::downgrade(&inner)));
        Self { inner }
    }

    pub fn deps(&self) -> &JobDeps {
        &self.inner.deps
    }

    /// Queue a job. Fails with `QueueFull` when `max_queue` jobs are already waiting.
    pub fn submit(&self, job: JobRequest) -> Result<JobSnapshot, MeshGenError> {
        self.prune();
        let id = new_job_id();
        let mut queue = self.inner.queue.lock().unwrap();
        let max = self.inner.deps.max_queue;
        if queue.len() >= max {
            return Err(MeshGenError::QueueFull { max });
        }
        let stage = Stage::Queued {
            position: queue.len() + 1,
        };
        let snapshot = JobSnapshot {
            job_id: id.clone(),
            status: JobStatus::Queued,
            progress_text: Some(stage.to_string()),
            progress: Some(stage),
            result: None,
            error: None,
        };
        let (tx, _) = watch::channel(snapshot.clone());
        let entry = Arc::new(JobEntry {
            tx,
            cancel: CancellationToken::new(),
            request: Mutex::new(Some(job)),
            finished_at: Mutex::new(None),
        });
        self.inner.jobs.lock().unwrap().insert(id.clone(), entry);
        queue.push_back(id);
        drop(queue);
        self.inner.wake.notify_one();
        Ok(snapshot)
    }

    pub fn get(&self, id: &str) -> Result<JobSnapshot, MeshGenError> {
        self.prune();
        Ok(self.entry(id)?.snapshot())
    }

    pub fn subscribe(&self, id: &str) -> Result<watch::Receiver<JobSnapshot>, MeshGenError> {
        Ok(self.entry(id)?.tx.subscribe())
    }

    /// Wait until the job reaches a terminal state.
    pub async fn wait(&self, id: &str) -> Result<JobSnapshot, MeshGenError> {
        let mut rx = self.subscribe(id)?;
        loop {
            let snap = rx.borrow_and_update().clone();
            if snap.status.is_terminal() {
                return Ok(snap);
            }
            if rx.changed().await.is_err() {
                return Err(MeshGenError::UnknownJob(id.to_string()));
            }
        }
    }

    /// Cancel a queued job immediately, or signal a running one. Idempotent on finished jobs.
    pub fn cancel(&self, id: &str) -> Result<JobSnapshot, MeshGenError> {
        let entry = self.entry(id)?;
        let removed = {
            let mut queue = self.inner.queue.lock().unwrap();
            let before = queue.len();
            queue.retain(|q| q != id);
            before != queue.len()
        };
        entry.cancel.cancel();
        if removed {
            entry.request.lock().unwrap().take();
            entry.finish(JobStatus::Cancelled, None, None);
            self.inner.renumber_queue();
        }
        Ok(entry.snapshot())
    }

    /// Cancel every queued and running job (server shutdown).
    pub fn cancel_all(&self) {
        let ids: Vec<String> = self.inner.jobs.lock().unwrap().keys().cloned().collect();
        for id in ids {
            let _ = self.cancel(&id);
        }
    }

    /// Jobs queued plus the one running.
    pub fn busy(&self) -> usize {
        self.inner.queue.lock().unwrap().len()
            + usize::from(self.inner.running.lock().unwrap().is_some())
    }

    /// Stop kept-alive workers now. Refused while a job is queued or running.
    pub async fn unload_all(&self) -> Result<Vec<String>, MeshGenError> {
        let busy = self.busy();
        if busy > 0 {
            return Err(MeshGenError::invalid(
                "unload_models",
                format!("{busy} job(s) queued or running; cancel them first or wait"),
            ));
        }
        let mut unloaded = Vec::new();
        for backend in self.inner.deps.backends.all() {
            if let Some(model) = backend.loaded_model() {
                backend.unload().await;
                unloaded.push(model);
            }
        }
        Ok(unloaded)
    }

    fn entry(&self, id: &str) -> Result<Arc<JobEntry>, MeshGenError> {
        self.inner
            .jobs
            .lock()
            .unwrap()
            .get(id)
            .cloned()
            .ok_or_else(|| MeshGenError::UnknownJob(id.to_string()))
    }

    fn prune(&self) {
        let ttl = self.inner.deps.job_ttl;
        self.inner.jobs.lock().unwrap().retain(|_, e| {
            e.finished_at
                .lock()
                .unwrap()
                .is_none_or(|t| t.elapsed() < ttl)
        });
    }
}

impl Inner {
    fn renumber_queue(&self) {
        let queue = self.queue.lock().unwrap();
        let jobs = self.jobs.lock().unwrap();
        for (i, id) in queue.iter().enumerate() {
            if let Some(e) = jobs.get(id) {
                e.set_stage(Stage::Queued { position: i + 1 });
            }
        }
    }

    fn next_job(&self) -> Option<(String, Arc<JobEntry>)> {
        let id = self.queue.lock().unwrap().pop_front()?;
        let entry = self.jobs.lock().unwrap().get(&id).cloned()?;
        *self.running.lock().unwrap() = Some(id.clone());
        self.renumber_queue();
        Some((id, entry))
    }

    async fn run(&self, id: &str, entry: &JobEntry) {
        let Some(job) = entry.request.lock().unwrap().take() else {
            return;
        };
        if entry.cancel.is_cancelled() {
            entry.finish(JobStatus::Cancelled, None, None);
            return;
        }
        entry.tx.send_modify(|s| s.status = JobStatus::Running);
        let started = Instant::now();

        // 1. The cross-process GPU lock, held to the end of the job (A2).
        let holder = Holder::this_process(Some(id));
        let lock = self
            .deps
            .gpu_lock
            .acquire(&holder, &entry.cancel, |current| {
                entry.set_stage(Stage::WaitingForGpu {
                    holder: current.map(|h| h.describe()),
                });
            })
            .await;
        let _lock = match lock {
            Ok(guard) => guard,
            Err(err) => return entry.fail(&err),
        };

        let backend_id = job.model.backend.clone();
        let Some(backend) = self.deps.backends.get(&backend_id) else {
            let reason = self.deps.backends.availability_of(&backend_id).await;
            return entry.fail(&MeshGenError::BackendUnavailable {
                backend: backend_id,
                reason: reason.reason().unwrap_or("not available").to_string(),
            });
        };

        // 2. Memory preflight, unless the model is already resident in a kept worker (R6).
        if backend.loaded_model().as_deref() != Some(job.model.id.as_str()) {
            let deps = &self.deps;
            if let Err(err) = preflight(
                &job.model,
                deps.memory_headroom_mb,
                deps.memory.as_ref(),
                || advice(&census(&deps.processes.processes(), &deps.heavy_processes)),
            ) {
                return entry.fail(&err);
            }
        }

        // 3. Reserve the names; from here on every exit path cleans the partials.
        let names = match output::reserve(&job.target) {
            Ok(names) => names,
            Err(err) => return entry.fail(&err),
        };
        match self
            .convert(id, entry, &job, &names, backend, started)
            .await
        {
            Ok(result) => {
                names.remove_partials();
                entry.finish(JobStatus::Done, Some(result), None);
            }
            Err(err) => {
                names.remove_partials();
                entry.fail(&err);
            }
        }
    }

    async fn convert(
        &self,
        id: &str,
        entry: &JobEntry,
        job: &JobRequest,
        names: &MeshNames,
        backend: Arc<dyn crate::backend::MeshBackend>,
        started: Instant,
    ) -> Result<JobResult, MeshGenError> {
        let req = &job.request;
        let backend_job = BackendJob {
            id: id.to_string(),
            request: req.clone(),
            out_glb: names.partial(&names.glb()),
            out_obj: req.obj.then(|| names.partial(&names.obj())),
            hf_cache: self.deps.hf_cache.clone(),
        };
        let tx = entry.tx.clone();
        let progress = ProgressSink::new(move |stage: Stage| {
            tx.send_modify(|s| {
                s.progress_text = Some(stage.to_string());
                s.progress = Some(stage);
            });
        });
        let generated: GeneratedMesh = match backend
            .generate(&backend_job, &job.model, progress, entry.cancel.clone())
            .await
        {
            _ if entry.cancel.is_cancelled() => return Err(MeshGenError::Cancelled),
            Err(BackendError::Cancelled) => return Err(MeshGenError::Cancelled),
            Err(e) => return Err(e.into_error(backend.id())),
            Ok(mesh) => mesh,
        };

        entry.set_stage(Stage::Saving);
        let mut files = MeshFiles {
            glb: names.finalize(&names.glb())?,
            sidecar: names.sidecar(),
            ..Default::default()
        };
        if req.obj {
            files.obj = Some(names.finalize(&names.obj())?);
            for extra in &generated.extra_files {
                files.obj_extras.push(finalize_extra(names, extra)?);
            }
        }
        let stats = crate::glb::stats(&files.glb).map_err(|e| MeshGenError::io(&files.glb, e))?;
        let mut warnings = req.warnings.clone();
        warnings.extend(generated.warnings.iter().cloned());
        let mut timings = generated.timings.clone();

        // 4. Previews and the engine copy, after the worker has exited (R8, R9).
        let mut game = None;
        if req.needs_blender() {
            match self.deps.post.as_ref().map(|p| (p, p.availability())) {
                Some((post, Ok(()))) => {
                    if req.previews {
                        entry.set_stage(Stage::Previews);
                        let t0 = Instant::now();
                        match post.previews(&files.glb, names, &entry.cancel).await {
                            Ok(paths) => {
                                for p in paths {
                                    files.previews.push(names.finalize(&p)?);
                                }
                            }
                            Err(e) if entry.cancel.is_cancelled() => {
                                let _ = e;
                                return Err(MeshGenError::Cancelled);
                            }
                            Err(e) => warnings.push(format!("previews failed: {e}")),
                        }
                        timings.insert("previews".into(), t0.elapsed().as_secs_f64());
                    }
                    if req.wants_game_copy() {
                        entry.set_stage(Stage::PostProcess);
                        let t0 = Instant::now();
                        match post.game_copy(&files.glb, names, req, &entry.cancel).await {
                            Ok(copy) => {
                                files.game_glb = Some(names.finalize(&copy.glb)?);
                                if let Some(fbx) = &copy.fbx {
                                    files.game_fbx = Some(names.finalize(fbx)?);
                                }
                                game = Some(copy);
                            }
                            Err(_) if entry.cancel.is_cancelled() => {
                                return Err(MeshGenError::Cancelled);
                            }
                            Err(e) => warnings.push(format!("engine copy failed: {e}")),
                        }
                        timings.insert("post_process".into(), t0.elapsed().as_secs_f64());
                    }
                }
                Some((_, Err(reason))) => warnings.push(format!(
                    "previews and the engine copy are unavailable ({reason})"
                )),
                None => warnings.push(
                    "previews and the engine copy are unavailable (no post-processor)".into(),
                ),
            }
        }
        timings.insert("total".into(), started.elapsed().as_secs_f64());

        let record = MeshRecord {
            model: job.model.id.clone(),
            backend: backend.id().to_string(),
            runs: job.model.runs.clone(),
            params: req.clone(),
            input: job.input.clone(),
            stats,
            game,
            timings,
            peak_rss_mb: generated.peak_rss_mb,
            fixes: generated.fixes.clone(),
            warnings,
            licences: job.model.licence.clone(),
            backend_details: generated.details.clone(),
            generator: format!("mcp-meshgen {}", env!("CARGO_PKG_VERSION")),
        };
        output::write_sidecar(
            &files.sidecar,
            &Sidecar {
                files: relative_files(&files),
                record: &record,
            },
        )?;
        let preview_png = if req.return_preview {
            files
                .previews
                .iter()
                .find(|p| p.to_string_lossy().contains("three_quarter"))
                .or(files.previews.first())
                .and_then(|p| std::fs::read(p).ok())
                .and_then(|png| output::make_preview(&png).ok())
        } else {
            None
        };
        Ok(JobResult {
            job_id: id.to_string(),
            files,
            record,
            preview_png,
        })
    }
}

/// The sidecar: the record plus file names relative to the sidecar, so a moved folder stays self-describing.
#[derive(Serialize)]
struct Sidecar<'a> {
    files: BTreeMap<String, serde_json::Value>,
    #[serde(flatten)]
    record: &'a MeshRecord,
}

fn relative_files(files: &MeshFiles) -> BTreeMap<String, serde_json::Value> {
    let name = |p: &Path| {
        serde_json::Value::from(
            p.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
        )
    };
    let mut out = BTreeMap::new();
    out.insert("glb".into(), name(&files.glb));
    if let Some(obj) = &files.obj {
        out.insert("obj".into(), name(obj));
    }
    if !files.obj_extras.is_empty() {
        out.insert(
            "obj_extras".into(),
            files.obj_extras.iter().map(|p| name(p)).collect(),
        );
    }
    if !files.previews.is_empty() {
        out.insert(
            "previews".into(),
            files.previews.iter().map(|p| name(p)).collect(),
        );
    }
    if let Some(g) = &files.game_glb {
        out.insert("game_glb".into(), name(g));
    }
    if let Some(f) = &files.game_fbx {
        out.insert("game_fbx".into(), name(f));
    }
    out
}

/// Rename an OBJ extra (`.mtl`, textures) from its partial, only when it belongs to this job's stem.
fn finalize_extra(names: &MeshNames, extra: &Path) -> Result<PathBuf, MeshGenError> {
    let file_name = extra
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let inside = extra.parent() == Some(names.dir.as_path())
        && (file_name.starts_with(&format!("{}.", names.stem))
            || file_name.starts_with(&format!("{}_", names.stem)));
    if !inside {
        return Err(MeshGenError::io(
            extra,
            "the backend reported a file outside this job's names",
        ));
    }
    names.finalize(extra)
}

async fn runner(inner: std::sync::Weak<Inner>) {
    loop {
        let Some(this) = inner.upgrade() else { return };
        if let Some((id, entry)) = this.next_job() {
            this.run(&id, &entry).await;
            *this.running.lock().unwrap() = None;
            continue;
        }
        // `notify_one` stores a permit, so a job submitted between the check above and this wait isn't missed.
        this.wake.notified().await;
    }
}

fn to_job_error(err: &MeshGenError) -> JobError {
    JobError {
        code: err.code().to_string(),
        message: err.to_string(),
    }
}

fn new_job_id() -> String {
    format!("job_{:016x}", rand::random::<u64>())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::backend::MeshBackend;
    use crate::backend::mock::MockBackend;
    use crate::config::MODELS_EXAMPLE;
    use crate::heavy::{FixedProcesses, ProcessInfo};
    use crate::memory::FixedMemory;
    use crate::post::MockPost;
    use crate::registry::ModelRegistry;
    use crate::request::SizeAxis;

    struct Rig {
        dir: tempfile::TempDir,
        jobs: JobManager,
        backend: Arc<MockBackend>,
    }

    fn model() -> ModelSpec {
        let mut m = ModelRegistry::from_toml(MODELS_EXAMPLE)
            .unwrap()
            .get("trellis2-4b")
            .unwrap()
            .clone();
        m.backend = "mock".into();
        m
    }

    fn rig_with(memory_mb: u64, post: Option<MockPost>, step: Duration) -> Rig {
        let dir = tempfile::tempdir().unwrap();
        let backend = Arc::new(MockBackend::new(step));
        let mut backends = BackendRegistry::default();
        backends.insert(backend.clone());
        let jobs = JobManager::new(JobDeps {
            backends,
            post: post.map(|p| Arc::new(p) as Arc<dyn PostProcessor>),
            gpu_lock: GpuLock::new(dir.path().join("lock/local-gpu.lock")),
            memory: Arc::new(FixedMemory(memory_mb)),
            processes: Arc::new(FixedProcesses(vec![ProcessInfo {
                name: "UnrealEditor".into(),
                resident_mb: 6100,
            }])),
            heavy_processes: vec!["UnrealEditor".into()],
            memory_headroom_mb: 2000,
            hf_cache: dir.path().join("hub"),
            max_queue: 2,
            job_ttl: Duration::from_secs(3600),
        });
        std::fs::create_dir_all(dir.path().join("out")).unwrap();
        Rig { dir, jobs, backend }
    }

    fn rig() -> Rig {
        rig_with(
            64_000,
            Some(MockPost {
                available: true,
                fail: false,
            }),
            Duration::from_millis(2),
        )
    }

    fn request(rig: &Rig, stem: &str) -> JobRequest {
        let input = rig.dir.path().join("in.png");
        JobRequest {
            model: model(),
            request: ResolvedRequest {
                model: "trellis2-4b".into(),
                image: input.clone(),
                seed: 7,
                quality: "512".into(),
                texture_size: 1024,
                steps: Some(2),
                target_triangles: 200_000,
                obj: false,
                previews: false,
                return_preview: false,
                size_m: None,
                size_axis: SizeAxis::Longest,
                fbx: false,
                keep_loaded: false,
                warnings: vec![],
            },
            target: MeshTarget {
                dir: rig.dir.path().join("out"),
                stem: stem.into(),
            },
            input: InputReport {
                path: input,
                format: "png".into(),
                width: 64,
                height: 64,
                sha256: "ab".repeat(32),
                transparent: 0.4,
                opaque: 0.6,
            },
        }
    }

    fn partials(dir: &Path) -> Vec<String> {
        std::fs::read_dir(dir.join("out"))
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".partial"))
            .collect()
    }

    #[tokio::test]
    async fn a_job_finalizes_its_files_and_writes_the_sidecar() {
        let rig = rig();
        let id = rig.jobs.submit(request(&rig, "crate")).unwrap().job_id;
        let snap = rig.jobs.wait(&id).await.unwrap();
        assert_eq!(snap.status, JobStatus::Done, "{:?}", snap.error);
        let result = snap.result.unwrap();
        assert_eq!(result.files.glb, rig.dir.path().join("out/crate.glb"));
        assert_eq!(result.record.stats.triangles, 12);
        assert_eq!(result.record.fixes, vec!["mps_sparse_decoder_cpu"]);
        assert!(result.record.timings.contains_key("total"));
        assert!(
            result
                .record
                .licences
                .iter()
                .any(|l| l.component.contains("DINOv3"))
        );
        let sidecar: serde_json::Value =
            serde_json::from_slice(&std::fs::read(rig.dir.path().join("out/crate.json")).unwrap())
                .unwrap();
        assert_eq!(sidecar["files"]["glb"], "crate.glb");
        assert_eq!(sidecar["input"]["sha256"], "ab".repeat(32));
        assert_eq!(sidecar["params"]["seed"], 7);
        assert_eq!(sidecar["stats"]["triangles"], 12);
        assert_eq!(sidecar["backend_details"]["voxels"], 4096);
        assert!(partials(rig.dir.path()).is_empty());
    }

    #[tokio::test]
    async fn jobs_run_one_at_a_time_and_never_share_names() {
        let rig = rig();
        let a = rig.jobs.submit(request(&rig, "same")).unwrap().job_id;
        let b = rig.jobs.submit(request(&rig, "same")).unwrap().job_id;
        assert_eq!(rig.jobs.busy(), 2);
        let (ra, rb) = (
            rig.jobs.wait(&a).await.unwrap(),
            rig.jobs.wait(&b).await.unwrap(),
        );
        assert_eq!(
            ra.result.unwrap().files.glb,
            rig.dir.path().join("out/same.glb")
        );
        assert_eq!(
            rb.result.unwrap().files.glb,
            rig.dir.path().join("out/same-1.glb")
        );
        assert_eq!(rig.backend.generations.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn queue_is_bounded() {
        let rig = rig_with(64_000, None, Duration::from_millis(200));
        let first = rig.jobs.submit(request(&rig, "q")).unwrap().job_id;
        let mut rx = rig.jobs.subscribe(&first).unwrap();
        while rx.borrow_and_update().status != JobStatus::Running {
            rx.changed().await.unwrap();
        }
        rig.jobs.submit(request(&rig, "q")).unwrap();
        rig.jobs.submit(request(&rig, "q")).unwrap();
        // One running, two waiting: the next is refused.
        assert_eq!(
            rig.jobs.submit(request(&rig, "q")).unwrap_err().code(),
            "queue_full"
        );
    }

    #[tokio::test]
    async fn cancelling_queued_and_running_jobs_leaves_no_partials() {
        let rig = rig_with(64_000, None, Duration::from_millis(100));
        let running = rig.jobs.submit(request(&rig, "a")).unwrap().job_id;
        let queued = rig.jobs.submit(request(&rig, "b")).unwrap().job_id;
        assert_eq!(
            rig.jobs.cancel(&queued).unwrap().status,
            JobStatus::Cancelled
        );
        let mut rx = rig.jobs.subscribe(&running).unwrap();
        while !matches!(rx.borrow_and_update().progress, Some(Stage::Shape { .. })) {
            rx.changed().await.unwrap();
        }
        assert!(
            !partials(rig.dir.path()).is_empty(),
            "the reserved partial exists while running"
        );
        rig.jobs.cancel(&running).unwrap();
        assert_eq!(
            rig.jobs.wait(&running).await.unwrap().status,
            JobStatus::Cancelled
        );
        assert!(
            partials(rig.dir.path()).is_empty(),
            "{:?}",
            partials(rig.dir.path())
        );
        assert!(!rig.dir.path().join("out/a.glb").exists());
    }

    #[tokio::test]
    async fn backend_failures_are_explained_and_cleaned_up() {
        let rig = rig();
        rig.backend.fail_next("texture", "empty_mesh");
        let id = rig.jobs.submit(request(&rig, "f")).unwrap().job_id;
        let snap = rig.jobs.wait(&id).await.unwrap();
        assert_eq!(snap.status, JobStatus::Failed);
        let err = snap.error.unwrap();
        assert_eq!(err.code, "backend_failed");
        assert!(
            err.message.contains("texture (empty_mesh)"),
            "{}",
            err.message
        );
        assert!(partials(rig.dir.path()).is_empty());
    }

    #[tokio::test]
    async fn low_memory_refuses_before_reserving_anything() {
        let rig = rig_with(9_000, None, Duration::from_millis(1));
        let id = rig.jobs.submit(request(&rig, "m")).unwrap().job_id;
        let err = rig.jobs.wait(&id).await.unwrap().error.unwrap();
        assert_eq!(err.code, "insufficient_memory");
        assert!(
            err.message.contains("Close UnrealEditor (6100 MB)"),
            "{}",
            err.message
        );
        assert!(
            std::fs::read_dir(rig.dir.path().join("out"))
                .unwrap()
                .next()
                .is_none()
        );
        assert_eq!(rig.backend.generations.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_resident_model_skips_the_memory_check() {
        let rig = rig_with(9_000, None, Duration::from_millis(1));
        // Load the model into the mock as a kept worker would.
        let mut warm = request(&rig, "warm");
        warm.request.keep_loaded = true;
        let dir = rig.dir.path().to_path_buf();
        let job = BackendJob {
            id: "w".into(),
            request: warm.request.clone(),
            out_glb: dir.join("out/w.glb.partial"),
            out_obj: None,
            hf_cache: dir.join("hub"),
        };
        rig.backend
            .generate(
                &job,
                &model(),
                ProgressSink::noop(),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        std::fs::remove_file(dir.join("out/w.glb.partial")).unwrap();
        let id = rig.jobs.submit(request(&rig, "m")).unwrap().job_id;
        assert_eq!(rig.jobs.wait(&id).await.unwrap().status, JobStatus::Done);
    }

    #[tokio::test]
    async fn a_busy_gpu_lock_makes_the_job_wait() {
        let rig = rig();
        let other = GpuLock::new(rig.dir.path().join("lock/local-gpu.lock"));
        let holder = Holder {
            server: "mcp-imagegen".into(),
            pid: 4242,
            job: None,
            since_unix: 0,
        };
        let guard = other.try_acquire(&holder).unwrap().unwrap();
        let id = rig.jobs.submit(request(&rig, "g")).unwrap().job_id;
        let mut rx = rig.jobs.subscribe(&id).unwrap();
        while !matches!(
            rx.borrow_and_update().progress,
            Some(Stage::WaitingForGpu { .. })
        ) {
            rx.changed().await.unwrap();
        }
        assert_eq!(
            rx.borrow().progress_text.as_deref(),
            Some("waiting for another local model (mcp-imagegen (pid 4242))")
        );
        drop(guard);
        assert_eq!(rig.jobs.wait(&id).await.unwrap().status, JobStatus::Done);
    }

    #[tokio::test]
    async fn previews_game_copy_and_inline_preview() {
        let rig = rig();
        let mut req = request(&rig, "p");
        req.request.previews = true;
        req.request.return_preview = true;
        req.request.size_m = Some(1.2);
        req.request.fbx = true;
        req.request.obj = true;
        let id = rig.jobs.submit(req).unwrap().job_id;
        let result = rig.jobs.wait(&id).await.unwrap().result.unwrap();
        assert_eq!(result.files.previews.len(), 4);
        assert!(result.files.previews.iter().all(|p| p.exists()));
        assert_eq!(
            result.files.game_glb.as_deref(),
            Some(rig.dir.path().join("out/p_game.glb").as_path())
        );
        assert!(result.files.game_fbx.as_ref().unwrap().exists());
        assert_eq!(
            result.files.obj.as_deref(),
            Some(rig.dir.path().join("out/p.obj").as_path())
        );
        assert_eq!(result.record.game.as_ref().unwrap().scale, 1.2);
        assert!(result.preview_png.is_some());
        assert!(
            result.record.timings.contains_key("previews")
                && result.record.timings.contains_key("post_process")
        );
        let sidecar: serde_json::Value =
            serde_json::from_slice(&std::fs::read(rig.dir.path().join("out/p.json")).unwrap())
                .unwrap();
        assert_eq!(sidecar["files"]["previews"].as_array().unwrap().len(), 4);
        assert_eq!(sidecar["files"]["game_fbx"], "p_game.fbx");
        assert!(partials(rig.dir.path()).is_empty());
    }

    #[tokio::test]
    async fn without_blender_the_mesh_still_succeeds_with_a_warning() {
        for post in [
            Some(MockPost {
                available: false,
                fail: false,
            }),
            Some(MockPost {
                available: true,
                fail: true,
            }),
            None,
        ] {
            let rig = rig_with(64_000, post, Duration::from_millis(1));
            let mut req = request(&rig, "nb");
            req.request.previews = true;
            req.request.fbx = true;
            let id = rig.jobs.submit(req).unwrap().job_id;
            let snap = rig.jobs.wait(&id).await.unwrap();
            assert_eq!(snap.status, JobStatus::Done);
            let result = snap.result.unwrap();
            assert!(result.files.glb.exists());
            assert!(result.files.previews.is_empty() && result.files.game_glb.is_none());
            assert!(!result.record.warnings.is_empty());
            assert!(partials(rig.dir.path()).is_empty());
        }
    }

    #[tokio::test]
    async fn progress_is_streamed_and_unload_waits_for_idle() {
        let rig = rig_with(64_000, None, Duration::from_millis(20));
        let mut req = request(&rig, "s");
        req.request.keep_loaded = true;
        let id = rig.jobs.submit(req).unwrap().job_id;
        let mut rx = rig.jobs.subscribe(&id).unwrap();
        let mut seen = Vec::new();
        loop {
            let snap = rx.borrow_and_update().clone();
            if let Some(t) = snap.progress_text
                && seen.last() != Some(&t)
            {
                seen.push(t);
            }
            if snap.status.is_terminal() {
                break;
            }
            if rx.changed().await.is_err() {
                break;
            }
        }
        assert!(seen.iter().any(|s| s.starts_with("shape")), "{seen:?}");
        // `watch` keeps only the latest value, so instant stages may be skipped; the slow ones are always seen.
        assert!(seen.iter().any(|s| s == "exporting"), "{seen:?}");
        // Unload is refused while busy, allowed after.
        let busy_id = rig.jobs.submit(request(&rig, "t")).unwrap().job_id;
        assert_eq!(
            rig.jobs.unload_all().await.unwrap_err().code(),
            "invalid_request"
        );
        rig.jobs.wait(&busy_id).await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        rig.jobs.unload_all().await.unwrap();
        assert_eq!(rig.backend.loaded_model(), None);
    }

    #[tokio::test]
    async fn finished_jobs_expire() {
        let mut rig = rig();
        rig.jobs.inner = Arc::new(Inner {
            jobs: Mutex::new(HashMap::new()),
            queue: Mutex::new(VecDeque::new()),
            running: Mutex::new(None),
            wake: Notify::new(),
            deps: JobDeps {
                job_ttl: Duration::from_millis(50),
                ..rig.jobs.inner.deps.clone()
            },
        });
        tokio::spawn(runner(Arc::downgrade(&rig.jobs.inner)));
        let id = rig.jobs.submit(request(&rig, "e")).unwrap().job_id;
        rig.jobs.wait(&id).await.unwrap();
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(rig.jobs.get(&id).unwrap_err().code(), "unknown_job");
    }
}
