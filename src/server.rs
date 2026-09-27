//! MCP surface (design §5, §7; R1, R3, R5, R6).

use std::sync::Arc;
use std::time::Duration;

use adk_mcp_sdk::{HealthCheck, HealthStatus};
use base64::Engine as _;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, schemars, tool, tool_router,
};
use serde::Serialize;
use serde_json::{Value, json};

use crate::backend::{Availability, BackendRegistry};
use crate::config::{Config, default_hf_cache};
use crate::error::MeshGenError;
use crate::gpu_lock::GpuLock;
use crate::heavy::{ProcessLister, SystemProcesses};
use crate::input::check_cutout;
use crate::jobs::{JobDeps, JobManager, JobRequest, JobSnapshot, JobStatus};
use crate::memory::{MemoryProbe, SystemMemory};
use crate::output::PathPolicy;
use crate::post::PostProcessor;
use crate::registry::{ModelRegistry, ModelSpec};
use crate::request::MeshRequest;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct JobIdInput {
    /// Job id returned by generate_mesh (with wait=false)
    pub job_id: String,
}

struct State {
    cfg: Config,
    registry: ModelRegistry,
    jobs: JobManager,
    paths: PathPolicy,
    memory: Arc<dyn MemoryProbe>,
}

#[derive(Clone)]
pub struct MeshGenServer {
    state: Arc<State>,
}

/// Backends compiled into this build.
pub fn default_backends(cfg: &Config) -> BackendRegistry {
    #[allow(unused_mut)]
    let mut backends = BackendRegistry::default();
    #[cfg(feature = "trellis_mac")]
    backends.insert(Arc::new(
        crate::backend::trellis_mac::TrellisMacBackend::new(
            cfg.trellis_mac.clone(),
            cfg.app_cache_dir(),
            cfg.idle_unload_s,
        ),
    ));
    #[cfg(feature = "mock")]
    {
        let step_ms = std::env::var("MCP_MESHGEN_MOCK_STEP_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50);
        let mock = crate::backend::mock::MockBackend::new(Duration::from_millis(step_ms));
        // "stage:kind" makes the first job fail there (protocol tests).
        if let Ok(spec) = std::env::var("MCP_MESHGEN_MOCK_FAIL")
            && let Some((stage, kind)) = spec.split_once(':')
        {
            mock.fail_next(stage, kind);
        }
        backends.insert(Arc::new(mock));
    }
    let _ = cfg;
    backends
}

/// The post-processor for previews and the engine copy: headless Blender (reports "Blender not found" when missing),
/// or the mock under the `mock` feature when `MCP_MESHGEN_MOCK_POST=1`.
pub fn default_post(cfg: &Config) -> Option<Arc<dyn PostProcessor>> {
    #[cfg(feature = "mock")]
    if std::env::var("MCP_MESHGEN_MOCK_POST").is_ok_and(|v| v == "1") {
        return Some(Arc::new(crate::post::MockPost {
            available: true,
            fail: false,
        }));
    }
    Some(Arc::new(crate::post::blender::BlenderPost::new(
        cfg.blender.as_deref(),
        cfg.app_cache_dir(),
    )))
}

impl MeshGenServer {
    /// Must be called inside a tokio runtime (spawns the job runner).
    pub fn new(
        cfg: Config,
        registry: ModelRegistry,
        backends: BackendRegistry,
        post: Option<Arc<dyn PostProcessor>>,
        memory: Arc<dyn MemoryProbe>,
        processes: Arc<dyn ProcessLister>,
    ) -> Self {
        let jobs = JobManager::new(JobDeps {
            backends,
            post,
            gpu_lock: GpuLock::new(cfg.gpu_lock_path()),
            memory: memory.clone(),
            processes,
            heavy_processes: cfg.heavy_processes.clone(),
            memory_headroom_mb: cfg.memory_headroom_mb,
            hf_cache: cfg.hf_cache.clone(),
            max_queue: cfg.max_queue,
            job_ttl: Duration::from_secs(cfg.job_ttl_s),
        });
        let paths = PathPolicy::from_config(&cfg);
        Self {
            state: Arc::new(State {
                cfg,
                registry,
                jobs,
                paths,
                memory,
            }),
        }
    }

    /// Load config and registry from the default location and build every compiled backend.
    pub fn from_default_config() -> Result<Self, MeshGenError> {
        let cfg = Config::load_or_init(&Config::default_dir())?;
        let registry = ModelRegistry::load(&cfg.models_path())?;
        let backends = default_backends(&cfg);
        let post = default_post(&cfg);
        Ok(Self::new(
            cfg,
            registry,
            backends,
            post,
            Arc::new(SystemMemory::default()),
            Arc::new(SystemProcesses::default()),
        ))
    }

    pub fn config(&self) -> &Config {
        &self.state.cfg
    }

    /// Stop everything, even mid-job: cancel the running job (which kills its worker tree) and stop kept workers.
    /// The worker runs in its own process group, so exiting the server alone wouldn't take it down.
    pub async fn shutdown(&self) {
        self.state.jobs.cancel_all();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while self.state.jobs.busy() > 0 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        for backend in self.state.jobs.deps().backends.all() {
            backend.unload().await;
        }
    }

    /// Validate paths and parameters (R10), check the cut-out (R2) and the model status (R3), then queue.
    async fn submit(&self, input: &MeshRequest) -> Result<JobSnapshot, MeshGenError> {
        let st = &self.state;
        let (request, model, target) = input.resolve(
            &st.registry,
            &st.cfg.default_model,
            st.cfg.keep_loaded,
            &st.paths,
        )?;
        let report = check_cutout(&request.image, &st.cfg.input)?;
        let availability = st
            .jobs
            .deps()
            .backends
            .availability_of(&model.backend)
            .await;
        let status = model.status(&st.cfg.hf_cache, &default_hf_cache(), availability.reason());
        if !status.is_ready() {
            return Err(status.not_ready_error(&model.id));
        }
        st.jobs.submit(JobRequest {
            model: model.clone(),
            request,
            target,
            input: report,
        })
    }

    async fn finish(&self, snap: JobSnapshot, wait: bool) -> Result<CallToolResult, McpError> {
        if !wait {
            return Ok(structured(json!({
                "job_id": snap.job_id,
                "status": snap.status,
                "progress": snap.progress_text,
                "hint": "poll get_job with this job_id; cancel_job stops it",
            })));
        }
        // If the call itself is cancelled (tasks/cancel, request cancellation), stop the job too.
        let guard = CancelOnDrop {
            jobs: self.state.jobs.clone(),
            job_id: Some(snap.job_id.clone()),
        };
        let done = self.wait_with_progress(&snap.job_id).await?;
        guard.disarm();
        match (&done.status, &done.result) {
            (JobStatus::Done, Some(result)) => {
                let mut value = serde_json::to_value(result).unwrap_or(Value::Null);
                value["status"] = json!("done");
                let mut out = CallToolResult::structured(value);
                if let Some(preview) = &result.preview_png {
                    out.content.push(ContentBlock::image(
                        base64::engine::general_purpose::STANDARD.encode(preview),
                        "image/png",
                    ));
                }
                Ok(out)
            }
            _ => Ok(job_failure(&done)),
        }
    }

    /// Wait for a job, forwarding progress to the MCP task status when running as a Task.
    async fn wait_with_progress(&self, job_id: &str) -> Result<JobSnapshot, MeshGenError> {
        let mut rx = self.state.jobs.subscribe(job_id)?;
        let mut last = String::new();
        loop {
            let snap = rx.borrow_and_update().clone();
            if let Some(text) = &snap.progress_text
                && *text != last
            {
                adk_mcp_sdk::set_current_task_status(text.clone());
                last = text.clone();
            }
            if snap.status.is_terminal() {
                return Ok(snap);
            }
            if rx.changed().await.is_err() {
                return Err(MeshGenError::UnknownJob(job_id.to_string()));
            }
        }
    }

    async fn model_entry(&self, model: &ModelSpec) -> Value {
        let st = &self.state;
        let backends = &st.jobs.deps().backends;
        let availability = backends.availability_of(&model.backend).await;
        let status = model.status(&st.cfg.hf_cache, &default_hf_cache(), availability.reason());
        let loaded = backends
            .get(&model.backend)
            .is_some_and(|b| b.loaded_model().as_deref() == Some(model.id.as_str()));
        let mut entry = json!({
            "id": model.id,
            "name": model.name,
            "backend": model.backend,
            "runs": model.runs,
            "status": status.code(),
            "loaded": loaded,
            "quality": model.quality,
            "defaults": model.defaults,
            "texture_sizes": model.texture_sizes,
            "max_triangles": model.max_triangles,
            "memory_mb": model.memory_mb,
            "needs_mb_free": model.memory_mb + st.cfg.memory_headroom_mb,
            "typical_s": model.typical_s,
            "licences": model.licence,
            "commercial_use": model.licence.iter().all(|l| l.commercial),
            "fixes": model.fixes,
        });
        if !status.is_ready() {
            entry["detail"] = json!(status.detail());
            if let Value::Object(extra) = serde_json::to_value(&status).unwrap_or(Value::Null) {
                for (k, v) in extra {
                    if k != "status" {
                        entry[k] = v;
                    }
                }
            }
        }
        entry
    }
}

fn structured(value: impl Serialize) -> CallToolResult {
    CallToolResult::structured(serde_json::to_value(value).unwrap_or(Value::Null))
}

/// Tool-level failure (the call worked, the conversion didn't).
fn job_failure(snap: &JobSnapshot) -> CallToolResult {
    CallToolResult::structured_error(json!({
        "job_id": snap.job_id,
        "status": snap.status,
        "error": snap.error,
    }))
}

#[tool_router]
impl MeshGenServer {
    #[tool(
        description = "Convert a cut-out image (PNG/WebP with a transparent background) into a textured 3D mesh (GLB, optional OBJ). \
Optional: previews (four PNG views), target_triangles (triangle budget, applied before texture baking), size_m + size_axis and \
fbx for an engine-ready copy (origin at the bottom centre, Unreal FBX settings). Takes minutes on a local model: pass wait=false \
to get a job_id and poll get_job. Inputs without a real cut-out are rejected; make one with an image editor first."
    )]
    async fn generate_mesh(
        &self,
        Parameters(input): Parameters<MeshRequest>,
    ) -> Result<CallToolResult, McpError> {
        let snap = self.submit(&input).await?;
        self.finish(snap, input.wait).await
    }

    #[tool(description = "Get status, stage progress and result of a mesh job")]
    async fn get_job(
        &self,
        Parameters(input): Parameters<JobIdInput>,
    ) -> Result<CallToolResult, McpError> {
        let snap = self.state.jobs.get(&input.job_id)?;
        Ok(if snap.status == JobStatus::Failed {
            job_failure(&snap)
        } else {
            structured(&snap)
        })
    }

    #[tool(
        description = "Cancel a queued or running mesh job. The worker is stopped and no partial files are left behind."
    )]
    async fn cancel_job(
        &self,
        Parameters(input): Parameters<JobIdInput>,
    ) -> Result<CallToolResult, McpError> {
        Ok(structured(self.state.jobs.cancel(&input.job_id)?))
    }

    #[tool(
        description = "List image-to-3D models: status (ready / missing_files with sizes and the exact download command / \
needs_licence_acceptance with the page to accept / backend_unavailable with why), licence of every component and whether \
commercial use is allowed, memory needed and typical time. Never loads weights."
    )]
    async fn list_models(&self) -> Result<CallToolResult, McpError> {
        let st = &self.state;
        let mut models = Vec::new();
        for model in st.registry.models() {
            models.push(self.model_entry(model).await);
        }
        let post = match &st.jobs.deps().post {
            Some(p) => match p.availability() {
                Ok(()) => json!({ "status": "ready" }),
                Err(reason) => json!({ "status": "unavailable", "reason": reason }),
            },
            None => json!({ "status": "unavailable", "reason": "no post-processor in this build" }),
        };
        let gpu_holder = st
            .jobs
            .deps()
            .gpu_lock
            .current_holder()
            .map(|h| h.describe());
        Ok(structured(json!({
            "models": models,
            "default_model": st.cfg.default_model,
            "available_memory_mb": st.memory.available_mb(),
            "allowed_roots": st.cfg.allowed_roots,
            "default_output_dir": st.cfg.default_output_dir,
            "config_dir": st.cfg.config_dir,
            "hf_cache": st.cfg.hf_cache,
            "previews_and_engine_copy": post,
            "gpu_lock_holder": gpu_holder,
            "jobs_busy": st.jobs.busy(),
        })))
    }

    #[tool(
        description = "Stop a kept-alive model worker to free its memory (only needed after keep_loaded jobs; by default the \
worker exits after each job). Refused while jobs are queued or running."
    )]
    async fn unload_models(&self) -> Result<CallToolResult, McpError> {
        let unloaded = self.state.jobs.unload_all().await?;
        Ok(structured(json!({
            "unloaded": unloaded,
            "available_memory_mb": self.state.memory.available_mb(),
        })))
    }
}

struct CancelOnDrop {
    jobs: JobManager,
    job_id: Option<String>,
}

impl CancelOnDrop {
    fn disarm(mut self) {
        self.job_id = None;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(id) = self.job_id.take() {
            tracing::info!(job_id = %id, "request dropped before completion; cancelling job");
            let _ = self.jobs.cancel(&id);
        }
    }
}

adk_mcp_sdk::mcp_2026_server! {
    server: MeshGenServer,
    task_tools: ["generate_mesh"],
    task_ttl_overrides: [("generate_mesh", 7_200_000)],
    approval_tools: [],
    mutating_tools: ["generate_mesh", "cancel_job", "unload_models"],
    destructive_tools: [],
    idempotent_tools: ["cancel_job", "unload_models"],
    cache_ttl_ms: 60_000,
    instructions: "Image-to-3D conversion with local models. Call list_models first to see which models are ready, what their \
licences allow and how much free memory they need. Inputs must already be cut out (PNG or WebP with a transparent background); \
make one with an image editor such as mcp-imagegen edit_image. generate_mesh takes minutes: pass wait=false and poll get_job if \
your client times out. One conversion runs at a time, and the model is unloaded after each job unless keep_loaded is set.",
}

#[async_trait::async_trait]
impl HealthCheck for MeshGenServer {
    async fn check_health(&self) -> HealthStatus {
        let started = std::time::Instant::now();
        let backends = &self.state.jobs.deps().backends;
        let mut parts = Vec::new();
        for id in backends.ids() {
            let state = match backends.availability_of(&id).await {
                Availability::Ready => "ready".to_string(),
                Availability::Unavailable { reason } => format!("unavailable ({reason})"),
            };
            parts.push(format!("{id}: {state}"));
        }
        HealthStatus {
            // A backend that can't run here is reported per model by list_models; the server itself is fine.
            healthy: true,
            message: Some(format!(
                "{} model(s); backends: {}",
                self.state.registry.models().len(),
                if parts.is_empty() {
                    "none".to_string()
                } else {
                    parts.join(", ")
                }
            )),
            latency_ms: Some(started.elapsed().as_millis() as u64),
        }
    }
}
