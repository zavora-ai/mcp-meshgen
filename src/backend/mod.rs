//! Mesh backends behind one trait (design §4.1; R4).
//!
//! A backend turns a checked cut-out into a GLB (and optionally an OBJ) written to the `*.partial` paths it's given;
//! the job manager renames them on success and deletes them on cancel or failure.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use serde::Serialize;
use tokio_util::sync::CancellationToken;

use crate::error::{BackendFailure, MeshGenError};
use crate::registry::ModelSpec;
use crate::request::ResolvedRequest;

#[cfg(any(test, feature = "mock"))]
pub mod mock;
pub mod process;
#[cfg(feature = "trellis_mac")]
pub mod trellis_mac;

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Availability {
    Ready,
    /// Why it can't run here, including how to fix it.
    Unavailable {
        reason: String,
    },
}

impl Availability {
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Ready => None,
            Self::Unavailable { reason } => Some(reason),
        }
    }
}

/// Where a job is (R5). Sampling stages carry step counts.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub enum Stage {
    Queued { position: usize },
    WaitingForGpu { holder: Option<String> },
    Starting,
    Loading,
    SparseStructure { step: u32, total: u32 },
    Shape { step: u32, total: u32 },
    Texture { step: u32, total: u32 },
    Decode,
    Bake,
    Export,
    Previews,
    PostProcess,
    Saving,
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Queued { position } => write!(f, "queued (position {position})"),
            Self::WaitingForGpu { holder } => {
                f.write_str(&crate::gpu_lock::waiting_message_for(holder.as_deref()))
            }
            Self::Starting => f.write_str("starting worker"),
            Self::Loading => f.write_str("loading model"),
            Self::SparseStructure { step, total } => write!(f, "sparse structure {step}/{total}"),
            Self::Shape { step, total } => write!(f, "shape {step}/{total}"),
            Self::Texture { step, total } => write!(f, "texture {step}/{total}"),
            Self::Decode => f.write_str("decoding"),
            Self::Bake => f.write_str("baking textures"),
            Self::Export => f.write_str("exporting"),
            Self::Previews => f.write_str("rendering previews"),
            Self::PostProcess => f.write_str("post-processing for the engine"),
            Self::Saving => f.write_str("saving"),
        }
    }
}

/// Cheap clonable progress callback. Backends call it; the job manager records it.
#[derive(Clone)]
pub struct ProgressSink(Arc<dyn Fn(Stage) + Send + Sync>);

impl ProgressSink {
    pub fn new(f: impl Fn(Stage) + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }

    pub fn noop() -> Self {
        Self::new(|_| {})
    }

    pub fn report(&self, stage: Stage) {
        (self.0)(stage)
    }
}

/// One conversion, fully resolved, with where to write.
#[derive(Debug, Clone)]
pub struct BackendJob {
    pub id: String,
    pub request: ResolvedRequest,
    /// The GLB goes here (a `*.partial` path).
    pub out_glb: PathBuf,
    /// The OBJ goes here when asked (a `*.partial` path; its .mtl and textures sit beside it, named after it).
    pub out_obj: Option<PathBuf>,
    pub hf_cache: PathBuf,
}

/// What a backend reports about a finished conversion. Mesh stats come from the GLB itself (`glb::stats`).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GeneratedMesh {
    /// Seconds per stage (`load`, `sparse_structure`, `shape`, `texture`, `decode`, `bake`, `export`, …).
    pub timings: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_rss_mb: Option<u64>,
    /// Fixes applied at load (R7), e.g. `mps_sparse_decoder_cpu`.
    pub fixes: Vec<String>,
    pub warnings: Vec<String>,
    /// Extra files written beside the OBJ (its .mtl and textures), final names; each was written as `<name>.partial`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub extra_files: Vec<PathBuf>,
    /// Backend-specific facts for the sidecar (voxel count, faces before simplification, which baker ran, …).
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub details: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BackendError {
    Cancelled,
    Unavailable {
        reason: String,
    },
    /// Explained failure (R7).
    Failed {
        stage: String,
        kind: String,
        message: String,
        hint: String,
        log_tail: String,
    },
}

impl BackendError {
    pub fn failed(
        stage: &str,
        kind: &str,
        message: impl Into<String>,
        hint: impl Into<String>,
    ) -> Self {
        Self::Failed {
            stage: stage.to_string(),
            kind: kind.to_string(),
            message: message.into(),
            hint: hint.into(),
            log_tail: String::new(),
        }
    }

    pub fn into_error(self, backend: &str) -> MeshGenError {
        match self {
            Self::Cancelled => MeshGenError::Cancelled,
            Self::Unavailable { reason } => MeshGenError::BackendUnavailable {
                backend: backend.to_string(),
                reason,
            },
            Self::Failed {
                stage,
                kind,
                message,
                hint,
                log_tail,
            } => MeshGenError::Backend(Box::new(BackendFailure {
                backend: backend.to_string(),
                stage,
                kind,
                message,
                hint,
                log_tail,
            })),
        }
    }
}

#[async_trait::async_trait]
pub trait MeshBackend: Send + Sync {
    fn id(&self) -> &'static str;

    /// File- and platform-level check only; never loads weights.
    async fn availability(&self) -> Availability;

    /// Run one conversion. Must honour `cancel`, report progress, and write only to the job's partial paths.
    async fn generate(
        &self,
        job: &BackendJob,
        model: &ModelSpec,
        progress: ProgressSink,
        cancel: CancellationToken,
    ) -> Result<GeneratedMesh, BackendError>;

    /// Free memory now (stop a kept-alive worker). Idempotent.
    async fn unload(&self);

    /// Id of the model currently resident, if any.
    fn loaded_model(&self) -> Option<String>;
}

/// Backends keyed by id.
#[derive(Clone, Default)]
pub struct BackendRegistry {
    backends: HashMap<String, Arc<dyn MeshBackend>>,
}

impl BackendRegistry {
    pub fn insert(&mut self, backend: Arc<dyn MeshBackend>) {
        self.backends.insert(backend.id().to_string(), backend);
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn MeshBackend>> {
        self.backends.get(id).cloned()
    }

    pub fn all(&self) -> impl Iterator<Item = &Arc<dyn MeshBackend>> {
        self.backends.values()
    }

    pub fn ids(&self) -> Vec<String> {
        let mut ids: Vec<_> = self.backends.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Availability for a model's backend, with a clear reason when this build doesn't include it.
    pub async fn availability_of(&self, backend: &str) -> Availability {
        match self.get(backend) {
            Some(b) => b.availability().await,
            None => Availability::Unavailable {
                reason: format!(
                    "backend `{backend}` is not in this build (built with: {}). Rebuild with `--features {backend}`",
                    if self.backends.is_empty() {
                        "none".to_string()
                    } else {
                        self.ids().join(", ")
                    }
                ),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_messages() {
        assert_eq!(
            Stage::Shape { step: 3, total: 12 }.to_string(),
            "shape 3/12"
        );
        assert_eq!(
            Stage::Queued { position: 2 }.to_string(),
            "queued (position 2)"
        );
        assert_eq!(
            Stage::WaitingForGpu {
                holder: Some("mcp-imagegen (pid 7)".into())
            }
            .to_string(),
            "waiting for another local model (mcp-imagegen (pid 7))"
        );
        let json = serde_json::to_value(Stage::Texture { step: 1, total: 12 }).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"stage": "texture", "step": 1, "total": 12})
        );
    }

    #[test]
    fn failures_map_to_explained_errors() {
        let err = BackendError::failed(
            "sparse_structure",
            "empty_structure",
            "0 voxels",
            "try another seed",
        )
        .into_error("trellis_mac");
        assert_eq!(err.code(), "backend_failed");
        assert!(
            err.to_string()
                .contains("sparse_structure (empty_structure)"),
            "{err}"
        );
        assert_eq!(BackendError::Cancelled.into_error("x").code(), "cancelled");
        let err = BackendError::Unavailable {
            reason: "no venv".into(),
        }
        .into_error("trellis_mac");
        assert_eq!(err.code(), "backend_unavailable");
    }

    #[tokio::test]
    async fn missing_backend_is_unavailable_with_a_reason() {
        let reg = BackendRegistry::default();
        let reason = reg
            .availability_of("trellis_mac")
            .await
            .reason()
            .unwrap()
            .to_string();
        assert!(
            reason.contains("not in this build") && reason.contains("--features trellis_mac"),
            "{reason}"
        );
    }
}
