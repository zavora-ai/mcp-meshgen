//! Error type (design §7). Every variant says what went wrong and what to do next (R7).

use std::fmt::Display;
use std::path::{Path, PathBuf};

use rmcp::ErrorData as McpError;

/// Where a background-removed input comes from (R2).
pub const CUTOUT_HINT: &str =
    "cut the subject out onto a transparent background first, e.g. with mcp-imagegen edit_image";

/// Details of a backend failure (boxed in `MeshGenError::Backend` to keep the error small).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{backend} failed during {stage} ({kind}): {message}{}{}", if hint.is_empty() { String::new() } else { format!(". Try: {hint}") }, if log_tail.is_empty() { String::new() } else { format!("\n--- worker log tail ---\n{log_tail}") })]
pub struct BackendFailure {
    pub backend: String,
    /// `loading`, `sparse_structure`, `shape`, `texture`, `decode`, `bake`, `export`, …
    pub stage: String,
    /// `empty_structure`, `empty_mesh`, `gpu_watchdog`, `worker_exception`, `worker_exited`, …
    pub kind: String,
    pub message: String,
    pub hint: String,
    pub log_tail: String,
}

#[derive(Debug, thiserror::Error)]
pub enum MeshGenError {
    #[error("config error in {}: {message}", path.display())]
    Config { path: PathBuf, message: String },

    #[error("invalid `{field}`: {message}")]
    InvalidRequest { field: String, message: String },

    #[error("unknown model `{id}`. Available: {}", available.join(", "))]
    UnknownModel { id: String, available: Vec<String> },

    /// The model exists but can't run yet; `detail` comes from the registry status (R3).
    #[error("model `{model}` is not ready ({status}): {detail}")]
    NotReady {
        model: String,
        status: String,
        detail: String,
    },

    #[error("input image {} rejected: {reason}. Fix: {CUTOUT_HINT}", path.display())]
    InputRejected { path: PathBuf, reason: String },

    #[error("path {} is outside the allowed roots ({})", path.display(), roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(", "))]
    PathNotAllowed { path: PathBuf, roots: Vec<PathBuf> },

    #[error(
        "not enough free memory to load `{model}`: needs ~{need_mb} MB (model plus headroom), {available_mb} MB available. {advice}"
    )]
    InsufficientMemory {
        model: String,
        need_mb: u64,
        available_mb: u64,
        /// Which heavy processes to close, from the census (§6).
        advice: String,
    },

    #[error("backend `{backend}` is unavailable: {reason}")]
    BackendUnavailable { backend: String, reason: String },

    /// A failure inside the backend, already explained (R7): which stage, what kind, what to try.
    #[error("{0}")]
    Backend(Box<BackendFailure>),

    #[error("headless Blender failed: {0}")]
    Blender(String),

    #[error("mesh queue is full ({max} waiting). Retry later or cancel a job with cancel_job")]
    QueueFull { max: usize },

    #[error("job was cancelled")]
    Cancelled,

    #[error(
        "unknown job `{0}` (finished jobs expire after job_ttl_s, and jobs don't survive a server restart)"
    )]
    UnknownJob(String),

    #[error("I/O error on {}: {message}", path.display())]
    Io { path: PathBuf, message: String },
}

impl MeshGenError {
    pub fn config(path: &Path, message: impl Display) -> Self {
        Self::Config {
            path: path.to_path_buf(),
            message: message.to_string(),
        }
    }

    pub fn invalid(field: &str, message: impl Display) -> Self {
        Self::InvalidRequest {
            field: field.to_string(),
            message: message.to_string(),
        }
    }

    pub fn io(path: &Path, err: impl Display) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            message: err.to_string(),
        }
    }

    /// Stable machine-readable code for structured results.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Config { .. } => "config_error",
            Self::InvalidRequest { .. } => "invalid_request",
            Self::UnknownModel { .. } => "unknown_model",
            Self::NotReady { .. } => "not_ready",
            Self::InputRejected { .. } => "input_rejected",
            Self::PathNotAllowed { .. } => "path_not_allowed",
            Self::InsufficientMemory { .. } => "insufficient_memory",
            Self::BackendUnavailable { .. } => "backend_unavailable",
            Self::Backend(_) => "backend_failed",
            Self::Blender(_) => "blender_failed",
            Self::QueueFull { .. } => "queue_full",
            Self::Cancelled => "cancelled",
            Self::UnknownJob(_) => "unknown_job",
            Self::Io { .. } => "io_error",
        }
    }

    /// Caller mistakes map to invalid_params; everything else is an internal error.
    pub fn is_caller_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidRequest { .. }
                | Self::UnknownModel { .. }
                | Self::InputRejected { .. }
                | Self::PathNotAllowed { .. }
                | Self::UnknownJob(_)
        )
    }
}

impl From<MeshGenError> for McpError {
    fn from(err: MeshGenError) -> Self {
        let mut data = serde_json::json!({ "code": err.code() });
        if let MeshGenError::Backend(failure) = &err {
            data["stage"] = failure.stage.clone().into();
            data["kind"] = failure.kind.clone().into();
        }
        if err.is_caller_error() {
            McpError::invalid_params(err.to_string(), Some(data))
        } else {
            McpError::internal_error(err.to_string(), Some(data))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend(hint: &str, log_tail: &str) -> MeshGenError {
        MeshGenError::Backend(Box::new(BackendFailure {
            backend: "trellis_mac".into(),
            stage: "sparse_structure".into(),
            kind: "empty_structure".into(),
            message: "the sparse structure decoded to 0 voxels".into(),
            hint: hint.into(),
            log_tail: log_tail.into(),
        }))
    }

    #[test]
    fn backend_error_names_stage_kind_and_hint() {
        let msg = backend("try another seed", "").to_string();
        assert_eq!(
            msg,
            "trellis_mac failed during sparse_structure (empty_structure): the sparse structure decoded to 0 voxels. Try: try another seed"
        );
        assert!(!msg.contains("log tail"));
        let with_log = backend("", "Traceback ...").to_string();
        assert!(!with_log.contains("Try:"), "{with_log}");
        assert!(with_log.ends_with("--- worker log tail ---\nTraceback ..."));
    }

    #[test]
    fn backend_error_data_carries_stage_and_kind() {
        let mcp: McpError = backend("", "").into();
        assert_eq!(mcp.code, rmcp::model::ErrorCode::INTERNAL_ERROR);
        let data = mcp.data.unwrap();
        assert_eq!(data["code"], "backend_failed");
        assert_eq!(data["stage"], "sparse_structure");
        assert_eq!(data["kind"], "empty_structure");
    }

    #[test]
    fn input_rejection_is_a_caller_error_with_the_cutout_hint() {
        let err = MeshGenError::InputRejected {
            path: PathBuf::from("/w/brazier.png"),
            reason: "no alpha channel".into(),
        };
        assert!(err.to_string().contains("edit_image"), "{err}");
        let mcp: McpError = err.into();
        assert_eq!(mcp.code, rmcp::model::ErrorCode::INVALID_PARAMS);
    }

    #[test]
    fn memory_refusal_includes_advice() {
        let err = MeshGenError::InsufficientMemory {
            model: "trellis2-4b".into(),
            need_mb: 17000,
            available_mb: 9000,
            advice: "Close Unreal Editor (6100 MB).".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("17000 MB") && msg.contains("9000 MB available"));
        assert!(msg.ends_with("Close Unreal Editor (6100 MB)."), "{msg}");
        assert!(!err.is_caller_error());
    }

    #[test]
    fn codes_are_stable() {
        assert_eq!(MeshGenError::Cancelled.code(), "cancelled");
        assert_eq!(MeshGenError::QueueFull { max: 4 }.code(), "queue_full");
        assert_eq!(
            MeshGenError::NotReady {
                model: "m".into(),
                status: "missing_files".into(),
                detail: "hf download org/repo a.safetensors".into(),
            }
            .code(),
            "not_ready"
        );
    }
}
