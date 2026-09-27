//! Deterministic fake backend for tests and demos (feature `mock`).
//!
//! Walks the real stage sequence with `step_delay` per step, writes the test cube as the GLB (and a matching OBJ),
//! honours cancellation, and can be told to fail the next job with a given stage and kind.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use super::{
    Availability, BackendError, BackendJob, GeneratedMesh, MeshBackend, ProgressSink, Stage,
};
use crate::registry::ModelSpec;

/// Steps per sampling stage when the request doesn't set any.
pub const MOCK_STEPS: u32 = 3;

pub struct MockBackend {
    id: &'static str,
    step_delay: Duration,
    availability: Mutex<Availability>,
    loaded: Mutex<Option<String>>,
    fail_next: Mutex<Option<(String, String)>>,
    pub generations: AtomicUsize,
    pub unloads: AtomicUsize,
}

impl MockBackend {
    pub fn new(step_delay: Duration) -> Self {
        Self::with_id("mock", step_delay)
    }

    /// A mock registered under another backend's id (e.g. to stand in for `trellis_mac` in tests).
    pub fn with_id(id: &'static str, step_delay: Duration) -> Self {
        Self {
            id,
            step_delay,
            availability: Mutex::new(Availability::Ready),
            loaded: Mutex::new(None),
            fail_next: Mutex::new(None),
            generations: AtomicUsize::new(0),
            unloads: AtomicUsize::new(0),
        }
    }

    /// Fail the next job during `stage` with `kind` (after that stage's progress starts).
    pub fn fail_next(&self, stage: &str, kind: &str) {
        *self.fail_next.lock().unwrap() = Some((stage.to_string(), kind.to_string()));
    }

    pub fn set_availability(&self, availability: Availability) {
        *self.availability.lock().unwrap() = availability;
    }

    fn check_failure(&self, stage: &str) -> Result<(), BackendError> {
        let mut fail = self.fail_next.lock().unwrap();
        if fail.as_ref().is_some_and(|(s, _)| s == stage) {
            let (stage, kind) = fail.take().unwrap();
            return Err(BackendError::failed(
                &stage,
                &kind,
                format!("injected {kind} failure"),
                "this is the mock backend",
            ));
        }
        Ok(())
    }

    async fn tick(&self, cancel: &CancellationToken) -> Result<(), BackendError> {
        tokio::select! {
            _ = cancel.cancelled() => Err(BackendError::Cancelled),
            _ = tokio::time::sleep(self.step_delay) => Ok(()),
        }
    }
}

#[async_trait::async_trait]
impl MeshBackend for MockBackend {
    fn id(&self) -> &'static str {
        self.id
    }

    async fn availability(&self) -> Availability {
        self.availability.lock().unwrap().clone()
    }

    async fn generate(
        &self,
        job: &BackendJob,
        model: &ModelSpec,
        progress: ProgressSink,
        cancel: CancellationToken,
    ) -> Result<GeneratedMesh, BackendError> {
        progress.report(Stage::Loading);
        *self.loaded.lock().unwrap() = Some(model.id.clone());
        self.tick(&cancel).await?;
        self.check_failure("loading")?;

        let total = job.request.steps.unwrap_or(MOCK_STEPS);
        let mut timings = BTreeMap::new();
        for (name, make) in [
            (
                "sparse_structure",
                (|step, total| Stage::SparseStructure { step, total }) as fn(u32, u32) -> Stage,
            ),
            ("shape", |step, total| Stage::Shape { step, total }),
            ("texture", |step, total| Stage::Texture { step, total }),
        ] {
            for step in 1..=total {
                progress.report(make(step, total));
                self.check_failure(name)?;
                self.tick(&cancel).await?;
            }
            timings.insert(
                name.to_string(),
                self.step_delay.as_secs_f64() * f64::from(total),
            );
        }
        for (name, stage) in [
            ("decode", Stage::Decode),
            ("bake", Stage::Bake),
            ("export", Stage::Export),
        ] {
            progress.report(stage);
            self.check_failure(name)?;
            self.tick(&cancel).await?;
            timings.insert(name.to_string(), self.step_delay.as_secs_f64());
        }

        let io = |e: std::io::Error, what: &std::path::Path| {
            BackendError::failed("export", "io", format!("{}: {e}", what.display()), "")
        };
        std::fs::write(&job.out_glb, crate::glb::cube()).map_err(|e| io(e, &job.out_glb))?;
        if let Some(obj) = &job.out_obj {
            std::fs::write(obj, CUBE_OBJ).map_err(|e| io(e, obj))?;
        }
        if !job.request.keep_loaded {
            self.loaded.lock().unwrap().take();
        }
        self.generations.fetch_add(1, Ordering::SeqCst);
        Ok(GeneratedMesh {
            timings,
            peak_rss_mb: Some(64),
            fixes: model.fixes.clone(),
            warnings: Vec::new(),
            extra_files: Vec::new(),
            details: BTreeMap::from([("voxels".to_string(), serde_json::json!(4096))]),
        })
    }

    async fn unload(&self) {
        if self.loaded.lock().unwrap().take().is_some() {
            self.unloads.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn loaded_model(&self) -> Option<String> {
        self.loaded.lock().unwrap().clone()
    }
}

/// Geometry-only OBJ of the same 1 m cube.
const CUBE_OBJ: &str = "# mcp-meshgen mock cube\n\
v -0.5 -0.5 -0.5\nv 0.5 -0.5 -0.5\nv 0.5 0.5 -0.5\nv -0.5 0.5 -0.5\n\
v -0.5 -0.5 0.5\nv 0.5 -0.5 0.5\nv 0.5 0.5 0.5\nv -0.5 0.5 0.5\n\
f 1 3 2\nf 1 4 3\nf 5 6 7\nf 5 7 8\nf 1 2 6\nf 1 6 5\n\
f 2 3 7\nf 2 7 6\nf 3 4 8\nf 3 8 7\nf 4 1 5\nf 4 5 8\n";

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::config::MODELS_EXAMPLE;
    use crate::registry::ModelRegistry;
    use crate::request::{ResolvedRequest, SizeAxis};

    fn model() -> ModelSpec {
        ModelRegistry::from_toml(MODELS_EXAMPLE)
            .unwrap()
            .get("trellis2-4b")
            .unwrap()
            .clone()
    }

    fn job(dir: &std::path::Path, obj: bool) -> BackendJob {
        BackendJob {
            id: "j1".into(),
            request: ResolvedRequest {
                model: "trellis2-4b".into(),
                image: dir.join("in.png"),
                seed: 1,
                quality: "512".into(),
                texture_size: 1024,
                steps: Some(2),
                target_triangles: 200_000,
                obj,
                previews: false,
                return_preview: false,
                size_m: None,
                size_axis: SizeAxis::Longest,
                fbx: false,
                keep_loaded: false,
                warnings: vec![],
            },
            out_glb: dir.join("m.glb.partial"),
            out_obj: obj.then(|| dir.join("m.obj.partial")),
            hf_cache: dir.to_path_buf(),
        }
    }

    #[tokio::test]
    async fn walks_every_stage_and_writes_the_cube() {
        let dir = tempfile::tempdir().unwrap();
        let backend = MockBackend::new(Duration::from_millis(1));
        let stages = Arc::new(Mutex::new(Vec::new()));
        let seen = stages.clone();
        let out = backend
            .generate(
                &job(dir.path(), true),
                &model(),
                ProgressSink::new(move |s| seen.lock().unwrap().push(s.to_string())),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let stages = stages.lock().unwrap().clone();
        assert_eq!(
            stages,
            [
                "loading model",
                "sparse structure 1/2",
                "sparse structure 2/2",
                "shape 1/2",
                "shape 2/2",
                "texture 1/2",
                "texture 2/2",
                "decoding",
                "baking textures",
                "exporting"
            ]
        );
        assert_eq!(out.fixes, vec!["mps_sparse_decoder_cpu"]);
        assert!(out.timings.contains_key("bake"));
        let stats = crate::glb::stats(&dir.path().join("m.glb.partial")).unwrap();
        assert_eq!(stats.triangles, 12);
        assert!(dir.path().join("m.obj.partial").exists());
        assert_eq!(
            backend.loaded_model(),
            None,
            "unloads after the job by default"
        );
    }

    #[tokio::test]
    async fn injected_failure_names_its_stage() {
        let dir = tempfile::tempdir().unwrap();
        let backend = MockBackend::new(Duration::from_millis(1));
        backend.fail_next("shape", "empty_mesh");
        let err = backend
            .generate(
                &job(dir.path(), false),
                &model(),
                ProgressSink::noop(),
                CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(
            matches!(&err, BackendError::Failed { stage, kind, .. } if stage == "shape" && kind == "empty_mesh")
        );
        // Only the next job fails.
        assert!(
            backend
                .generate(
                    &job(dir.path(), false),
                    &model(),
                    ProgressSink::noop(),
                    CancellationToken::new()
                )
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn honours_cancel() {
        let dir = tempfile::tempdir().unwrap();
        let backend = MockBackend::new(Duration::from_secs(10));
        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = backend
            .generate(
                &job(dir.path(), false),
                &model(),
                ProgressSink::noop(),
                cancel,
            )
            .await
            .unwrap_err();
        assert_eq!(err, BackendError::Cancelled);
        assert!(!dir.path().join("m.glb.partial").exists());
    }

    #[tokio::test]
    async fn keep_loaded_holds_the_model_until_unload() {
        let dir = tempfile::tempdir().unwrap();
        let backend = MockBackend::new(Duration::from_millis(1));
        let mut j = job(dir.path(), false);
        j.request.keep_loaded = true;
        backend
            .generate(&j, &model(), ProgressSink::noop(), CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(backend.loaded_model().as_deref(), Some("trellis2-4b"));
        backend.unload().await;
        backend.unload().await;
        assert_eq!(backend.unloads.load(Ordering::SeqCst), 1);
    }
}
