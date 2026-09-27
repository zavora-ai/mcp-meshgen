//! The trellis_mac backend against `fake-worker` (the real spawn, protocol, cancel and keep-alive code paths).
#![cfg(all(feature = "mock", feature = "trellis_mac"))]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mcp_meshgen::backend::trellis_mac::{Launcher, TrellisMacBackend};
use mcp_meshgen::backend::{BackendError, BackendJob, MeshBackend, ProgressSink};
use mcp_meshgen::config::{MODELS_EXAMPLE, TrellisMacConfig};
use mcp_meshgen::registry::{ModelRegistry, ModelSpec};
use mcp_meshgen::request::{ResolvedRequest, SizeAxis};
use tokio_util::sync::CancellationToken;

struct Fixture {
    dir: tempfile::TempDir,
    backend: TrellisMacBackend,
}

fn fixture(mode: &str, idle_unload_s: u64) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let checkout = dir.path().join("trellis-mac");
    std::fs::create_dir_all(checkout.join("TRELLIS.2")).unwrap();
    std::fs::create_dir_all(dir.path().join("out")).unwrap();
    let backend = TrellisMacBackend::with_launcher(
        TrellisMacConfig {
            dir: Some(checkout),
            python: None,
        },
        dir.path().join("cache"),
        idle_unload_s,
        Launcher::Program {
            path: PathBuf::from(env!("CARGO_BIN_EXE_fake-worker")),
            env: vec![
                ("FAKE_WORKER_MODE".into(), mode.into()),
                (
                    "FAKE_WORKER_LOG".into(),
                    dir.path().join("requests.log").display().to_string(),
                ),
            ],
        },
    );
    Fixture { dir, backend }
}

fn model() -> ModelSpec {
    ModelRegistry::from_toml(MODELS_EXAMPLE)
        .unwrap()
        .get("trellis2-4b")
        .unwrap()
        .clone()
}

fn job(root: &Path, id: &str, keep_loaded: bool, obj: bool) -> BackendJob {
    let out = root.join("out");
    BackendJob {
        id: id.into(),
        request: ResolvedRequest {
            model: "trellis2-4b".into(),
            image: root.join("in.png"),
            seed: 42,
            quality: "512".into(),
            texture_size: 1024,
            steps: Some(2),
            target_triangles: 20_000,
            obj,
            previews: false,
            return_preview: false,
            size_m: None,
            size_axis: SizeAxis::Longest,
            fbx: false,
            keep_loaded,
            warnings: vec![],
        },
        out_glb: out.join(format!("{id}.glb.partial")),
        out_obj: obj.then(|| out.join(format!("{id}.obj.partial"))),
        hf_cache: root.join("hub"),
    }
}

fn recorder() -> (ProgressSink, Arc<Mutex<Vec<String>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s = seen.clone();
    (
        ProgressSink::new(move |stage| s.lock().unwrap().push(stage.to_string())),
        seen,
    )
}

#[tokio::test]
async fn success_reports_stages_and_writes_the_glb() {
    let f = fixture("ok", 0);
    let (progress, seen) = recorder();
    let out = f
        .backend
        .generate(
            &job(f.dir.path(), "a", false, true),
            &model(),
            progress,
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen[0], "starting worker");
    assert_eq!(seen[1], "loading model");
    for stage in [
        "sparse structure 2/2",
        "shape 1/2",
        "texture 2/2",
        "decoding",
        "baking textures",
        "exporting",
    ] {
        assert!(
            seen.iter().any(|s| s == stage),
            "{stage} missing from {seen:?}"
        );
    }
    assert_eq!(out.fixes, vec!["mps_sparse_decoder_cpu"]);
    assert_eq!(out.peak_rss_mb, Some(100));
    assert_eq!(out.details["voxels"], 22683);
    assert!(out.timings.contains_key("load"));
    let stats = mcp_meshgen::glb::stats(&f.dir.path().join("out/a.glb.partial")).unwrap();
    assert_eq!(stats.triangles, 12);
    assert_eq!(out.extra_files, vec![f.dir.path().join("out/a.mtl")]);
    assert!(f.dir.path().join("out/a.mtl.partial").exists());
    assert_eq!(
        f.backend.loaded_model(),
        None,
        "the worker exits after the job by default"
    );
    // The request carried the resolved parameters, including the bake-time triangle budget.
    let sent = std::fs::read_to_string(f.dir.path().join("requests.log")).unwrap();
    let req: serde_json::Value = serde_json::from_str(sent.lines().next().unwrap()).unwrap();
    assert_eq!(req["decimation_target"], 20_000);
    assert_eq!(req["pretrained"], "microsoft/TRELLIS.2-4B");
    assert_eq!(req["steps"], 2);
}

async fn failure(mode: &str) -> BackendError {
    let f = fixture(mode, 0);
    f.backend
        .generate(
            &job(f.dir.path(), "a", false, false),
            &model(),
            ProgressSink::noop(),
            CancellationToken::new(),
        )
        .await
        .unwrap_err()
}

#[tokio::test]
async fn explained_errors_keep_stage_kind_hint_and_log() {
    match failure("error:shape:empty_structure").await {
        BackendError::Failed {
            stage,
            kind,
            hint,
            log_tail,
            ..
        } => {
            assert_eq!(
                (stage.as_str(), kind.as_str(), hint.as_str()),
                ("shape", "empty_structure", "fake hint")
            );
            assert!(log_tail.contains("fake worker ready"), "{log_tail}");
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn a_crash_is_reported_with_its_exit_and_traceback() {
    match failure("crash:texture").await {
        BackendError::Failed {
            kind,
            message,
            log_tail,
            ..
        } => {
            assert_eq!(kind, "worker_exited");
            assert!(message.contains("exited with code 3"), "{message}");
            assert!(log_tail.contains("fake crash in texture"), "{log_tail}");
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn the_metal_watchdog_message_upgrades_the_error() {
    match failure("watchdog").await {
        BackendError::Failed { kind, hint, .. } => {
            assert_eq!(kind, "gpu_watchdog");
            assert!(hint.contains("GPU watchdog"), "{hint}");
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn a_protocol_mismatch_is_caught_at_hello() {
    match failure("bad_protocol").await {
        BackendError::Failed { kind, .. } => assert_eq!(kind, "worker_protocol"),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn cancel_stops_the_worker_mid_stage() {
    let f = fixture("slow", 0);
    let cancel = CancellationToken::new();
    let c = cancel.clone();
    let (progress, seen) = recorder();
    let seen2 = seen.clone();
    tokio::spawn(async move {
        loop {
            if seen2.lock().unwrap().iter().any(|s| s.starts_with("shape")) {
                c.cancel();
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    });
    let started = Instant::now();
    let err = f
        .backend
        .generate(
            &job(f.dir.path(), "a", false, false),
            &model(),
            progress,
            cancel,
        )
        .await
        .unwrap_err();
    assert_eq!(err, BackendError::Cancelled);
    assert!(started.elapsed() < Duration::from_secs(10));
    assert!(
        !f.dir.path().join("out/a.glb.partial").exists(),
        "no GLB written"
    );
    assert_eq!(f.backend.loaded_model(), None);
}

#[tokio::test]
async fn keep_loaded_reuses_the_worker_until_unload() {
    let f = fixture("ok", 0);
    let first = f
        .backend
        .generate(
            &job(f.dir.path(), "a", true, false),
            &model(),
            ProgressSink::noop(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert!(first.timings.contains_key("load"));
    assert_eq!(f.backend.loaded_model().as_deref(), Some("trellis2-4b"));
    let (progress, seen) = recorder();
    let second = f
        .backend
        .generate(
            &job(f.dir.path(), "b", true, false),
            &model(),
            progress,
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert!(
        !second.timings.contains_key("load"),
        "the second job ran in the same, already loaded worker"
    );
    assert!(!seen.lock().unwrap().iter().any(|s| s == "starting worker"));
    f.backend.unload().await;
    assert_eq!(f.backend.loaded_model(), None);
    let sent = std::fs::read_to_string(f.dir.path().join("requests.log")).unwrap();
    assert!(sent.lines().last().unwrap().contains("unload"), "{sent}");
}

#[tokio::test]
async fn a_kept_worker_unloads_when_idle() {
    let f = fixture("ok", 1);
    f.backend
        .generate(
            &job(f.dir.path(), "a", true, false),
            &model(),
            ProgressSink::noop(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert!(f.backend.loaded_model().is_some());
    tokio::time::sleep(Duration::from_millis(1800)).await;
    assert_eq!(f.backend.loaded_model(), None);
}
