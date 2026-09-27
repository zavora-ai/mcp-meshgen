//! Stands in for `worker/trellis_mac_worker.py` in tests (feature `mock`; never installed).
//!
//! Speaks the same JSON-lines protocol and writes the test cube as the GLB. `FAKE_WORKER_MODE` picks the behaviour:
//!   ok (default)            the real event sequence, then a result
//!   error:<stage>:<kind>    an explained error at that stage
//!   crash:<stage>           exit(3) during that stage, after writing the partial GLB
//!   watchdog                print the Metal watchdog message to stderr, then an empty_mesh error at decode
//!   bad_protocol            say hello with protocol 99
//!   slow                    300 ms per step (for cancel tests)
//! `FAKE_WORKER_LOG` (optional): append each received request line to this file.

use std::io::{BufRead, Write};
use std::time::Duration;

fn emit(value: serde_json::Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{value}");
    let _ = out.flush();
}

fn main() {
    let mode = std::env::var("FAKE_WORKER_MODE").unwrap_or_else(|_| "ok".into());
    let log = std::env::var("FAKE_WORKER_LOG").ok();
    let protocol = if mode == "bad_protocol" { 99 } else { 1 };
    emit(serde_json::json!({"event": "hello", "protocol": protocol, "pid": std::process::id()}));
    eprintln!("fake worker ready (mode {mode})");

    let step_delay = Duration::from_millis(if mode == "slow" { 300 } else { 5 });
    let mut loaded = false;
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        if let Some(path) = &log
            && let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
        {
            let _ = writeln!(f, "{line}");
        }
        let Ok(req) = serde_json::from_str::<serde_json::Value>(&line) else {
            emit(
                serde_json::json!({"event": "error", "stage": "protocol", "kind": "bad_request", "message": "not JSON"}),
            );
            continue;
        };
        let id = req["id"].as_str().unwrap_or("").to_string();
        match req["op"].as_str() {
            Some("unload") => {
                emit(serde_json::json!({"event": "unloaded", "id": id}));
                return;
            }
            Some("ping") => emit(serde_json::json!({"event": "pong", "id": id})),
            Some("generate") => {
                let mut timings = serde_json::Map::new();
                emit(serde_json::json!({"event": "loading", "id": id}));
                if !loaded {
                    timings.insert("load".into(), 0.01.into());
                    loaded = true;
                }
                let total = req["steps"].as_u64().unwrap_or(3);
                let fail_at = |stage: &str| -> bool {
                    if let Some(rest) = mode.strip_prefix("error:") {
                        let (s, kind) = rest.split_once(':').unwrap_or((rest, "worker_exception"));
                        if s == stage {
                            emit(
                                serde_json::json!({"event": "error", "id": id, "stage": s, "kind": kind,
                                "message": format!("fake {kind}"), "hint": "fake hint"}),
                            );
                            return true;
                        }
                    }
                    if mode.strip_prefix("crash:") == Some(stage) {
                        if let Some(glb) = req["out_glb"].as_str() {
                            let _ = std::fs::write(glb, b"half a glb");
                        }
                        eprintln!("Traceback (most recent call last):\n  fake crash in {stage}");
                        std::process::exit(3);
                    }
                    false
                };
                let mut failed = false;
                'stages: for stage in ["sparse_structure", "shape", "texture"] {
                    for step in 0..=total {
                        emit(
                            serde_json::json!({"event": "progress", "id": id, "stage": stage, "step": step, "total": total}),
                        );
                        if step == 1 && fail_at(stage) {
                            failed = true;
                            break 'stages;
                        }
                        std::thread::sleep(step_delay);
                    }
                    timings.insert(
                        stage.into(),
                        (step_delay.as_secs_f64() * total as f64).into(),
                    );
                }
                if failed {
                    continue;
                }
                for stage in ["decode", "bake", "export"] {
                    emit(serde_json::json!({"event": "progress", "id": id, "stage": stage}));
                    if mode == "watchdog" && stage == "decode" {
                        eprintln!(
                            "Error: command buffer exited with error status.\n\tThe Metal Performance Shaders operations encoded on it may not have completed.\n\tError: (null)\n\tImpacting Interactivity (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)"
                        );
                        std::thread::sleep(Duration::from_millis(50));
                        emit(
                            serde_json::json!({"event": "error", "id": id, "stage": "decode", "kind": "empty_mesh",
                            "message": "the decoder produced an empty mesh", "hint": "watchdog?"}),
                        );
                        failed = true;
                        break;
                    }
                    if fail_at(stage) {
                        failed = true;
                        break;
                    }
                    std::thread::sleep(step_delay);
                }
                if failed {
                    continue;
                }
                let glb = req["out_glb"].as_str().expect("out_glb");
                std::fs::write(glb, mcp_meshgen::glb::cube()).expect("write glb");
                let mut files = Vec::new();
                if let Some(obj) = req["out_obj"].as_str() {
                    let final_obj = obj.trim_end_matches(".partial");
                    let stem = std::path::Path::new(final_obj)
                        .file_stem()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();
                    let dir = std::path::Path::new(final_obj).parent().unwrap();
                    std::fs::write(
                        obj,
                        format!("mtllib {stem}.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"),
                    )
                    .unwrap();
                    let mtl = dir.join(format!("{stem}.mtl"));
                    std::fs::write(format!("{}.partial", mtl.display()), "newmtl m\n").unwrap();
                    files.push(mtl.display().to_string());
                }
                timings.insert("total".into(), 0.1.into());
                emit(
                    serde_json::json!({"event": "result", "id": id, "timings": timings, "peak_rss_mb": 100,
                    "fixes": ["mps_sparse_decoder_cpu"], "warnings": [], "details": {"voxels": 22683, "bake": "fake"},
                    "files": files}),
                );
            }
            _ => emit(
                serde_json::json!({"event": "error", "id": id, "stage": "protocol", "kind": "bad_request",
                "message": "unknown op"}),
            ),
        }
    }
}
