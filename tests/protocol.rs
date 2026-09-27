//! Protocol tests (design §8): spawn the real binary over stdio with the mock backend and drive it as MCP 2025-11-25
//! (legacy) and 2026-07-28 (Tasks) clients.
#![cfg(feature = "mock")]

use std::path::{Path, PathBuf};
use std::time::Duration;

use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ClientCapabilities, ClientInfo,
    GetTaskParams, Implementation, ProtocolVersion, TaskPayload,
};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::TokioChildProcess;
use rmcp::{ClientHandler, ServiceExt};
use serde_json::{Value, json};

#[derive(Clone)]
struct LegacyClient;

impl ClientHandler for LegacyClient {
    fn get_info(&self) -> ClientInfo {
        let mut info = ClientInfo::new(
            ClientCapabilities::default(),
            Implementation::new("legacy-test", "1"),
        );
        info.protocol_version = ProtocolVersion::V_2025_11_25;
        info
    }
}

#[derive(Clone)]
struct TaskClient;

impl ClientHandler for TaskClient {
    fn get_info(&self) -> ClientInfo {
        let mut info = ClientInfo::new(
            ClientCapabilities::builder().enable_tasks().build(),
            Implementation::new("task-test", "1"),
        );
        info.protocol_version = ProtocolVersion::V_2026_07_28;
        info
    }
}

struct Env {
    dir: tempfile::TempDir,
    cfg: PathBuf,
    root: PathBuf,
}

impl Env {
    fn cutout(&self) -> String {
        self.root.join("refs/brazier.png").display().to_string()
    }

    fn flat(&self) -> String {
        self.root.join("refs/flat.png").display().to_string()
    }
}

/// A temp config: one allowed root with a cut-out and a flat image, mock models in every status.
fn env() -> Env {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().canonicalize().unwrap();
    let cfg = base.join(".meshgen");
    let root = base.join("game");
    std::fs::create_dir_all(&cfg).unwrap();
    std::fs::create_dir_all(root.join("refs")).unwrap();
    let cutout = image::RgbaImage::from_fn(64, 64, |x, y| {
        let inside = (16..48).contains(&x) && (8..56).contains(&y);
        image::Rgba([180, 140, 60, if inside { 255 } else { 0 }])
    });
    cutout.save(root.join("refs/brazier.png")).unwrap();
    image::RgbImage::from_pixel(64, 64, image::Rgb([200, 30, 30]))
        .save(root.join("refs/flat.png"))
        .unwrap();
    std::fs::write(
        cfg.join("config.toml"),
        format!(
            "allowed_roots = [\"game\"]\ndefault_model = \"mock-mesh\"\nmemory_headroom_mb = 0\nhf_cache = {hf:?}\nheavy_processes = []\n",
            hf = base.join("hf").display().to_string(),
        ),
    )
    .unwrap();
    std::fs::write(
        cfg.join("models.toml"),
        r#"
[[model]]
id = "mock-mesh"
backend = "mock"
quality = ["512", "1024"]
defaults = { quality = "512", texture_size = 1024 }
texture_sizes = [512, 1024]
memory_mb = 1
typical_s = 1
fixes = ["mps_sparse_decoder_cpu"]
licence = [{ component = "mock", licence = "MIT", commercial = true }]

[[model]]
id = "needs-weights"
backend = "mock"
quality = ["512"]
defaults = { quality = "512", texture_size = 1024 }
texture_sizes = [1024]
memory_mb = 1
typical_s = 1
licence = [{ component = "weights", licence = "MIT", commercial = true }]
files = [{ repo = "org/model", paths = ["pipeline.json", "ckpts/a.safetensors"], mb = 7000 }]

[[model]]
id = "gated"
backend = "mock"
quality = ["512"]
defaults = { quality = "512", texture_size = 1024 }
texture_sizes = [1024]
memory_mb = 1
typical_s = 1
licence = [{ component = "encoder", licence = "Custom", commercial = false }]
files = [{ repo = "org/gated-encoder", paths = ["model.safetensors"], mb = 1200, gated = true }]

[[model]]
id = "no-backend"
backend = "cuda_official"
quality = ["512"]
defaults = { quality = "512", texture_size = 1024 }
texture_sizes = [1024]
memory_mb = 1
typical_s = 1
licence = [{ component = "x", licence = "MIT", commercial = true }]

[[model]]
id = "huge"
backend = "mock"
quality = ["512"]
defaults = { quality = "512", texture_size = 1024 }
texture_sizes = [1024]
memory_mb = 100000000
typical_s = 1
licence = [{ component = "x", licence = "MIT", commercial = true }]
"#,
    )
    .unwrap();
    Env { dir, cfg, root }
}

async fn spawn_with<C: ClientHandler>(
    client: C,
    env: &Env,
    extra: &[(&str, &str)],
) -> RunningService<RoleClient, C> {
    let mut cmd = tokio::process::Command::new(env!("CARGO_BIN_EXE_mcp-meshgen"));
    cmd.env("MCP_MESHGEN_CONFIG_DIR", &env.cfg)
        .env("MCP_MESHGEN_CACHE_DIR", env.dir.path().join("cache"))
        .env("MCP_MESHGEN_MOCK_STEP_MS", "20")
        .env("MCP_MESHGEN_MOCK_POST", "1")
        .env("RUST_LOG", "warn")
        .current_dir(std::env::temp_dir());
    for (k, v) in extra {
        cmd.env(k, v);
    }
    client
        .serve(TokioChildProcess::new(cmd).unwrap())
        .await
        .unwrap()
}

async fn spawn<C: ClientHandler>(client: C, env: &Env) -> RunningService<RoleClient, C> {
    spawn_with(client, env, &[]).await
}

fn call(name: &'static str, v: Value) -> CallToolRequestParams {
    CallToolRequestParams::new(name).with_arguments(v.as_object().unwrap().clone())
}

fn sc(result: &CallToolResult) -> &Value {
    result
        .structured_content
        .as_ref()
        .expect("structured content")
}

/// Every file under `dir`, recursively, as names relative to it.
fn files(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().filter_map(Result::ok) {
            if e.path().is_dir() {
                stack.push(e.path());
            } else {
                out.push(e.path().strip_prefix(dir).unwrap().display().to_string());
            }
        }
    }
    out.sort();
    out
}

async fn poll_until_terminal<C: ClientHandler>(
    client: &RunningService<RoleClient, C>,
    job_id: &str,
) -> CallToolResult {
    loop {
        let r = client
            .call_tool(call("get_job", json!({"job_id": job_id})))
            .await
            .unwrap();
        let status = sc(&r)["status"].as_str().unwrap().to_string();
        if !matches!(status.as_str(), "queued" | "running") {
            return r;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
}

#[tokio::test]
async fn legacy_client_converts_with_previews_and_an_engine_copy() {
    let env = env();
    let client = spawn(LegacyClient, &env).await;

    let manifest = adk_mcp_sdk::ServerManifest::from_toml(mcp_meshgen::MANIFEST_TOML).unwrap();
    let mut want: Vec<_> = manifest.tools.iter().map(|t| t.name.clone()).collect();
    want.sort();
    let mut got: Vec<_> = client
        .list_tools(None)
        .await
        .unwrap()
        .tools
        .iter()
        .map(|t| t.name.to_string())
        .collect();
    got.sort();
    assert_eq!(got, want);

    let result = client
        .call_tool(call(
            "generate_mesh",
            json!({
                "image": "game/refs/brazier.png",
                "seed": 99,
                "output_dir": "game/meshes",
                "obj": true,
                "return_preview": true,
                "size_m": 1.2,
                "fbx": true
            }),
        ))
        .await
        .unwrap();
    let v = sc(&result);
    assert_eq!(v["status"], "done");
    assert_eq!(v["params"]["seed"], 99);
    assert_eq!(v["stats"]["triangles"], 12);
    assert_eq!(v["fixes"], json!(["mps_sparse_decoder_cpu"]));
    assert_eq!(v["input"]["sha256"].as_str().unwrap().len(), 64);
    assert_eq!(v["licences"][0]["licence"], "MIT");
    let glb = PathBuf::from(v["files"]["glb"].as_str().unwrap());
    assert_eq!(glb, env.root.join("meshes/brazier-99.glb"));
    assert!(glb.exists());
    assert_eq!(v["files"]["previews"].as_array().unwrap().len(), 4);
    assert!(
        v["files"]["game_fbx"]
            .as_str()
            .unwrap()
            .ends_with("brazier-99_game.fbx")
    );
    assert!(
        result.content.iter().any(
            |c| matches!(c, rmcp::model::ContentBlock::Image(i) if i.mime_type == "image/png")
        )
    );
    let sidecar: Value =
        serde_json::from_slice(&std::fs::read(env.root.join("meshes/brazier-99.json")).unwrap())
            .unwrap();
    assert_eq!(sidecar["files"]["glb"], "brazier-99.glb");
    assert!(
        files(&env.root).iter().all(|f| !f.ends_with(".partial")),
        "{:?}",
        files(&env.root)
    );
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn task_client_gets_a_task_with_stage_status() {
    let env = env();
    let client = spawn(TaskClient, &env).await;
    let response = client
        .call_tool_once(call(
            "generate_mesh",
            json!({"image": env.cutout(), "steps": 6}),
        ))
        .await
        .unwrap();
    let created = match response {
        CallToolResponse::Task(created) => created,
        other => panic!("expected a task, got {other:?}"),
    };
    assert_eq!(created.task.ttl_ms, Some(7_200_000));
    let mut messages = Vec::new();
    let result = loop {
        let task = client
            .peer()
            .get_task(GetTaskParams::new(created.task.task_id.clone()))
            .await
            .unwrap()
            .task;
        if let Some(m) = &task.task.status_message {
            messages.push(m.clone());
        }
        if task.status().is_terminal() {
            match task.payload {
                TaskPayload::Completed { result } => break result,
                other => panic!("unexpected terminal payload: {other:?}"),
            }
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    };
    assert_eq!(result["structuredContent"]["status"], "done");
    assert!(
        messages.iter().any(|m| m.starts_with("shape ")),
        "status messages seen: {messages:?}"
    );
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn no_wait_then_poll_get_job_on_both_protocols() {
    let env = env();
    for legacy in [true, false] {
        let job = |client_result: CallToolResult| {
            sc(&client_result)["job_id"].as_str().unwrap().to_string()
        };
        let args = json!({"image": env.cutout(), "wait": false});
        let done = if legacy {
            let client = spawn(LegacyClient, &env).await;
            let id = job(client.call_tool(call("generate_mesh", args)).await.unwrap());
            let r = poll_until_terminal(&client, &id).await;
            client.cancel().await.unwrap();
            r
        } else {
            // A Tasks client gets a task even with wait=false; the task completes at once with the job_id,
            // and get_job (a plain tool) works the same on this protocol.
            let client = spawn(TaskClient, &env).await;
            let created = match client
                .call_tool_once(call("generate_mesh", args))
                .await
                .unwrap()
            {
                CallToolResponse::Task(created) => created,
                other => panic!("expected a task, got {other:?}"),
            };
            let payload = loop {
                let task = client
                    .peer()
                    .get_task(GetTaskParams::new(created.task.task_id.clone()))
                    .await
                    .unwrap()
                    .task;
                if task.status().is_terminal() {
                    match task.payload {
                        TaskPayload::Completed { result } => break result,
                        other => panic!("unexpected terminal payload: {other:?}"),
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            };
            let id = payload["structuredContent"]["job_id"]
                .as_str()
                .unwrap()
                .to_string();
            let r = poll_until_terminal(&client, &id).await;
            client.cancel().await.unwrap();
            r
        };
        assert_eq!(sc(&done)["status"], "done", "legacy={legacy}");
        assert!(
            sc(&done)["result"]["files"]["glb"]
                .as_str()
                .unwrap()
                .ends_with(".glb")
        );
    }
}

#[tokio::test]
async fn cancel_mid_stage_leaves_no_partials() {
    let env = env();
    let client = spawn(LegacyClient, &env).await;
    let queued = client
        .call_tool(call(
            "generate_mesh",
            json!({"image": env.cutout(), "steps": 50, "wait": false, "output_dir": "game/slow"}),
        ))
        .await
        .unwrap();
    let job_id = sc(&queued)["job_id"].as_str().unwrap().to_string();
    loop {
        let r = client
            .call_tool(call("get_job", json!({"job_id": job_id})))
            .await
            .unwrap();
        if sc(&r)["progress_text"]
            .as_str()
            .is_some_and(|p| p.starts_with("shape"))
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        files(&env.root.join("slow"))
            .iter()
            .any(|f| f.ends_with(".partial")),
        "reserved while running"
    );
    client
        .call_tool(call("cancel_job", json!({"job_id": job_id})))
        .await
        .unwrap();
    let r = poll_until_terminal(&client, &job_id).await;
    assert_eq!(sc(&r)["status"], "cancelled");
    assert!(
        files(&env.root.join("slow")).is_empty(),
        "{:?}",
        files(&env.root.join("slow"))
    );
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn backend_failures_come_back_explained() {
    let env = env();
    let client = spawn_with(
        LegacyClient,
        &env,
        &[("MCP_MESHGEN_MOCK_FAIL", "sparse_structure:empty_structure")],
    )
    .await;
    let r = client
        .call_tool(call(
            "generate_mesh",
            json!({"image": env.cutout(), "output_dir": "game/fail"}),
        ))
        .await
        .unwrap();
    assert_eq!(r.is_error, Some(true));
    let v = sc(&r);
    assert_eq!(v["status"], "failed");
    assert_eq!(v["error"]["code"], "backend_failed");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(msg.contains("sparse_structure (empty_structure)"), "{msg}");
    assert!(!env.root.join("fail").exists() || files(&env.root.join("fail")).is_empty());
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn rejections_say_what_to_do() {
    let env = env();
    let client = spawn(LegacyClient, &env).await;
    let err = |args: Value| {
        let client = &client;
        async move {
            client
                .call_tool(call("generate_mesh", args))
                .await
                .unwrap_err()
                .to_string()
        }
    };
    let e = err(json!({"image": env.flat()})).await;
    assert!(
        e.contains("no alpha channel") && e.contains("edit_image"),
        "{e}"
    );
    let outside = tempfile::tempdir().unwrap();
    let foreign = outside.path().join("x.png");
    std::fs::copy(env.cutout(), &foreign).unwrap();
    let e = err(json!({"image": foreign.display().to_string()})).await;
    assert!(e.contains("outside the allowed roots"), "{e}");
    let e = err(json!({"image": env.cutout(), "model": "needs-weights"})).await;
    assert!(
        e.contains("missing_files")
            && e.contains("hf download org/model pipeline.json ckpts/a.safetensors"),
        "{e}"
    );
    let e = err(json!({"image": env.cutout(), "model": "gated"})).await;
    assert!(
        e.contains("needs_licence_acceptance")
            && e.contains("https://huggingface.co/org/gated-encoder"),
        "{e}"
    );
    let e = err(json!({"image": env.cutout(), "model": "no-backend"})).await;
    assert!(
        e.contains("backend_unavailable") && e.contains("not in this build"),
        "{e}"
    );
    let e = err(json!({"image": env.cutout(), "quality": "4096"})).await;
    assert!(e.contains("`quality`"), "{e}");
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn insufficient_memory_refuses_the_job() {
    let env = env();
    let client = spawn(LegacyClient, &env).await;
    let r = client
        .call_tool(call(
            "generate_mesh",
            json!({"image": env.cutout(), "model": "huge", "output_dir": "game/huge"}),
        ))
        .await
        .unwrap();
    assert_eq!(r.is_error, Some(true));
    assert_eq!(sc(&r)["error"]["code"], "insufficient_memory");
    assert!(!env.root.join("huge").exists(), "nothing reserved");
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn list_models_reports_every_status_without_loading() {
    let env = env();
    let client = spawn(TaskClient, &env).await;
    let r = client
        .call_tool(call("list_models", json!({})))
        .await
        .unwrap();
    let v = sc(&r);
    let status = |id: &str| {
        v["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == id)
            .unwrap_or_else(|| panic!("{id}"))
            .clone()
    };
    assert_eq!(status("mock-mesh")["status"], "ready");
    assert_eq!(status("mock-mesh")["loaded"], false);
    let needs = status("needs-weights");
    assert_eq!(needs["status"], "missing_files");
    assert_eq!(needs["download_mb"], 7000);
    // The test cache isn't the default one, so the command names it.
    let command = needs["missing"][0]["command"].as_str().unwrap();
    assert!(
        command.starts_with("hf download org/model pipeline.json ckpts/a.safetensors --cache-dir "),
        "{command}"
    );
    assert_eq!(status("gated")["status"], "needs_licence_acceptance");
    assert_eq!(status("gated")["commercial_use"], false);
    assert_eq!(status("no-backend")["status"], "backend_unavailable");
    assert_eq!(v["previews_and_engine_copy"]["status"], "ready");
    assert_eq!(v["default_model"], "mock-mesh");
    assert_eq!(v["jobs_busy"], 0);
    client.cancel().await.unwrap();
}
