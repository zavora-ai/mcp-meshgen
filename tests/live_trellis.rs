//! Live test on an Apple Silicon Mac with trellis-mac set up (design §8; R1, R7, R8, R9). Run by hand with nothing heavy
//! open (it needs ~17 GB free):
//!
//!   MESHGEN_LIVE_TRELLIS_DIR=~/tools/trellis-mac \
//!   MESHGEN_LIVE_IMAGE=/path/to/cutout.png \
//!   cargo test --release --test live_trellis -- --ignored --nocapture
//!
//! Optional: MESHGEN_LIVE_OUT (output folder; default: a temp dir, kept and printed), MESHGEN_LIVE_HF_CACHE,
//! MESHGEN_LIVE_HEADROOM_MB (default 2000; a 24 GB Mac with everything closed needs ~1000).

use std::path::PathBuf;
use std::time::Instant;

use rmcp::model::{
    CallToolRequestParams, ClientCapabilities, ClientInfo, Implementation, ProtocolVersion,
};
use rmcp::transport::TokioChildProcess;
use rmcp::{ClientHandler, ServiceExt};
use serde_json::{Value, json};

#[derive(Clone)]
struct Client;

impl ClientHandler for Client {
    fn get_info(&self) -> ClientInfo {
        let mut info = ClientInfo::new(
            ClientCapabilities::default(),
            Implementation::new("live-test", "1"),
        );
        info.protocol_version = ProtocolVersion::V_2025_11_25;
        info
    }
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .filter(|v| !v.is_empty())
        .map(|v| mcp_meshgen::config::expand_tilde(&v))
}

#[tokio::test]
#[ignore = "needs trellis-mac, the weights and ~17 GB free; run by hand"]
async fn brazier_end_to_end() {
    let trellis = env_path("MESHGEN_LIVE_TRELLIS_DIR").expect("set MESHGEN_LIVE_TRELLIS_DIR");
    let image = env_path("MESHGEN_LIVE_IMAGE").expect("set MESHGEN_LIVE_IMAGE (a cut-out PNG)");
    let work = tempfile::Builder::new()
        .prefix("meshgen-live-")
        .tempdir()
        .unwrap()
        .keep();
    let out = env_path("MESHGEN_LIVE_OUT").unwrap_or_else(|| work.join("out"));
    std::fs::create_dir_all(&out).unwrap();
    let cfg = work.join("cfg");
    std::fs::create_dir_all(&cfg).unwrap();
    let mut config = format!(
        "allowed_roots = [{img:?}, {out:?}]\n[backends.trellis_mac]\ndir = {trellis:?}\n",
        img = image.parent().unwrap().display().to_string(),
        out = out.display().to_string(),
        trellis = trellis.display().to_string(),
    );
    if let Ok(headroom) = std::env::var("MESHGEN_LIVE_HEADROOM_MB") {
        config = format!(
            "memory_headroom_mb = {}\n{config}",
            headroom.parse::<u64>().expect("a number")
        );
    }
    if let Some(hf) = env_path("MESHGEN_LIVE_HF_CACHE") {
        config = format!("hf_cache = {:?}\n{config}", hf.display().to_string());
    }
    std::fs::write(cfg.join("config.toml"), config).unwrap();
    std::fs::write(cfg.join("models.toml"), mcp_meshgen::config::MODELS_EXAMPLE).unwrap();

    let mut cmd = tokio::process::Command::new(env!("CARGO_BIN_EXE_mcp-meshgen"));
    cmd.env("MCP_MESHGEN_CONFIG_DIR", &cfg)
        .env("RUST_LOG", "info");
    let client = Client
        .serve(TokioChildProcess::new(cmd).unwrap())
        .await
        .unwrap();

    let listed = client
        .call_tool(CallToolRequestParams::new("list_models"))
        .await
        .unwrap();
    let listed = listed.structured_content.unwrap();
    let model = &listed["models"][0];
    println!(
        "list_models: {} {} (needs {} MB free, {} MB available)",
        model["id"], model["status"], model["needs_mb_free"], listed["available_memory_mb"]
    );
    assert_eq!(model["status"], "ready", "{model}");

    let started = Instant::now();
    let args = json!({
        "image": image.display().to_string(),
        "output_dir": out.display().to_string(),
        "seed": 42,
        "previews": true,
        "target_triangles": 20000,
        "size_m": 1.2,
        "fbx": true,
    });
    let result = client
        .call_tool(
            CallToolRequestParams::new("generate_mesh")
                .with_arguments(args.as_object().unwrap().clone()),
        )
        .await
        .unwrap();
    let v: Value = result.structured_content.clone().unwrap();
    println!("{}", serde_json::to_string_pretty(&v).unwrap());
    println!(
        "wall time {:.0} s; files in {}",
        started.elapsed().as_secs_f64(),
        out.display()
    );
    assert_eq!(v["status"], "done", "{v}");

    let triangles = v["stats"]["triangles"].as_u64().unwrap();
    assert!(
        triangles > 0 && triangles <= 20_500,
        "decimated before the bake: {triangles}"
    );
    assert_eq!(v["stats"]["materials"], 1);
    assert!(!v["stats"]["textures"].as_array().unwrap().is_empty());
    for stage in [
        "load",
        "sparse_structure",
        "shape",
        "texture",
        "decode",
        "bake",
        "export",
        "previews",
        "post_process",
        "total",
    ] {
        assert!(
            v["timings"][stage].as_f64().is_some(),
            "timing {stage} missing"
        );
    }
    assert!(v["peak_rss_mb"].as_u64().unwrap() > 1000);
    assert!(
        v["fixes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f == "mps_sparse_decoder_cpu"),
        "{}",
        v["fixes"]
    );
    assert_eq!(v["files"]["previews"].as_array().unwrap().len(), 4);
    for p in v["files"]["previews"].as_array().unwrap() {
        assert!(PathBuf::from(p.as_str().unwrap()).is_file());
    }
    let game = &v["game"];
    let size: Vec<f64> = game["stats"]["size"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let longest = size.iter().cloned().fold(0.0, f64::max);
    assert!(
        (longest - 1.2).abs() < 0.01,
        "engine copy longest side {longest}"
    );
    assert!(
        game["stats"]["bounds_min"][1].as_f64().unwrap().abs() < 1e-3,
        "bottom on the origin"
    );
    assert!(PathBuf::from(v["files"]["game_fbx"].as_str().unwrap()).is_file());
    client.cancel().await.unwrap();
}
