//! Headless Blender for previews and the engine copy (design §4.4; R8, R9, R10).
//!
//! Runs `blender -b --factory-startup --python-exit-code 1 --python <script> -- <json>` after the worker has exited,
//! so the two never hold memory together. Scripts are embedded and written to the cache dir once per version.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use super::{GameCopy, PostProcessor};
use crate::backend::process::{find_executable, kill_tree, own_process_group};
use crate::output::{MeshNames, PREVIEW_VIEWS};
use crate::request::{ResolvedRequest, SizeAxis};

const COMMON: &str = include_str!("../../scripts/blender/common.py");
const PREVIEWS: &str = include_str!("../../scripts/blender/previews.py");
const POSTPROCESS: &str = include_str!("../../scripts/blender/postprocess.py");
const RESULT_MARKER: &str = "MESHGEN_RESULT ";
const RUN_TIMEOUT: Duration = Duration::from_secs(900);

pub struct BlenderPost {
    blender: Result<PathBuf, String>,
    cache_dir: PathBuf,
    /// Preview size in pixels and EEVEE samples.
    pub preview_size: u32,
    pub preview_samples: u32,
}

#[derive(Debug, Deserialize)]
struct ScriptResult {
    ok: bool,
    #[serde(default)]
    error: String,
    #[serde(default)]
    scale: f64,
}

impl BlenderPost {
    /// `configured`: the config's `blender` (a path or a bare name), else discovery.
    pub fn new(configured: Option<&Path>, cache_dir: PathBuf) -> Self {
        Self {
            blender: discover(configured),
            cache_dir,
            preview_size: 768,
            preview_samples: 32,
        }
    }

    pub fn blender_path(&self) -> Option<&Path> {
        self.blender.as_deref().ok()
    }

    fn script(&self, name: &str, body: &str) -> Result<PathBuf, String> {
        let source = format!("{COMMON}\n# ---- {name} ----\n{body}");
        let hash = format!("{:x}", Sha256::digest(source.as_bytes()));
        let dir = self.cache_dir.join("blender");
        let path = dir.join(format!("{name}-{}.py", &hash[..12]));
        if !path.is_file() {
            std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            // Unique per writer: two threads (or servers) may write the same script at once.
            let tmp = dir.join(format!(
                ".{name}-{}.{:016x}.tmp",
                &hash[..12],
                rand::random::<u64>()
            ));
            std::fs::write(&tmp, &source).map_err(|e| format!("{}: {e}", tmp.display()))?;
            std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
        }
        Ok(path)
    }

    async fn run(
        &self,
        name: &str,
        body: &str,
        args: serde_json::Value,
        cancel: &CancellationToken,
    ) -> Result<ScriptResult, String> {
        let blender = self.blender.clone()?;
        let script = self.script(name, body)?;
        let mut cmd = Command::new(&blender);
        cmd.args([
            "-b",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
        ])
        .arg(&script)
        .arg("--")
        .arg(args.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
        own_process_group(&mut cmd);
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("couldn't start {}: {e}", blender.display()))?;
        let mut stdout = child.stdout.take().expect("piped");
        let mut stderr = child.stderr.take().expect("piped");
        let read_out = async {
            let mut s = String::new();
            let _ = stdout.read_to_string(&mut s).await;
            s
        };
        let read_err = async {
            let mut s = String::new();
            let _ = stderr.read_to_string(&mut s).await;
            s
        };
        let finished = async { tokio::join!(read_out, read_err, child.wait()) };
        let (out, err, status) = tokio::select! {
            r = finished => r,
            _ = cancel.cancelled() => {
                kill_tree(&mut child).await;
                return Err("cancelled".into());
            }
            _ = tokio::time::sleep(RUN_TIMEOUT) => {
                kill_tree(&mut child).await;
                return Err(format!("Blender {name} took longer than {} s", RUN_TIMEOUT.as_secs()));
            }
        };
        let parsed = out
            .lines()
            .rev()
            .find_map(|l| l.strip_prefix(RESULT_MARKER))
            .and_then(|json| serde_json::from_str::<ScriptResult>(json).ok());
        match (parsed, status) {
            (Some(r), _) if r.ok => Ok(r),
            (Some(r), _) => Err(format!("Blender {name}: {}", r.error)),
            (None, status) => {
                let tail: Vec<&str> = out
                    .lines()
                    .chain(err.lines())
                    .filter(|l| !l.trim().is_empty())
                    .collect();
                let tail = tail[tail.len().saturating_sub(8)..].join(" | ");
                Err(format!(
                    "Blender {name} exited ({}) without a result: {tail}",
                    status
                        .map(|s| s.to_string())
                        .unwrap_or_else(|e| e.to_string())
                ))
            }
        }
    }
}

#[async_trait::async_trait]
impl PostProcessor for BlenderPost {
    fn availability(&self) -> Result<(), String> {
        self.blender.as_ref().map(|_| ()).map_err(|e| e.clone())
    }

    async fn previews(
        &self,
        glb: &Path,
        names: &MeshNames,
        cancel: &CancellationToken,
    ) -> Result<Vec<PathBuf>, String> {
        let finals: Vec<(&str, PathBuf)> = PREVIEW_VIEWS
            .iter()
            .map(|v| (*v, names.preview(v)))
            .collect();
        let out: serde_json::Map<String, serde_json::Value> = finals
            .iter()
            .map(|(view, path)| {
                (
                    view.to_string(),
                    names.partial(path).display().to_string().into(),
                )
            })
            .collect();
        let args = serde_json::json!({
            "glb": glb, "out": out, "size": self.preview_size, "samples": self.preview_samples,
            "engine": "BLENDER_EEVEE",
        });
        self.run("previews", PREVIEWS, args, cancel).await?;
        for (_, path) in &finals {
            if !names.partial(path).is_file() {
                return Err(format!(
                    "Blender didn't write {}",
                    names.partial(path).display()
                ));
            }
        }
        Ok(finals.into_iter().map(|(_, p)| p).collect())
    }

    async fn game_copy(
        &self,
        glb: &Path,
        names: &MeshNames,
        request: &ResolvedRequest,
        cancel: &CancellationToken,
    ) -> Result<GameCopy, String> {
        let out_glb = names.partial(&names.game_glb());
        let out_fbx = request.fbx.then(|| names.partial(&names.game_fbx()));
        let args = serde_json::json!({
            "glb": glb,
            "out_glb": out_glb,
            "out_fbx": out_fbx,
            "size_m": request.size_m,
            "size_axis": match request.size_axis { SizeAxis::Longest => "longest", SizeAxis::Height => "height" },
        });
        let r = self.run("postprocess", POSTPROCESS, args, cancel).await?;
        let stats = crate::glb::stats(&out_glb)?;
        if out_fbx.as_ref().is_some_and(|f| !f.is_file()) {
            return Err("Blender didn't write the FBX".into());
        }
        Ok(GameCopy {
            glb: names.game_glb(),
            fbx: request.fbx.then(|| names.game_fbx()),
            scale: r.scale,
            stats,
        })
    }
}

/// The configured executable, else `blender` on PATH, else the usual install locations for this platform.
pub fn discover(configured: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(path) = configured {
        return find_executable(path).ok_or_else(|| {
            format!(
                "Blender not found at the configured `blender` ({}); fix it in config.toml",
                path.display()
            )
        });
    }
    if let Some(found) = find_executable(Path::new("blender")) {
        return Ok(found);
    }
    platform_candidates()
        .into_iter()
        .find(|p| p.is_file())
        .ok_or_else(|| {
            "Blender not found: install it (blender.org) or set `blender` in config.toml"
                .to_string()
        })
}

fn platform_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if cfg!(target_os = "macos") {
        out.push(PathBuf::from(
            "/Applications/Blender.app/Contents/MacOS/Blender",
        ));
        out.push(crate::config::home_dir().join("Applications/Blender.app/Contents/MacOS/Blender"));
    } else if cfg!(windows) {
        for var in ["ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"] {
            if let Ok(base) = std::env::var(var) {
                out.extend(newest_versions(
                    &Path::new(&base).join("Blender Foundation"),
                    "blender.exe",
                ));
            }
        }
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            out.extend(newest_versions(
                &Path::new(&local)
                    .join("Programs")
                    .join("Blender Foundation"),
                "blender.exe",
            ));
        }
    } else {
        for p in [
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "/snap/bin/blender",
            "/var/lib/flatpak/exports/bin/org.blender.Blender",
        ] {
            out.push(PathBuf::from(p));
        }
        out.push(
            crate::config::home_dir().join(".local/share/flatpak/exports/bin/org.blender.Blender"),
        );
    }
    out
}

/// `<base>/Blender 4.2/<exe>`, `<base>/Blender 5.2/<exe>`, … newest version first.
fn newest_versions(base: &Path, exe: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(base) else {
        return Vec::new();
    };
    let mut dirs: Vec<(Vec<u32>, PathBuf)> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .map(|p| {
            let version: Vec<u32> = p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .split(|c: char| !c.is_ascii_digit())
                .filter_map(|n| n.parse().ok())
                .collect();
            (version, p.join(exe))
        })
        .collect();
    dirs.sort_by(|a, b| b.0.cmp(&a.0));
    dirs.into_iter().map(|(_, p)| p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_path_must_exist() {
        let err = discover(Some(Path::new("/definitely/not/blender"))).unwrap_err();
        assert!(err.contains("configured `blender`"), "{err}");
        let here = std::env::current_exe().unwrap();
        assert_eq!(discover(Some(&here)).unwrap(), here);
    }

    #[test]
    fn newest_version_first() {
        let base = tempfile::tempdir().unwrap();
        for v in ["Blender 4.2", "Blender 5.2", "Blender 4.10"] {
            std::fs::create_dir_all(base.path().join(v)).unwrap();
        }
        let found = newest_versions(base.path(), "blender.exe");
        let names: Vec<String> = found
            .iter()
            .map(|p| {
                p.parent()
                    .unwrap()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(names, ["Blender 5.2", "Blender 4.10", "Blender 4.2"]);
    }

    #[test]
    fn scripts_carry_the_shared_helpers() {
        let cache = tempfile::tempdir().unwrap();
        let post = BlenderPost {
            blender: Err("none".into()),
            cache_dir: cache.path().into(),
            preview_size: 64,
            preview_samples: 1,
        };
        let path = post.script("previews", PREVIEWS).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("def import_glb") && text.contains("VIEWS = {"));
        assert_eq!(post.script("previews", PREVIEWS).unwrap(), path);
        assert!(post.availability().is_err());
    }
}
