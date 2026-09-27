//! Server configuration (design §3): config dir discovery, allowed roots, limits and backend settings (R10).

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::MeshGenError;

pub const CONFIG_EXAMPLE: &str = include_str!("../config.example.toml");
pub const MODELS_EXAMPLE: &str = include_str!("../models.example.toml");

/// Environment variable overriding the config directory.
pub const CONFIG_DIR_ENV: &str = "MCP_MESHGEN_CONFIG_DIR";
/// Environment variable overriding the cache root (tests and sandboxes). It moves the shared GPU lock too, so a server
/// started with it no longer coordinates with sister servers.
pub const CACHE_DIR_ENV: &str = "MCP_MESHGEN_CACHE_DIR";

const APP_DIR: &str = "mcp-meshgen";
const DEFAULT_ROOT: &str = "~/Documents/mcp-meshgen";
const DEFAULT_HEAVY: [&str; 6] = [
    "UnrealEditor",
    "Blender",
    "Unity",
    "sd-server",
    "mcp-imagegen",
    "ollama",
];

#[derive(Debug, Clone)]
pub struct Config {
    pub config_dir: PathBuf,
    /// The platform cache root (`~/Library/Caches`, `$XDG_CACHE_HOME`, `%LOCALAPPDATA%`), shared with sister servers.
    pub cache_root: PathBuf,
    pub allowed_roots: Vec<PathBuf>,
    /// `None` = write next to the input image (always inside a root, since the input is).
    pub default_output_dir: Option<PathBuf>,
    pub default_model: String,
    pub hf_cache: PathBuf,
    /// `None` = discovery (§4.4).
    pub blender: Option<PathBuf>,
    pub memory_headroom_mb: u64,
    pub keep_loaded: bool,
    /// 0 = unload only on `unload_models`.
    pub idle_unload_s: u64,
    pub max_queue: usize,
    pub job_ttl_s: u64,
    pub heavy_processes: Vec<String>,
    pub input: InputThresholds,
    pub trellis_mac: TrellisMacConfig,
}

/// Cut-out check thresholds (§4.3; R2).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct InputThresholds {
    pub min_transparent: f64,
    pub max_transparent: f64,
    pub min_opaque: f64,
    pub opaque_alpha: u8,
}

impl Default for InputThresholds {
    fn default() -> Self {
        Self {
            min_transparent: 0.05,
            max_transparent: 0.95,
            min_opaque: 0.02,
            opaque_alpha: 205,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrellisMacConfig {
    /// The trellis-mac checkout; `None` = not configured.
    pub dir: Option<PathBuf>,
    /// Explicit venv python; `None` = the venv inside `dir`.
    pub python: Option<PathBuf>,
}

impl TrellisMacConfig {
    /// The python to run the worker with: `python`, else `<dir>/.venv/bin/python` (`<dir>\.venv\Scripts\python.exe` on Windows).
    pub fn python_path(&self) -> Option<PathBuf> {
        self.python
            .clone()
            .or_else(|| self.dir.as_deref().map(venv_python))
    }
}

/// The interpreter inside `<dir>/.venv` for this platform.
pub fn venv_python(dir: &Path) -> PathBuf {
    if cfg!(windows) {
        dir.join(".venv").join("Scripts").join("python.exe")
    } else {
        dir.join(".venv").join("bin").join("python")
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    allowed_roots: Option<Vec<String>>,
    default_output_dir: Option<String>,
    default_model: Option<String>,
    hf_cache: Option<String>,
    blender: Option<String>,
    memory_headroom_mb: Option<u64>,
    keep_loaded: Option<bool>,
    idle_unload_s: Option<u64>,
    max_queue: Option<usize>,
    job_ttl_s: Option<u64>,
    heavy_processes: Option<Vec<String>>,
    #[serde(default)]
    input: InputThresholds,
    #[serde(default)]
    backends: RawBackends,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBackends {
    #[serde(default)]
    trellis_mac: RawTrellisMac,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTrellisMac {
    dir: Option<String>,
    python: Option<String>,
}

impl Config {
    /// Resolve the config directory: `$MCP_MESHGEN_CONFIG_DIR`, else `<platform config dir>/mcp-meshgen`.
    pub fn default_dir() -> PathBuf {
        let cwd = std::env::current_dir().unwrap_or_else(|_| home_dir());
        config_dir_from(non_empty_env(CONFIG_DIR_ENV), dirs::config_dir(), &cwd)
    }

    /// Load `config.toml` from `dir`, writing the example config and model registry first if absent.
    pub fn load_or_init(dir: &Path) -> Result<Self, MeshGenError> {
        std::fs::create_dir_all(dir).map_err(|e| MeshGenError::config(dir, e))?;
        let config_path = dir.join("config.toml");
        if !config_path.exists() {
            std::fs::write(&config_path, CONFIG_EXAMPLE)
                .map_err(|e| MeshGenError::config(&config_path, e))?;
        }
        let models_path = dir.join("models.toml");
        if !models_path.exists() {
            std::fs::write(&models_path, MODELS_EXAMPLE)
                .map_err(|e| MeshGenError::config(&models_path, e))?;
        }
        let text = std::fs::read_to_string(&config_path)
            .map_err(|e| MeshGenError::config(&config_path, e))?;
        Self::from_toml(dir, &text).map_err(|msg| MeshGenError::config(&config_path, msg))
    }

    /// Parse config text; `dir` is recorded as the config directory and anchors relative paths.
    pub fn from_toml(dir: &Path, text: &str) -> Result<Self, String> {
        let raw: RawConfig = toml::from_str(text).map_err(|e| e.to_string())?;
        let base = dir.parent().unwrap_or(dir);
        let resolve = |p: &str| resolve_path(base, p);
        let non_empty = |v: Option<String>| v.filter(|s| !s.trim().is_empty());

        let allowed_roots: Vec<PathBuf> = match raw.allowed_roots {
            Some(roots) if roots.iter().any(|r| !r.trim().is_empty()) => roots
                .iter()
                .filter(|r| !r.trim().is_empty())
                .map(|r| resolve(r))
                .collect(),
            _ => vec![expand_tilde(DEFAULT_ROOT)],
        };
        let default_output_dir = non_empty(raw.default_output_dir).map(|d| resolve(&d));
        if let Some(out) = &default_output_dir
            && !allowed_roots.iter().any(|root| out.starts_with(root))
        {
            return Err(format!(
                "default_output_dir {} is not inside any allowed_roots entry",
                out.display()
            ));
        }

        let default_model =
            non_empty(raw.default_model).unwrap_or_else(|| "trellis2-4b".to_string());
        let max_queue = raw.max_queue.unwrap_or(4);
        if max_queue == 0 {
            return Err("max_queue must be at least 1".into());
        }

        let input = raw.input;
        for (name, value) in [
            ("input.min_transparent", input.min_transparent),
            ("input.max_transparent", input.max_transparent),
            ("input.min_opaque", input.min_opaque),
        ] {
            if !(0.0..=1.0).contains(&value) {
                return Err(format!("{name} must be between 0 and 1 (got {value})"));
            }
        }
        if input.min_transparent >= input.max_transparent {
            return Err(format!(
                "input.min_transparent ({}) must be below input.max_transparent ({})",
                input.min_transparent, input.max_transparent
            ));
        }
        if input.opaque_alpha == 0 {
            return Err("input.opaque_alpha must be at least 1".into());
        }

        let trellis = raw.backends.trellis_mac;
        let trellis_mac = TrellisMacConfig {
            dir: non_empty(trellis.dir).map(|d| resolve(&d)),
            python: non_empty(trellis.python).map(|p| resolve(&p)),
        };

        Ok(Self {
            config_dir: dir.to_path_buf(),
            cache_root: non_empty_env(CACHE_DIR_ENV)
                .map(|d| expand_tilde(&d))
                .or_else(dirs::cache_dir)
                .unwrap_or_else(std::env::temp_dir),
            allowed_roots,
            default_output_dir,
            default_model,
            hf_cache: non_empty(raw.hf_cache)
                .map(|d| resolve(&d))
                .unwrap_or_else(default_hf_cache),
            blender: non_empty(raw.blender).map(|b| resolve_program(base, &b)),
            memory_headroom_mb: raw.memory_headroom_mb.unwrap_or(2000),
            keep_loaded: raw.keep_loaded.unwrap_or(false),
            idle_unload_s: raw.idle_unload_s.unwrap_or(600),
            max_queue,
            job_ttl_s: raw.job_ttl_s.unwrap_or(3600),
            heavy_processes: raw
                .heavy_processes
                .unwrap_or_else(|| DEFAULT_HEAVY.iter().map(|s| s.to_string()).collect()),
            input,
            trellis_mac,
        })
    }

    pub fn models_path(&self) -> PathBuf {
        self.config_dir.join("models.toml")
    }

    /// This server's own cache (the embedded worker and Blender scripts are written here).
    pub fn app_cache_dir(&self) -> PathBuf {
        self.cache_root.join(APP_DIR)
    }

    /// The cross-process lock for local GPU jobs, shared with sister servers (§6; A2).
    pub fn gpu_lock_path(&self) -> PathBuf {
        self.cache_root.join("zavora-mcp").join("local-gpu.lock")
    }
}

/// `env` if set (made absolute against `cwd`), else `<platform>/mcp-meshgen`, else `~/.config/mcp-meshgen`.
fn config_dir_from(env: Option<String>, platform: Option<PathBuf>, cwd: &Path) -> PathBuf {
    if let Some(dir) = env {
        return resolve_path(cwd, &dir);
    }
    platform
        .unwrap_or_else(|| home_dir().join(".config"))
        .join(APP_DIR)
}

/// `$HF_HUB_CACHE`, then `$HF_HOME/hub`, then `~/.cache/huggingface/hub` (what `huggingface_hub` itself uses).
pub fn default_hf_cache() -> PathBuf {
    hf_cache_from(
        non_empty_env("HF_HUB_CACHE"),
        non_empty_env("HF_HOME"),
        &home_dir(),
    )
}

fn hf_cache_from(hub_cache: Option<String>, hf_home: Option<String>, home: &Path) -> PathBuf {
    if let Some(dir) = hub_cache {
        return expand_tilde(&dir);
    }
    if let Some(dir) = hf_home {
        return expand_tilde(&dir).join("hub");
    }
    home.join(".cache").join("huggingface").join("hub")
}

fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

/// The user's home directory (the OS's notion of it), else the temp dir.
pub fn home_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(std::env::temp_dir)
}

/// Expand a leading `~`, `~/` or `~\` to the home directory.
pub fn expand_tilde(path: &str) -> PathBuf {
    if path == "~" {
        home_dir()
    } else if let Some(rest) = path.strip_prefix("~/").or_else(|| path.strip_prefix("~\\")) {
        home_dir().join(rest)
    } else {
        PathBuf::from(path)
    }
}

/// Expand `~`, then anchor a relative path at `base`.
pub fn resolve_path(base: &Path, path: &str) -> PathBuf {
    let expanded = expand_tilde(path.trim());
    if expanded.is_absolute() {
        expanded
    } else {
        base.join(expanded)
    }
}

/// Like `resolve_path`, but a bare program name (no separator) stays a name to look up on PATH.
fn resolve_program(base: &Path, program: &str) -> PathBuf {
    let program = program.trim();
    if program.contains('/') || program.contains('\\') || program.starts_with('~') {
        resolve_path(base, program)
    } else {
        PathBuf::from(program)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(text: &str) -> Config {
        Config::from_toml(Path::new("/w/.meshgen"), text).unwrap()
    }

    #[test]
    fn example_config_parses() {
        let c = cfg(CONFIG_EXAMPLE);
        assert_eq!(
            c.allowed_roots,
            vec![home_dir().join("Documents/mcp-meshgen")]
        );
        assert_eq!(c.default_output_dir, None);
        assert_eq!(c.default_model, "trellis2-4b");
        assert_eq!(c.memory_headroom_mb, 2000);
        assert!(!c.keep_loaded);
        assert_eq!(c.idle_unload_s, 600);
        assert_eq!(c.max_queue, 4);
        assert_eq!(c.input, InputThresholds::default());
        assert_eq!(c.trellis_mac, TrellisMacConfig::default());
        assert_eq!(c.blender, None);
        assert_eq!(c.models_path(), PathBuf::from("/w/.meshgen/models.toml"));
    }

    #[test]
    fn empty_config_matches_example_defaults() {
        let (empty, example) = (cfg(""), cfg(CONFIG_EXAMPLE));
        assert_eq!(empty.allowed_roots, example.allowed_roots);
        assert_eq!(empty.heavy_processes, example.heavy_processes);
        assert_eq!(empty.job_ttl_s, example.job_ttl_s);
        assert_eq!(empty.hf_cache, default_hf_cache());
    }

    #[test]
    fn relative_paths_resolve_against_the_config_dirs_parent() {
        let c = cfg(r#"
allowed_roots = ["generated", "game", "lab/assets", "/abs/root", ""]
default_output_dir = "generated/meshes"
hf_cache = "cache/hub"
blender = "tools/blender/blender"

[backends.trellis_mac]
dir = "../tools/trellis-mac"
"#);
        assert_eq!(
            c.allowed_roots,
            vec![
                PathBuf::from("/w/generated"),
                PathBuf::from("/w/game"),
                PathBuf::from("/w/lab/assets"),
                PathBuf::from("/abs/root"),
            ]
        );
        assert_eq!(
            c.default_output_dir,
            Some(PathBuf::from("/w/generated/meshes"))
        );
        assert_eq!(c.hf_cache, PathBuf::from("/w/cache/hub"));
        assert_eq!(c.blender, Some(PathBuf::from("/w/tools/blender/blender")));
        let dir = PathBuf::from("/w/../tools/trellis-mac");
        assert_eq!(c.trellis_mac.dir, Some(dir.clone()));
        assert_eq!(c.trellis_mac.python_path(), Some(venv_python(&dir)));
    }

    #[test]
    fn bare_blender_name_stays_a_path_lookup() {
        assert_eq!(
            cfg("blender = \"blender\"").blender,
            Some(PathBuf::from("blender"))
        );
    }

    #[test]
    fn explicit_python_wins_over_the_venv() {
        let c = cfg("[backends.trellis_mac]\ndir = \"/t\"\npython = \"/py/bin/python3\"\n");
        assert_eq!(
            c.trellis_mac.python_path(),
            Some(PathBuf::from("/py/bin/python3"))
        );
    }

    #[test]
    fn venv_python_follows_the_platform_layout() {
        let p = venv_python(Path::new("/t"));
        if cfg!(windows) {
            assert!(p.ends_with(".venv/Scripts/python.exe"));
        } else {
            assert_eq!(p, PathBuf::from("/t/.venv/bin/python"));
        }
    }

    #[test]
    fn default_output_dir_must_be_inside_a_root() {
        let err = Config::from_toml(
            Path::new("/w/.meshgen"),
            "allowed_roots = [\"a\"]\ndefault_output_dir = \"b/out\"\n",
        )
        .unwrap_err();
        assert!(err.contains("not inside any allowed_roots"), "{err}");
    }

    #[test]
    fn thresholds_are_validated() {
        let bad = |t: &str| Config::from_toml(Path::new("/c"), &format!("[input]\n{t}\n"));
        assert!(
            bad("min_transparent = 1.5")
                .unwrap_err()
                .contains("between 0 and 1")
        );
        assert!(
            bad("min_transparent = 0.9\nmax_transparent = 0.5")
                .unwrap_err()
                .contains("below")
        );
        assert!(bad("opaque_alpha = 0").is_err());
        // Partial tables keep the other defaults.
        let c = Config::from_toml(Path::new("/c"), "[input]\nmin_opaque = 0.1\n").unwrap();
        assert_eq!(c.input.min_opaque, 0.1);
        assert_eq!(c.input.opaque_alpha, 205);
    }

    #[test]
    fn unknown_field_is_an_error_with_location() {
        let err = Config::from_toml(Path::new("/c"), "allowed_root = []\n").unwrap_err();
        assert!(err.contains("allowed_root"), "{err}");
        assert!(err.contains("line 1"), "{err}");
    }

    #[test]
    fn zero_queue_rejected() {
        assert!(Config::from_toml(Path::new("/c"), "max_queue = 0").is_err());
    }

    #[test]
    fn config_dir_discovery() {
        let cwd = Path::new("/proj");
        assert_eq!(
            config_dir_from(Some(".meshgen".into()), Some("/plat".into()), cwd),
            PathBuf::from("/proj/.meshgen")
        );
        assert_eq!(
            config_dir_from(Some("/abs/cfg".into()), None, cwd),
            PathBuf::from("/abs/cfg")
        );
        assert_eq!(
            config_dir_from(None, Some("/plat".into()), cwd),
            PathBuf::from("/plat/mcp-meshgen")
        );
        assert_eq!(
            config_dir_from(None, None, cwd),
            home_dir().join(".config/mcp-meshgen")
        );
    }

    #[test]
    fn hf_cache_discovery() {
        let home = Path::new("/h");
        assert_eq!(
            hf_cache_from(Some("/hub".into()), Some("/hfhome".into()), home),
            PathBuf::from("/hub")
        );
        assert_eq!(
            hf_cache_from(None, Some("/hfhome".into()), home),
            PathBuf::from("/hfhome/hub")
        );
        assert_eq!(
            hf_cache_from(None, None, home),
            PathBuf::from("/h/.cache/huggingface/hub")
        );
    }

    #[test]
    fn cache_paths() {
        let mut c = cfg("");
        c.cache_root = PathBuf::from("/cache");
        assert_eq!(c.app_cache_dir(), PathBuf::from("/cache/mcp-meshgen"));
        assert_eq!(
            c.gpu_lock_path(),
            PathBuf::from("/cache/zavora-mcp/local-gpu.lock")
        );
    }

    #[test]
    fn tilde_expansion() {
        let home = home_dir();
        assert_eq!(expand_tilde("~"), home);
        assert_eq!(expand_tilde("~/x/y"), home.join("x/y"));
        assert_eq!(expand_tilde("/abs"), PathBuf::from("/abs"));
        assert_eq!(expand_tilde("~other"), PathBuf::from("~other"));
    }

    #[test]
    fn load_or_init_writes_examples_once() {
        let dir = tempfile::tempdir().unwrap();
        let c = Config::load_or_init(dir.path()).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("config.toml")).unwrap(),
            CONFIG_EXAMPLE
        );
        assert_eq!(
            std::fs::read_to_string(c.models_path()).unwrap(),
            MODELS_EXAMPLE
        );
        std::fs::write(dir.path().join("config.toml"), "max_queue = 2\n").unwrap();
        assert_eq!(Config::load_or_init(dir.path()).unwrap().max_queue, 2);
    }

    #[test]
    fn load_or_init_reports_the_file_on_parse_errors() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.toml"), "max_queue = \"x\"\n").unwrap();
        let err = Config::load_or_init(dir.path()).unwrap_err();
        assert_eq!(err.code(), "config_error");
        assert!(err.to_string().contains("config.toml"), "{err}");
    }
}
