//! Model registry (design §3): `models.toml` parsing and per-model status (R3, R4).
//!
//! Status never loads anything: it checks the backend's availability (passed in) and whether every listed file is in the
//! Hugging Face cache, in the snapshot `refs/main` points at (what an offline `huggingface_hub` load resolves).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::MeshGenError;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDefaults {
    pub quality: String,
    pub texture_size: u32,
}

/// One component's licence (R3: every component, and whether commercial use is allowed).
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Licence {
    pub component: String,
    pub licence: String,
    pub commercial: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Files the model needs from one Hugging Face repo.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RepoFiles {
    pub repo: String,
    pub paths: Vec<String>,
    /// Download size of all `paths`, for the hint.
    pub mb: u64,
    /// The repo needs its licence accepted on the Hub (signed in) before it can be downloaded.
    #[serde(default)]
    pub gated: bool,
    /// Pinned revision (commit, tag or branch); default `main`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub backend: String,
    /// The pipeline to load, as the backend names it (trellis_mac: the Hugging Face repo with `pipeline.json`).
    /// Default: the first `files` repo.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<String>,
    /// `local` or `cloud` (R3); v1 backends are all local.
    #[serde(default = "local")]
    pub runs: String,
    pub quality: Vec<String>,
    pub defaults: ModelDefaults,
    pub texture_sizes: Vec<u32>,
    pub memory_mb: u64,
    pub typical_s: u64,
    /// Most faces the backend can bake onto (and the default budget).
    #[serde(default = "default_max_triangles")]
    pub max_triangles: u32,
    #[serde(default)]
    pub fixes: Vec<String>,
    pub licence: Vec<Licence>,
    #[serde(default)]
    pub files: Vec<RepoFiles>,
}

fn local() -> String {
    "local".into()
}

fn default_max_triangles() -> u32 {
    200_000
}

/// Missing files from one repo, with the command that fetches them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MissingRepo {
    pub repo: String,
    pub files: Vec<String>,
    /// Upper bound: the repo's listed size (exact when every file is missing).
    pub download_mb: u64,
    pub gated: bool,
    /// Where to accept the licence, for gated repos.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accept_at: Option<String>,
    pub command: String,
}

/// What `list_models` reports per model (R3).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ModelStatus {
    Ready,
    MissingFiles {
        missing: Vec<MissingRepo>,
        download_mb: u64,
    },
    /// Only gated files are missing: accept the licence, sign in, then download.
    NeedsLicenceAcceptance {
        missing: Vec<MissingRepo>,
        download_mb: u64,
    },
    BackendUnavailable {
        reason: String,
    },
}

impl ModelStatus {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::MissingFiles { .. } => "missing_files",
            Self::NeedsLicenceAcceptance { .. } => "needs_licence_acceptance",
            Self::BackendUnavailable { .. } => "backend_unavailable",
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    /// What to do next, in one message (also the `NotReady` error detail).
    pub fn detail(&self) -> String {
        match self {
            Self::Ready => "ready".into(),
            Self::BackendUnavailable { reason } => reason.clone(),
            Self::MissingFiles {
                missing,
                download_mb,
            }
            | Self::NeedsLicenceAcceptance {
                missing,
                download_mb,
            } => {
                let steps: Vec<String> = missing
                    .iter()
                    .map(|m| match &m.accept_at {
                        Some(url) => format!(
                            "accept the licence at {url}, sign in with `hf auth login`, then run: {}",
                            m.command
                        ),
                        None => format!("run: {}", m.command),
                    })
                    .collect();
                format!(
                    "{} file(s) missing, up to {download_mb} MB to download (needs your OK). {}",
                    missing.iter().map(|m| m.files.len()).sum::<usize>(),
                    steps.join("; ")
                )
            }
        }
    }

    pub fn not_ready_error(&self, model: &str) -> MeshGenError {
        MeshGenError::NotReady {
            model: model.to_string(),
            status: self.code().to_string(),
            detail: self.detail(),
        }
    }
}

impl ModelSpec {
    /// `pipeline`, else the first `files` repo.
    pub fn pipeline_id(&self) -> Option<&str> {
        self.pipeline
            .as_deref()
            .or_else(|| self.files.first().map(|f| f.repo.as_str()))
    }

    fn validate(&self) -> Result<(), String> {
        let id = &self.id;
        if id.trim().is_empty() {
            return Err("a model has an empty `id`".into());
        }
        if self.backend.trim().is_empty() {
            return Err(format!("model `{id}`: `backend` is empty"));
        }
        if !matches!(self.runs.as_str(), "local" | "cloud") {
            return Err(format!(
                "model `{id}`: `runs` must be \"local\" or \"cloud\""
            ));
        }
        if self.quality.is_empty() {
            return Err(format!("model `{id}`: `quality` needs at least one option"));
        }
        if !self.quality.contains(&self.defaults.quality) {
            return Err(format!(
                "model `{id}`: defaults.quality `{}` is not one of {:?}",
                self.defaults.quality, self.quality
            ));
        }
        if !self.texture_sizes.contains(&self.defaults.texture_size) {
            return Err(format!(
                "model `{id}`: defaults.texture_size {} is not one of {:?}",
                self.defaults.texture_size, self.texture_sizes
            ));
        }
        if self.max_triangles < crate::request::MIN_TRIANGLES {
            return Err(format!(
                "model `{id}`: `max_triangles` must be at least {}",
                crate::request::MIN_TRIANGLES
            ));
        }
        if self.memory_mb == 0 {
            return Err(format!("model `{id}`: `memory_mb` must be above 0"));
        }
        if self.licence.is_empty() {
            return Err(format!(
                "model `{id}`: list the licence of every component in `licence`"
            ));
        }
        for files in &self.files {
            let parts: Vec<&str> = files.repo.split('/').collect();
            if parts.len() != 2 || parts.iter().any(|p| p.is_empty()) {
                return Err(format!(
                    "model `{id}`: repo `{}` must look like `owner/name`",
                    files.repo
                ));
            }
            if files.paths.is_empty() {
                return Err(format!(
                    "model `{id}`: repo `{}` lists no paths",
                    files.repo
                ));
            }
            for path in &files.paths {
                let p = Path::new(path);
                if path.is_empty()
                    || p.is_absolute()
                    || path.starts_with('/')
                    || path.starts_with('\\')
                    || p.components()
                        .any(|c| matches!(c, std::path::Component::ParentDir))
                {
                    return Err(format!(
                        "model `{id}`: `{path}` in `{}` must be a relative path inside the repo",
                        files.repo
                    ));
                }
            }
        }
        Ok(())
    }

    /// Registry status for this model. `backend_unavailable` is the backend's reason when it can't run here.
    pub fn status(
        &self,
        hf_cache: &Path,
        default_cache: &Path,
        backend_unavailable: Option<&str>,
    ) -> ModelStatus {
        if let Some(reason) = backend_unavailable {
            return ModelStatus::BackendUnavailable {
                reason: reason.to_string(),
            };
        }
        let missing = self.missing_files(hf_cache, default_cache);
        if missing.is_empty() {
            return ModelStatus::Ready;
        }
        let download_mb = missing.iter().map(|m| m.download_mb).sum();
        if missing.iter().all(|m| m.gated) {
            ModelStatus::NeedsLicenceAcceptance {
                missing,
                download_mb,
            }
        } else {
            ModelStatus::MissingFiles {
                missing,
                download_mb,
            }
        }
    }

    /// Files not found in the cache, grouped by repo.
    pub fn missing_files(&self, hf_cache: &Path, default_cache: &Path) -> Vec<MissingRepo> {
        self.files
            .iter()
            .filter_map(|files| {
                let snapshot = snapshot_dir(hf_cache, &files.repo, files.revision.as_deref());
                let absent: Vec<String> = files
                    .paths
                    .iter()
                    .filter(|p| !snapshot.as_ref().is_some_and(|s| s.join(p).is_file()))
                    .cloned()
                    .collect();
                (!absent.is_empty()).then(|| MissingRepo {
                    command: download_command(files, &absent, hf_cache, default_cache),
                    accept_at: files
                        .gated
                        .then(|| format!("https://huggingface.co/{}", files.repo)),
                    repo: files.repo.clone(),
                    files: absent,
                    download_mb: files.mb,
                    gated: files.gated,
                })
            })
            .collect()
    }
}

/// The snapshot an offline load would use: `refs/<revision or main>`, a snapshot named by the revision itself (a commit),
/// else the newest snapshot.
pub fn snapshot_dir(hf_cache: &Path, repo: &str, revision: Option<&str>) -> Option<PathBuf> {
    let repo_dir = hf_cache.join(format!("models--{}", repo.replace('/', "--")));
    let snapshots = repo_dir.join("snapshots");
    let revision = revision.unwrap_or("main");
    if let Ok(commit) = std::fs::read_to_string(repo_dir.join("refs").join(revision)) {
        let dir = snapshots.join(commit.trim());
        if dir.is_dir() {
            return Some(dir);
        }
    }
    let named = snapshots.join(revision);
    if named.is_dir() {
        return Some(named);
    }
    let mut dirs: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let modified = entry.metadata().and_then(|m| m.modified()).ok()?;
            entry.path().is_dir().then(|| (modified, entry.path()))
        })
        .collect();
    dirs.sort_by_key(|d| std::cmp::Reverse(d.0));
    dirs.into_iter().next().map(|(_, p)| p)
}

/// The exact `hf download` command for the missing files (`--cache-dir` only when the cache isn't the default one).
fn download_command(
    files: &RepoFiles,
    absent: &[String],
    hf_cache: &Path,
    default_cache: &Path,
) -> String {
    let mut parts = vec!["hf".to_string(), "download".into(), quote(&files.repo)];
    parts.extend(absent.iter().map(|p| quote(p)));
    if let Some(rev) = &files.revision {
        parts.push("--revision".into());
        parts.push(quote(rev));
    }
    if hf_cache != default_cache {
        parts.push("--cache-dir".into());
        parts.push(quote(&hf_cache.display().to_string()));
    }
    parts.join(" ")
}

/// Double-quote an argument unless it's plainly safe (double quotes work in sh, PowerShell and cmd).
fn quote(arg: &str) -> String {
    if !arg.is_empty()
        && arg
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-/:=+@".contains(c))
    {
        arg.to_string()
    } else {
        format!("\"{arg}\"")
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    models: Vec<ModelSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryFile {
    #[serde(default)]
    model: Vec<ModelSpec>,
}

impl ModelRegistry {
    pub fn load(path: &Path) -> Result<Self, MeshGenError> {
        let text = std::fs::read_to_string(path).map_err(|e| MeshGenError::config(path, e))?;
        Self::from_toml(&text).map_err(|msg| MeshGenError::config(path, msg))
    }

    pub fn from_toml(text: &str) -> Result<Self, String> {
        let file: RegistryFile = toml::from_str(text).map_err(|e| e.to_string())?;
        let mut seen = HashSet::new();
        for model in &file.model {
            model.validate()?;
            if !seen.insert(model.id.clone()) {
                return Err(format!("duplicate model id `{}`", model.id));
            }
        }
        Ok(Self { models: file.model })
    }

    pub fn models(&self) -> &[ModelSpec] {
        &self.models
    }

    pub fn get(&self, id: &str) -> Result<&ModelSpec, MeshGenError> {
        self.models
            .iter()
            .find(|m| m.id == id)
            .ok_or_else(|| MeshGenError::UnknownModel {
                id: id.to_string(),
                available: self.ids(),
            })
    }

    pub fn ids(&self) -> Vec<String> {
        self.models.iter().map(|m| m.id.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MODELS_EXAMPLE;

    const DEFAULT: &str = "/default/hub";

    fn registry() -> ModelRegistry {
        ModelRegistry::from_toml(MODELS_EXAMPLE).unwrap()
    }

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"x").unwrap();
    }

    /// Lay out a repo like huggingface_hub does: `refs/main` → `snapshots/<commit>/...`.
    fn cache_repo(cache: &Path, repo: &str, commit: &str, paths: &[String]) {
        let dir = cache.join(format!("models--{}", repo.replace('/', "--")));
        std::fs::create_dir_all(dir.join("refs")).unwrap();
        std::fs::write(dir.join("refs/main"), commit).unwrap();
        for p in paths {
            touch(&dir.join("snapshots").join(commit).join(p));
        }
    }

    fn cache_all(cache: &Path, spec: &ModelSpec) {
        for f in &spec.files {
            cache_repo(cache, &f.repo, "abc123", &f.paths);
        }
    }

    #[test]
    fn example_registry_parses() {
        let reg = registry();
        assert_eq!(reg.ids(), vec!["trellis2-4b"]);
        let m = reg.get("trellis2-4b").unwrap();
        assert_eq!(m.backend, "trellis_mac");
        assert_eq!(m.pipeline_id(), Some("microsoft/TRELLIS.2-4B"));
        assert_eq!(m.runs, "local");
        assert_eq!(m.quality, vec!["512", "1024", "1024_cascade"]);
        assert_eq!(m.defaults.quality, "512");
        assert_eq!(m.defaults.texture_size, 1024);
        assert_eq!(m.fixes, vec!["mps_sparse_decoder_cpu"]);
        assert!(m.licence.iter().all(|l| l.commercial));
        assert_eq!(m.files.len(), 3);
        assert!(m.files[2].gated);
        assert_eq!(
            m.files.iter().map(|f| f.mb).sum::<u64>(),
            14134 + 141 + 1157
        );
    }

    #[test]
    fn ready_when_every_file_is_in_the_main_snapshot() {
        let cache = tempfile::tempdir().unwrap();
        let reg = registry();
        let m = reg.get("trellis2-4b").unwrap();
        cache_all(cache.path(), m);
        assert_eq!(
            m.status(cache.path(), Path::new(DEFAULT), None),
            ModelStatus::Ready
        );
    }

    #[test]
    fn missing_files_give_sizes_and_exact_commands() {
        let cache = tempfile::tempdir().unwrap();
        let reg = registry();
        let m = reg.get("trellis2-4b").unwrap();
        // Everything but one decoder and the image-large repo.
        let mut trellis = m.files[0].paths.clone();
        trellis.retain(|p| !p.starts_with("ckpts/tex_dec"));
        cache_repo(cache.path(), &m.files[0].repo, "abc", &trellis);
        cache_repo(cache.path(), &m.files[2].repo, "def", &m.files[2].paths);

        let status = m.status(cache.path(), cache.path(), None);
        assert_eq!(status.code(), "missing_files");
        let ModelStatus::MissingFiles {
            missing,
            download_mb,
        } = &status
        else {
            panic!("{status:?}")
        };
        assert_eq!(missing.len(), 2);
        assert_eq!(
            missing[0].command,
            "hf download microsoft/TRELLIS.2-4B ckpts/tex_dec_next_dc_f16c32_fp16.json ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"
        );
        assert_eq!(
            missing[1].command,
            "hf download microsoft/TRELLIS-image-large ckpts/ss_dec_conv3d_16l8_fp16.json ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
        );
        assert_eq!(*download_mb, 14134 + 141);
        let detail = status.detail();
        assert!(
            detail.starts_with("4 file(s) missing, up to 14275 MB"),
            "{detail}"
        );
        assert!(detail.contains("needs your OK"), "{detail}");
    }

    #[test]
    fn only_gated_files_missing_needs_licence_acceptance() {
        let cache = tempfile::tempdir().unwrap();
        let reg = registry();
        let m = reg.get("trellis2-4b").unwrap();
        cache_repo(cache.path(), &m.files[0].repo, "a", &m.files[0].paths);
        cache_repo(cache.path(), &m.files[1].repo, "b", &m.files[1].paths);
        let status = m.status(cache.path(), Path::new(DEFAULT), None);
        assert_eq!(status.code(), "needs_licence_acceptance");
        let detail = status.detail();
        assert!(
            detail.contains(
                "accept the licence at https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"
            ),
            "{detail}"
        );
        // A non-default cache goes into the command.
        assert!(detail.contains("--cache-dir"), "{detail}");
    }

    #[test]
    fn backend_unavailable_wins() {
        let cache = tempfile::tempdir().unwrap();
        let reg = registry();
        let m = reg.get("trellis2-4b").unwrap();
        let status = m.status(
            cache.path(),
            cache.path(),
            Some("trellis-mac needs Apple Silicon"),
        );
        assert_eq!(status.code(), "backend_unavailable");
        assert_eq!(status.detail(), "trellis-mac needs Apple Silicon");
        let err = status.not_ready_error("trellis2-4b");
        assert_eq!(err.code(), "not_ready");
        assert!(err.to_string().contains("(backend_unavailable)"), "{err}");
    }

    #[test]
    fn files_in_a_stale_snapshot_dont_count() {
        let cache = tempfile::tempdir().unwrap();
        let repo = "o/r";
        let dir = cache.path().join("models--o--r");
        touch(&dir.join("snapshots/old/f.bin"));
        std::fs::create_dir_all(dir.join("snapshots/new")).unwrap();
        std::fs::create_dir_all(dir.join("refs")).unwrap();
        std::fs::write(dir.join("refs/main"), "new\n").unwrap();
        assert_eq!(
            snapshot_dir(cache.path(), repo, None),
            Some(dir.join("snapshots/new"))
        );
        // No refs: the newest snapshot; a revision naming a snapshot directly: that one.
        std::fs::remove_dir_all(dir.join("refs")).unwrap();
        assert_eq!(
            snapshot_dir(cache.path(), repo, Some("old")),
            Some(dir.join("snapshots/old"))
        );
        assert!(snapshot_dir(cache.path(), "x/none", None).is_none());
    }

    #[test]
    fn pinned_revision_goes_into_the_command() {
        let files = RepoFiles {
            repo: "o/r".into(),
            paths: vec!["a b.bin".into()],
            mb: 1,
            gated: false,
            revision: Some("v1.0".into()),
        };
        assert_eq!(
            download_command(&files, &files.paths, Path::new("/c"), Path::new("/c")),
            "hf download o/r \"a b.bin\" --revision v1.0"
        );
    }

    #[test]
    fn validation_names_the_problem() {
        let base = MODELS_EXAMPLE;
        let bad = |from: &str, to: &str| {
            assert!(base.contains(from), "{from}");
            ModelRegistry::from_toml(&base.replacen(from, to, 1)).unwrap_err()
        };
        assert!(bad("quality = \"512\"", "quality = \"2048\"").contains("defaults.quality"));
        assert!(bad("texture_size = 1024", "texture_size = 999").contains("defaults.texture_size"));
        assert!(bad("memory_mb = 15000", "memory_mb = 0").contains("memory_mb"));
        assert!(
            bad(
                "repo = \"microsoft/TRELLIS-image-large\"",
                "repo = \"nope\""
            )
            .contains("owner/name")
        );
        assert!(bad("\"pipeline.json\",", "\"../pipeline.json\",").contains("relative path"));
        assert!(bad("id = \"trellis2-4b\"", "id = \"\"").contains("empty `id`"));
        let dup = format!("{base}\n{}", &base[base.find("[[model]]").unwrap()..]);
        assert!(
            ModelRegistry::from_toml(&dup)
                .unwrap_err()
                .contains("duplicate")
        );
        let typo =
            ModelRegistry::from_toml("[[model]]\nid = \"x\"\nbackend = \"b\"\nqualty = []\n")
                .unwrap_err();
        assert!(typo.contains("qualty"), "{typo}");
    }

    #[test]
    fn unknown_model_lists_available() {
        let err = registry().get("nope").unwrap_err();
        assert_eq!(err.code(), "unknown_model");
        assert!(err.to_string().contains("trellis2-4b"), "{err}");
    }

    #[test]
    fn status_serializes_with_a_status_tag() {
        let json =
            serde_json::to_value(ModelStatus::BackendUnavailable { reason: "r".into() }).unwrap();
        assert_eq!(json["status"], "backend_unavailable");
        assert_eq!(
            serde_json::to_value(ModelStatus::Ready).unwrap()["status"],
            "ready"
        );
    }

    #[test]
    fn load_reports_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.toml");
        std::fs::write(&path, "[[model]]\nid = 1\n").unwrap();
        let err = ModelRegistry::load(&path).unwrap_err();
        assert!(err.to_string().contains("models.toml"), "{err}");
    }
}

/// Run by hand on a machine with the weights: `cargo test real_cache -- --ignored --nocapture`.
#[cfg(test)]
mod live {
    use super::*;
    use crate::config::{MODELS_EXAMPLE, default_hf_cache};

    #[test]
    #[ignore]
    fn real_cache_status() {
        let reg = ModelRegistry::from_toml(MODELS_EXAMPLE).unwrap();
        let cache = default_hf_cache();
        for m in reg.models() {
            let status = m.status(&cache, &cache, None);
            println!("{}: {} ({})", m.id, status.code(), status.detail());
        }
    }
}
