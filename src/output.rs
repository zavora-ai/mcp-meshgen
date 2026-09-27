//! Paths in and out (design §5, §7; R1, R5, R10), adapted from mcp-imagegen.
//!
//! Inputs and outputs are jailed under the configured roots. A job writes a family of files sharing one stem
//! (`x.glb`, `x.json`, `x.obj`, `x_game.glb`, `x_front.png`, …), so a stem is free only when nothing in the folder
//! starts with `x.` or `x_`. The stem is reserved by creating `x.glb.partial` exclusively; existing files are never
//! overwritten, and partials are renamed on success or deleted on cancel and failure.

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

use serde::Serialize;

use crate::config::{Config, expand_tilde};
use crate::error::MeshGenError;

/// Suffix of files still being written.
pub const PARTIAL: &str = "partial";

/// The four preview views (R8), in file-name form.
pub const PREVIEW_VIEWS: [&str; 4] = ["front", "side", "back", "three_quarter"];

#[derive(Debug, Clone, PartialEq)]
pub struct PathPolicy {
    /// Relative paths resolve here: the config dir's parent, like the roots.
    pub base: PathBuf,
    /// `None` = next to the input image.
    pub default_dir: Option<PathBuf>,
    pub allowed_roots: Vec<PathBuf>,
}

/// Where a job's files go, before collision handling.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshTarget {
    pub dir: PathBuf,
    pub stem: String,
}

/// A reserved stem and every path derived from it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MeshNames {
    pub dir: PathBuf,
    pub stem: String,
}

impl PathPolicy {
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            base: cfg
                .config_dir
                .parent()
                .unwrap_or(&cfg.config_dir)
                .to_path_buf(),
            default_dir: cfg.default_output_dir.clone(),
            allowed_roots: cfg.allowed_roots.clone(),
        }
    }

    /// Expand `~` and anchor a relative path at `base`.
    pub fn absolutize(&self, p: &str) -> PathBuf {
        let p = expand_tilde(p.trim());
        if p.is_absolute() {
            p
        } else {
            self.base.join(p)
        }
    }

    /// The input image: an existing file inside the roots (R10).
    pub fn input(&self, image: &str) -> Result<PathBuf, MeshGenError> {
        if image.trim().is_empty() {
            return Err(MeshGenError::invalid("image", "is required"));
        }
        let path = self.absolutize(image);
        self.jail(&path)?;
        if !path.is_file() {
            return Err(MeshGenError::invalid(
                "image",
                format!("{} is not a file", path.display()),
            ));
        }
        Ok(path)
    }

    /// Where the mesh should go: `output_path` (a `.glb`), else `output_dir`, else the default dir or the input's folder.
    /// The default stem is `<input stem>-<seed>`.
    pub fn plan(
        &self,
        output_path: Option<&str>,
        output_dir: Option<&str>,
        input: &Path,
        seed: u64,
    ) -> Result<MeshTarget, MeshGenError> {
        let default_stem = format!(
            "{}-{seed}",
            slug(&input.file_stem().unwrap_or_default().to_string_lossy())
        );
        let target = match (output_path, output_dir) {
            (Some(p), _) if !p.trim().is_empty() => {
                let p = self.absolutize(p);
                let is_glb = p
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("glb"));
                let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned());
                match (is_glb, stem, p.parent()) {
                    (true, Some(stem), Some(dir)) if !stem.is_empty() => MeshTarget {
                        dir: dir.to_path_buf(),
                        stem,
                    },
                    _ => return Err(MeshGenError::invalid("output_path", "must end in .glb")),
                }
            }
            (_, Some(d)) if !d.trim().is_empty() => MeshTarget {
                dir: self.absolutize(d),
                stem: default_stem,
            },
            _ => MeshTarget {
                dir: self
                    .default_dir
                    .clone()
                    .or_else(|| input.parent().map(Path::to_path_buf))
                    .unwrap_or_else(|| self.base.clone()),
                stem: default_stem,
            },
        };
        self.jail(&target.dir.join(format!("{}.glb", target.stem)))?;
        Ok(target)
    }

    fn jail(&self, target: &Path) -> Result<(), MeshGenError> {
        let not_allowed = || MeshGenError::PathNotAllowed {
            path: target.to_path_buf(),
            roots: self.allowed_roots.clone(),
        };
        // `..` resolves lexically on Windows but not on Unix (and can step around symlinks), so
        // reject it outright for identical, safe behaviour everywhere.
        if target.components().any(|c| c == Component::ParentDir) {
            return Err(not_allowed());
        }
        if within_roots(target, &self.allowed_roots) {
            Ok(())
        } else {
            Err(not_allowed())
        }
    }
}

/// Canonicalize the deepest existing ancestor (resolving symlinks) and append the rest.
/// `..` in the existing prefix is resolved by canonicalize; in the missing remainder it has no
/// file name to peel off, so the walk returns `None` and the path is rejected.
pub(crate) fn resolve_lenient(path: &Path) -> Option<PathBuf> {
    let mut existing = path.to_path_buf();
    let mut rest = Vec::new();
    while !existing.exists() {
        let name = existing.file_name()?.to_owned();
        rest.push(name);
        existing = existing.parent()?.to_path_buf();
    }
    let mut out = existing.canonicalize().ok()?;
    for part in rest.iter().rev() {
        match Path::new(part).components().next() {
            Some(Component::Normal(_)) => out.push(part),
            _ => return None,
        }
    }
    Some(out)
}

/// True if `path` resolves (symlinks followed) strictly inside one of `roots`.
pub(crate) fn within_roots(path: &Path, roots: &[PathBuf]) -> bool {
    let Some(resolved) = resolve_lenient(path) else {
        return false;
    };
    roots
        .iter()
        .filter_map(|root| resolve_lenient(root))
        .any(|root| resolved.starts_with(&root) && resolved != root)
}

/// True if nothing in `dir` belongs to the `stem` family (`stem.*` or `stem_*`).
fn stem_is_free(dir: &Path, stem: &str) -> bool {
    let (dot, under) = (format!("{stem}."), format!("{stem}_"));
    match std::fs::read_dir(dir) {
        Ok(entries) => !entries.filter_map(Result::ok).any(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.starts_with(&dot) || name.starts_with(&under)
        }),
        Err(_) => true,
    }
}

/// Reserve the first free stem (`x`, `x-1`, …) by creating `<stem>.glb.partial` exclusively, so two jobs never
/// share a name even when they target the same file.
pub fn reserve(target: &MeshTarget) -> Result<MeshNames, MeshGenError> {
    std::fs::create_dir_all(&target.dir).map_err(|e| MeshGenError::io(&target.dir, e))?;
    for n in 0.. {
        let stem = if n == 0 {
            target.stem.clone()
        } else {
            format!("{}-{n}", target.stem)
        };
        if !stem_is_free(&target.dir, &stem) {
            continue;
        }
        let names = MeshNames {
            dir: target.dir.clone(),
            stem,
        };
        let marker = names.partial(&names.glb());
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&marker)
        {
            Ok(_) => return Ok(names),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(MeshGenError::io(&marker, e)),
        }
    }
    unreachable!("unbounded search always finds a free stem")
}

impl MeshNames {
    fn file(&self, suffix: &str) -> PathBuf {
        self.dir.join(format!("{}{suffix}", self.stem))
    }

    pub fn glb(&self) -> PathBuf {
        self.file(".glb")
    }

    pub fn obj(&self) -> PathBuf {
        self.file(".obj")
    }

    pub fn sidecar(&self) -> PathBuf {
        self.file(".json")
    }

    pub fn game_glb(&self) -> PathBuf {
        self.file("_game.glb")
    }

    pub fn game_fbx(&self) -> PathBuf {
        self.file("_game.fbx")
    }

    pub fn preview(&self, view: &str) -> PathBuf {
        self.file(&format!("_{view}.png"))
    }

    /// `<path>.partial`: where a file is written before it's final.
    pub fn partial(&self, path: &Path) -> PathBuf {
        let mut name: OsString = path.file_name().unwrap_or_default().to_owned();
        name.push(format!(".{PARTIAL}"));
        path.with_file_name(name)
    }

    /// Rename `<path>.partial` to `<path>`, never replacing an existing file.
    pub fn finalize(&self, path: &Path) -> Result<PathBuf, MeshGenError> {
        let partial = self.partial(path);
        if path.exists() {
            return Err(MeshGenError::io(
                path,
                "already exists; refusing to overwrite",
            ));
        }
        std::fs::rename(&partial, path).map_err(|e| MeshGenError::io(&partial, e))?;
        Ok(path.to_path_buf())
    }

    /// Delete every `<stem>*.partial` in the folder (cancel, failure, or a finished job's leftovers).
    pub fn remove_partials(&self) -> usize {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return 0;
        };
        let (dot, under, tail) = (
            format!("{}.", self.stem),
            format!("{}_", self.stem),
            format!(".{PARTIAL}"),
        );
        entries
            .filter_map(Result::ok)
            .filter(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                (name.starts_with(&dot) || name.starts_with(&under)) && name.ends_with(&tail)
            })
            .filter(|e| std::fs::remove_file(e.path()).is_ok())
            .count()
    }
}

/// Write the sidecar JSON atomically (`.tmp` + rename). Never overwrites.
pub fn write_sidecar(path: &Path, metadata: &impl Serialize) -> Result<(), MeshGenError> {
    if path.exists() {
        return Err(MeshGenError::io(
            path,
            "already exists; refusing to overwrite",
        ));
    }
    let json = serde_json::to_vec_pretty(metadata).map_err(|e| MeshGenError::io(path, e))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json).map_err(|e| MeshGenError::io(&tmp, e))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        MeshGenError::io(path, e)
    })
}

/// Longest side of an inline preview.
pub const PREVIEW_MAX_SIDE: u32 = 512;
/// Upper bound on preview size so results stay well under MCP message limits.
pub const PREVIEW_MAX_BYTES: usize = 1_000_000;

/// Downscaled PNG for inline display: longest side ≤ 512 px and under 1 MB (from mcp-imagegen).
pub fn make_preview(png: &[u8]) -> Result<Vec<u8>, String> {
    let img = image::load_from_memory(png).map_err(|e| e.to_string())?;
    let mut side = PREVIEW_MAX_SIDE;
    loop {
        let thumb = if img.width().max(img.height()) > side {
            img.thumbnail(side, side)
        } else {
            img.clone()
        };
        let mut out = Vec::new();
        thumb
            .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .map_err(|e| e.to_string())?;
        if out.len() <= PREVIEW_MAX_BYTES || side <= 64 {
            return Ok(out);
        }
        side /= 2;
    }
}

/// File-name-safe slug: lowercase ASCII words joined by `-`, at most 48 chars.
pub fn slug(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
        if out.len() >= 48 {
            break;
        }
    }
    let out = out.trim_matches('-').to_string();
    if out.is_empty() { "mesh".into() } else { out }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(root: &Path) -> PathPolicy {
        PathPolicy {
            base: root.to_path_buf(),
            default_dir: None,
            allowed_roots: vec![root.join("game"), root.join("generated")],
        }
    }

    fn setup() -> (tempfile::TempDir, PathPolicy, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let p = policy(dir.path());
        let input = dir.path().join("game/refs/Brazier Ref.png");
        std::fs::create_dir_all(input.parent().unwrap()).unwrap();
        std::fs::write(&input, b"png").unwrap();
        std::fs::create_dir_all(dir.path().join("generated")).unwrap();
        (dir, p, input)
    }

    #[test]
    fn input_must_be_an_existing_file_inside_a_root() {
        let (dir, p, input) = setup();
        assert_eq!(p.input("game/refs/Brazier Ref.png").unwrap(), input);
        let outside = tempfile::tempdir().unwrap();
        let foreign = outside.path().join("x.png");
        std::fs::write(&foreign, b"png").unwrap();
        assert_eq!(
            p.input(foreign.to_str().unwrap()).unwrap_err().code(),
            "path_not_allowed"
        );
        assert_eq!(
            p.input("game/../secret.png").unwrap_err().code(),
            "path_not_allowed"
        );
        assert_eq!(
            p.input("game/refs/missing.png").unwrap_err().code(),
            "invalid_request"
        );
        assert_eq!(p.input(" ").unwrap_err().code(), "invalid_request");
        std::fs::create_dir_all(dir.path().join("game/folder.png")).unwrap();
        assert_eq!(
            p.input("game/folder.png").unwrap_err().code(),
            "invalid_request"
        );
    }

    #[test]
    fn default_target_is_next_to_the_input() {
        let (_dir, p, input) = setup();
        let t = p.plan(None, None, &input, 42).unwrap();
        assert_eq!(t.dir, input.parent().unwrap());
        assert_eq!(t.stem, "brazier-ref-42");
    }

    #[test]
    fn default_dir_output_dir_and_output_path() {
        let (dir, mut p, input) = setup();
        p.default_dir = Some(dir.path().join("generated/meshes"));
        assert_eq!(
            p.plan(None, None, &input, 7).unwrap().dir,
            dir.path().join("generated/meshes")
        );
        let t = p.plan(None, Some("game/assets/source"), &input, 7).unwrap();
        assert_eq!(t.dir, dir.path().join("game/assets/source"));
        assert_eq!(t.stem, "brazier-ref-7");
        let t = p
            .plan(Some("generated/props/Brazier.GLB"), None, &input, 7)
            .unwrap();
        assert_eq!(
            (t.dir, t.stem),
            (dir.path().join("generated/props"), "Brazier".into())
        );
    }

    #[test]
    fn rejects_escapes_and_bad_extensions() {
        let (_dir, p, input) = setup();
        let outside = tempfile::tempdir().unwrap();
        let foreign = outside.path().join("x.glb");
        for bad in [
            foreign.to_str().unwrap(),
            "../x.glb",
            "generated/../../x.glb",
            "elsewhere/x.glb",
        ] {
            let err = p.plan(Some(bad), None, &input, 1).unwrap_err();
            assert_eq!(err.code(), "path_not_allowed", "{bad}: {err}");
        }
        assert_eq!(
            p.plan(None, Some("../out"), &input, 1).unwrap_err().code(),
            "path_not_allowed"
        );
        for bad in ["generated/x.obj", "generated/.glb", "generated/x"] {
            assert_eq!(
                p.plan(Some(bad), None, &input, 1).unwrap_err().code(),
                "invalid_request",
                "{bad}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape() {
        let (dir, p, input) = setup();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), dir.path().join("generated/link")).unwrap();
        let err = p.plan(None, Some("generated/link"), &input, 1).unwrap_err();
        assert_eq!(err.code(), "path_not_allowed");
    }

    #[test]
    fn reserve_skips_every_member_of_a_taken_family() {
        let (dir, _p, _input) = setup();
        let out = dir.path().join("generated");
        let target = MeshTarget {
            dir: out.clone(),
            stem: "crate".into(),
        };
        let a = reserve(&target).unwrap();
        assert_eq!(a.stem, "crate");
        assert!(out.join("crate.glb.partial").exists());
        // The partial alone reserves the name for a second job.
        assert_eq!(reserve(&target).unwrap().stem, "crate-1");
        // Any family member blocks a stem: a stray sidecar or a preview.
        std::fs::write(out.join("crate-2.json"), b"{}").unwrap();
        std::fs::write(out.join("crate-3_front.png"), b"x").unwrap();
        assert_eq!(reserve(&target).unwrap().stem, "crate-4");
        // A different stem that merely shares a prefix doesn't.
        let other = MeshTarget {
            dir: out.clone(),
            stem: "cra".into(),
        };
        assert_eq!(reserve(&other).unwrap().stem, "cra");
    }

    #[test]
    fn reserve_creates_the_folder() {
        let (dir, _p, _input) = setup();
        let target = MeshTarget {
            dir: dir.path().join("generated/new/deep"),
            stem: "m".into(),
        };
        assert_eq!(
            reserve(&target).unwrap().glb(),
            dir.path().join("generated/new/deep/m.glb")
        );
    }

    #[test]
    fn names_finalize_and_cleanup() {
        let (dir, _p, _input) = setup();
        let out = dir.path().join("generated");
        let names = reserve(&MeshTarget {
            dir: out.clone(),
            stem: "b".into(),
        })
        .unwrap();
        assert_eq!(names.sidecar(), out.join("b.json"));
        assert_eq!(names.game_fbx(), out.join("b_game.fbx"));
        assert_eq!(
            names.preview("three_quarter"),
            out.join("b_three_quarter.png")
        );
        assert_eq!(names.partial(&names.obj()), out.join("b.obj.partial"));

        std::fs::write(names.partial(&names.glb()), b"glTF").unwrap();
        assert_eq!(names.finalize(&names.glb()).unwrap(), out.join("b.glb"));
        assert_eq!(std::fs::read(out.join("b.glb")).unwrap(), b"glTF");
        // Never overwrites.
        std::fs::write(names.partial(&names.glb()), b"second").unwrap();
        assert!(names.finalize(&names.glb()).is_err());
        assert_eq!(std::fs::read(out.join("b.glb")).unwrap(), b"glTF");

        std::fs::write(names.partial(&names.obj()), b"o").unwrap();
        std::fs::write(names.partial(&names.game_glb()), b"g").unwrap();
        std::fs::write(out.join("other.glb.partial"), b"keep").unwrap();
        assert_eq!(names.remove_partials(), 3);
        assert!(out.join("other.glb.partial").exists());
        assert!(out.join("b.glb").exists());
    }

    #[test]
    fn sidecar_is_written_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.json");
        write_sidecar(&path, &serde_json::json!({"seed": 3})).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(v["seed"], 3);
        assert!(!dir.path().join("m.json.tmp").exists());
        assert!(write_sidecar(&path, &serde_json::json!({})).is_err());
    }

    #[test]
    fn preview_is_downscaled() {
        let img = image::RgbaImage::from_pixel(1024, 768, image::Rgba([1, 2, 3, 255]));
        let mut png = Vec::new();
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let preview = image::load_from_memory(&make_preview(&png).unwrap()).unwrap();
        assert_eq!((preview.width(), preview.height()), (512, 384));
    }

    #[test]
    fn slugs() {
        assert_eq!(slug("  Brazier, Gold  "), "brazier-gold");
        assert_eq!(slug("日本語"), "mesh");
        assert!(slug(&"word ".repeat(40)).len() <= 48);
    }
}
