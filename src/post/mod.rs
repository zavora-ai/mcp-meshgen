//! Post-processing after the worker has exited (design §4.4; R8, R9): previews and the engine copy.
//!
//! The job manager only knows this trait; `blender.rs` implements it with headless Blender, `MockPost` in tests.

use std::path::{Path, PathBuf};

use serde::Serialize;
use tokio_util::sync::CancellationToken;

use crate::glb::MeshStats;
use crate::output::MeshNames;
use crate::request::ResolvedRequest;

pub mod blender;

/// The engine-ready copy (R9).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GameCopy {
    pub glb: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fbx: Option<PathBuf>,
    /// Uniform scale applied to reach `size_m` (1.0 without it).
    pub scale: f64,
    /// Stats of the game GLB (glTF axes: Y up).
    pub stats: MeshStats,
}

#[async_trait::async_trait]
pub trait PostProcessor: Send + Sync {
    /// `Err(reason)` when it can't run here (e.g. Blender not found).
    fn availability(&self) -> Result<(), String>;

    /// Render the four previews for `glb`; files are written as `<preview>.partial` and returned under their final names.
    async fn previews(
        &self,
        glb: &Path,
        names: &MeshNames,
        cancel: &CancellationToken,
    ) -> Result<Vec<PathBuf>, String>;

    /// Scale, re-origin and export the engine copy; files are written as partials and returned under their final names.
    async fn game_copy(
        &self,
        glb: &Path,
        names: &MeshNames,
        request: &ResolvedRequest,
        cancel: &CancellationToken,
    ) -> Result<GameCopy, String>;
}

/// Test stand-in: copies the GLB as the game copy and writes tiny PNG previews.
#[cfg(any(test, feature = "mock"))]
pub struct MockPost {
    pub available: bool,
    pub fail: bool,
}

#[cfg(any(test, feature = "mock"))]
#[async_trait::async_trait]
impl PostProcessor for MockPost {
    fn availability(&self) -> Result<(), String> {
        if self.available {
            Ok(())
        } else {
            Err("Blender not found (mock)".into())
        }
    }

    async fn previews(
        &self,
        _glb: &Path,
        names: &MeshNames,
        _cancel: &CancellationToken,
    ) -> Result<Vec<PathBuf>, String> {
        if self.fail {
            return Err("mock preview failure".into());
        }
        let png = {
            let img = image::RgbaImage::from_pixel(64, 48, image::Rgba([120, 90, 40, 255]));
            let mut out = Vec::new();
            image::DynamicImage::ImageRgba8(img)
                .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
                .map_err(|e| e.to_string())?;
            out
        };
        let mut paths = Vec::new();
        for view in crate::output::PREVIEW_VIEWS {
            let path = names.preview(view);
            std::fs::write(names.partial(&path), &png).map_err(|e| e.to_string())?;
            paths.push(path);
        }
        Ok(paths)
    }

    async fn game_copy(
        &self,
        glb: &Path,
        names: &MeshNames,
        request: &ResolvedRequest,
        _cancel: &CancellationToken,
    ) -> Result<GameCopy, String> {
        if self.fail {
            return Err("mock post-process failure".into());
        }
        let data = std::fs::read(glb).map_err(|e| e.to_string())?;
        std::fs::write(names.partial(&names.game_glb()), &data).map_err(|e| e.to_string())?;
        let fbx = if request.fbx {
            std::fs::write(names.partial(&names.game_fbx()), b"fake fbx")
                .map_err(|e| e.to_string())?;
            Some(names.game_fbx())
        } else {
            None
        };
        let stats = crate::glb::stats_from_bytes(&data)?;
        Ok(GameCopy {
            glb: names.game_glb(),
            fbx,
            scale: request.size_m.unwrap_or(1.0),
            stats,
        })
    }
}
