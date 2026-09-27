//! `generate_mesh` input, validation and defaults (design §5; R1, R8, R9, R10).

use std::path::PathBuf;

use rmcp::schemars;
use serde::{Deserialize, Serialize};

use crate::error::MeshGenError;
use crate::output::{MeshTarget, PathPolicy};
use crate::registry::{ModelRegistry, ModelSpec};

pub const MAX_STEPS: u32 = 50;
/// Below this the baker's BVH has too little to work with, and a prop falls apart.
pub const MIN_TRIANGLES: u32 = 1_000;
pub const MIN_SIZE_M: f64 = 0.01;
pub const MAX_SIZE_M: f64 = 1_000.0;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum SizeAxis {
    /// Scale so the longest side of the bounding box equals size_m.
    #[default]
    Longest,
    /// Scale so the height (Z, after Z-up) equals size_m.
    Height,
}

#[derive(Debug, Clone, Default, Deserialize, schemars::JsonSchema)]
pub struct MeshRequest {
    /// Input image: a cut-out (PNG or WebP with a transparent background) inside an allowed root. Relative paths
    /// resolve against the config dir's parent.
    pub image: String,
    /// Model id from list_models. Defaults to config `default_model`.
    #[serde(default)]
    pub model: Option<String>,
    /// Seed for reproducibility. Random when omitted; always reported back.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Pipeline resolution, one of the model's quality options (e.g. "512", "1024", "1024_cascade").
    #[serde(default)]
    pub quality: Option<String>,
    /// Baked texture size in pixels, one of the model's texture_sizes.
    #[serde(default)]
    pub texture_size: Option<u32>,
    /// Sampler steps for every flow stage (1-50). Defaults to the pipeline's own (12 for TRELLIS.2).
    #[serde(default)]
    pub steps: Option<u32>,
    /// Exact output file (.glb) inside an allowed root. Never overwrites: a taken name gets a -1, -2… suffix.
    #[serde(default)]
    pub output_path: Option<String>,
    /// Output folder; the file name is <image name>-<seed>.glb. Default: config default_output_dir, else next to the image.
    #[serde(default)]
    pub output_dir: Option<String>,
    /// Also write an OBJ (with its .mtl and textures).
    #[serde(default)]
    pub obj: bool,
    /// Render front, side, back and three-quarter PNGs with headless Blender.
    #[serde(default)]
    pub previews: bool,
    /// Include a small inline preview image in the result (implies previews).
    #[serde(default)]
    pub return_preview: bool,
    /// Triangle budget: the mesh is simplified to this before the textures are baked (1000 up to the model's max).
    #[serde(default)]
    pub target_triangles: Option<u32>,
    /// Real size in metres for the engine copy: scales the longest side or the height (size_axis) to this.
    #[serde(default)]
    pub size_m: Option<f64>,
    /// Which dimension size_m sets. Default "longest".
    #[serde(default)]
    pub size_axis: Option<SizeAxis>,
    /// Also export <name>_game.fbx with the verified Unreal settings (needs Blender).
    #[serde(default)]
    pub fbx: bool,
    /// Keep the model loaded after this job (overrides config keep_loaded). Holds ~15-20 GB until unload_models.
    #[serde(default)]
    pub keep_loaded: Option<bool>,
    /// Wait for the mesh (default). false returns a job_id to poll with get_job.
    #[serde(default = "default_true")]
    pub wait: bool,
}

fn default_true() -> bool {
    true
}

/// Fully resolved parameters: what the backend runs and the sidecar records.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResolvedRequest {
    pub model: String,
    pub image: PathBuf,
    pub seed: u64,
    pub quality: String,
    pub texture_size: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    /// Faces kept before baking; the model's max when the request gives none.
    pub target_triangles: u32,
    pub obj: bool,
    pub previews: bool,
    pub return_preview: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_m: Option<f64>,
    pub size_axis: SizeAxis,
    pub fbx: bool,
    pub keep_loaded: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl ResolvedRequest {
    /// The engine copy (`_game.glb`, optional `_game.fbx`) is made when a size or an FBX is asked for (R9).
    pub fn wants_game_copy(&self) -> bool {
        self.size_m.is_some() || self.fbx
    }

    /// Whether Blender has any work (previews or the engine copy).
    pub fn needs_blender(&self) -> bool {
        self.previews || self.wants_game_copy()
    }
}

impl MeshRequest {
    /// Validate against the registry and the path policy, and fill in defaults. The input file is checked to exist
    /// inside a root; its cut-out check (`input.rs`) comes next.
    pub fn resolve<'r>(
        &self,
        registry: &'r ModelRegistry,
        default_model: &str,
        default_keep_loaded: bool,
        policy: &PathPolicy,
    ) -> Result<(ResolvedRequest, &'r ModelSpec, MeshTarget), MeshGenError> {
        let model_id = self
            .model
            .as_deref()
            .map(str::trim)
            .filter(|m| !m.is_empty())
            .unwrap_or(default_model);
        let model = registry.get(model_id)?;
        let image = policy.input(&self.image)?;
        let mut warnings = Vec::new();

        let quality = match self
            .quality
            .as_deref()
            .map(str::trim)
            .filter(|q| !q.is_empty())
        {
            Some(q) if model.quality.iter().any(|m| m == q) => q.to_string(),
            Some(q) => {
                return Err(MeshGenError::invalid(
                    "quality",
                    format!("`{q}` is not one of {:?} for {}", model.quality, model.id),
                ));
            }
            None => model.defaults.quality.clone(),
        };
        let texture_size = match self.texture_size {
            Some(t) if model.texture_sizes.contains(&t) => t,
            Some(t) => {
                return Err(MeshGenError::invalid(
                    "texture_size",
                    format!(
                        "{t} is not one of {:?} for {}",
                        model.texture_sizes, model.id
                    ),
                ));
            }
            None => model.defaults.texture_size,
        };
        if let Some(steps) = self.steps
            && !(1..=MAX_STEPS).contains(&steps)
        {
            return Err(MeshGenError::invalid(
                "steps",
                format!("must be 1-{MAX_STEPS} (got {steps})"),
            ));
        }
        let target_triangles = match self.target_triangles {
            Some(t) if (MIN_TRIANGLES..=model.max_triangles).contains(&t) => t,
            Some(t) => {
                return Err(MeshGenError::invalid(
                    "target_triangles",
                    format!(
                        "must be {MIN_TRIANGLES}-{} for {} (got {t})",
                        model.max_triangles, model.id
                    ),
                ));
            }
            None => model.max_triangles,
        };
        if let Some(size) = self.size_m
            && !(size.is_finite() && (MIN_SIZE_M..=MAX_SIZE_M).contains(&size))
        {
            return Err(MeshGenError::invalid(
                "size_m",
                format!("must be {MIN_SIZE_M}-{MAX_SIZE_M} metres (got {size})"),
            ));
        }
        if self.size_axis.is_some() && self.size_m.is_none() {
            warnings.push("size_axis is ignored without size_m".to_string());
        }

        let seed = self
            .seed
            .unwrap_or_else(|| u64::from(rand::random::<u32>()));
        let target = policy.plan(
            self.output_path.as_deref(),
            self.output_dir.as_deref(),
            &image,
            seed,
        )?;
        let resolved = ResolvedRequest {
            model: model.id.clone(),
            image,
            seed,
            quality,
            texture_size,
            steps: self.steps,
            target_triangles,
            obj: self.obj,
            previews: self.previews || self.return_preview,
            return_preview: self.return_preview,
            size_m: self.size_m,
            size_axis: self.size_axis.unwrap_or_default(),
            fbx: self.fbx,
            keep_loaded: self.keep_loaded.unwrap_or(default_keep_loaded),
            warnings,
        };
        Ok((resolved, model, target))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::config::MODELS_EXAMPLE;

    struct Fixture {
        _dir: tempfile::TempDir,
        root: PathBuf,
        policy: PathPolicy,
        registry: ModelRegistry,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("game/refs")).unwrap();
        std::fs::write(root.join("game/refs/brazier.png"), b"png").unwrap();
        Fixture {
            policy: PathPolicy {
                base: root.clone(),
                default_dir: None,
                allowed_roots: vec![root.join("game")],
            },
            registry: ModelRegistry::from_toml(MODELS_EXAMPLE).unwrap(),
            root,
            _dir: dir,
        }
    }

    fn req(image: &str) -> MeshRequest {
        MeshRequest {
            image: image.into(),
            wait: true,
            ..Default::default()
        }
    }

    fn resolve(f: &Fixture, r: &MeshRequest) -> Result<ResolvedRequest, MeshGenError> {
        r.resolve(&f.registry, "trellis2-4b", false, &f.policy)
            .map(|(res, _, _)| res)
    }

    #[test]
    fn defaults_come_from_the_registry_and_config() {
        let f = fixture();
        let (r, model, target) = req("game/refs/brazier.png")
            .resolve(&f.registry, "trellis2-4b", true, &f.policy)
            .unwrap();
        assert_eq!(model.id, "trellis2-4b");
        assert_eq!(r.image, f.root.join("game/refs/brazier.png"));
        assert_eq!((r.quality.as_str(), r.texture_size), ("512", 1024));
        assert_eq!(r.target_triangles, 200_000);
        assert_eq!(r.steps, None);
        assert!(r.keep_loaded, "config default applies");
        assert!(!r.previews && !r.wants_game_copy() && !r.needs_blender());
        assert_eq!(target.dir, f.root.join("game/refs"));
        assert_eq!(target.stem, format!("brazier-{}", r.seed));
        assert!(r.seed <= u64::from(u32::MAX));
    }

    #[test]
    fn explicit_values_are_kept() {
        let f = fixture();
        let r = resolve(
            &f,
            &MeshRequest {
                model: Some("trellis2-4b".into()),
                seed: Some(42),
                quality: Some("1024_cascade".into()),
                texture_size: Some(2048),
                steps: Some(20),
                target_triangles: Some(20_000),
                size_m: Some(1.2),
                size_axis: Some(SizeAxis::Height),
                fbx: true,
                keep_loaded: Some(true),
                ..req("game/refs/brazier.png")
            },
        )
        .unwrap();
        assert_eq!(r.seed, 42);
        assert_eq!(
            (r.quality.as_str(), r.texture_size, r.steps),
            ("1024_cascade", 2048, Some(20))
        );
        assert_eq!(r.target_triangles, 20_000);
        assert_eq!((r.size_m, r.size_axis), (Some(1.2), SizeAxis::Height));
        assert!(r.wants_game_copy() && r.needs_blender() && r.keep_loaded);
    }

    #[test]
    fn return_preview_implies_previews() {
        let f = fixture();
        let r = resolve(
            &f,
            &MeshRequest {
                return_preview: true,
                ..req("game/refs/brazier.png")
            },
        )
        .unwrap();
        assert!(r.previews && r.needs_blender() && !r.wants_game_copy());
    }

    #[test]
    fn size_axis_without_size_warns() {
        let f = fixture();
        let r = resolve(
            &f,
            &MeshRequest {
                size_axis: Some(SizeAxis::Height),
                ..req("game/refs/brazier.png")
            },
        )
        .unwrap();
        assert_eq!(r.warnings, vec!["size_axis is ignored without size_m"]);
    }

    #[test]
    fn invalid_values_name_the_field() {
        let f = fixture();
        let field = |r: MeshRequest| match resolve(&f, &r).unwrap_err() {
            MeshGenError::InvalidRequest { field, .. } => field,
            other => panic!("{other}"),
        };
        let base = || req("game/refs/brazier.png");
        assert_eq!(
            field(MeshRequest {
                quality: Some("2048".into()),
                ..base()
            }),
            "quality"
        );
        assert_eq!(
            field(MeshRequest {
                texture_size: Some(4096),
                ..base()
            }),
            "texture_size"
        );
        assert_eq!(
            field(MeshRequest {
                steps: Some(0),
                ..base()
            }),
            "steps"
        );
        assert_eq!(
            field(MeshRequest {
                steps: Some(51),
                ..base()
            }),
            "steps"
        );
        assert_eq!(
            field(MeshRequest {
                target_triangles: Some(999),
                ..base()
            }),
            "target_triangles"
        );
        assert_eq!(
            field(MeshRequest {
                target_triangles: Some(200_001),
                ..base()
            }),
            "target_triangles"
        );
        assert_eq!(
            field(MeshRequest {
                size_m: Some(0.0),
                ..base()
            }),
            "size_m"
        );
        assert_eq!(
            field(MeshRequest {
                size_m: Some(f64::NAN),
                ..base()
            }),
            "size_m"
        );
        assert_eq!(field(req("")), "image");
        assert_eq!(
            field(MeshRequest {
                output_path: Some("game/x.fbx".into()),
                ..base()
            }),
            "output_path"
        );
    }

    #[test]
    fn model_and_paths_are_checked() {
        let f = fixture();
        let err = resolve(
            &f,
            &MeshRequest {
                model: Some("nope".into()),
                ..req("game/refs/brazier.png")
            },
        )
        .unwrap_err();
        assert_eq!(err.code(), "unknown_model");
        let outside = tempfile::tempdir().unwrap();
        let foreign = outside.path().join("x.png");
        std::fs::write(&foreign, b"png").unwrap();
        assert_eq!(
            resolve(&f, &req(foreign.to_str().unwrap()))
                .unwrap_err()
                .code(),
            "path_not_allowed"
        );
        let err = resolve(
            &f,
            &MeshRequest {
                output_dir: Some("elsewhere".into()),
                ..req("game/refs/brazier.png")
            },
        )
        .unwrap_err();
        assert_eq!(err.code(), "path_not_allowed");
    }

    #[test]
    fn output_path_sets_the_stem() {
        let f = fixture();
        let (_, _, target) = MeshRequest {
            output_path: Some("game/assets/export/SM_Brazier.glb".into()),
            ..req("game/refs/brazier.png")
        }
        .resolve(&f.registry, "trellis2-4b", false, &f.policy)
        .unwrap();
        assert_eq!(target.dir, f.root.join("game/assets/export"));
        assert_eq!(target.stem, "SM_Brazier");
        assert!(Path::new(&target.dir).starts_with(&f.root));
    }

    #[test]
    fn schema_lists_every_parameter() {
        let schema = serde_json::to_value(schemars::schema_for!(MeshRequest)).unwrap();
        let props = schema["properties"].as_object().unwrap();
        for key in [
            "image",
            "model",
            "seed",
            "quality",
            "texture_size",
            "steps",
            "output_path",
            "output_dir",
            "obj",
            "previews",
            "return_preview",
            "target_triangles",
            "size_m",
            "size_axis",
            "fbx",
            "keep_loaded",
            "wait",
        ] {
            assert!(props.contains_key(key), "missing {key}");
        }
        assert_eq!(schema["required"], serde_json::json!(["image"]));
    }
}
