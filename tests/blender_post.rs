//! Headless Blender previews and engine copy on a test box. Runs only when Blender is found (config-free discovery).
#![cfg(feature = "mock")]

use std::path::Path;

use mcp_meshgen::output::{MeshNames, MeshTarget, reserve};
use mcp_meshgen::post::PostProcessor;
use mcp_meshgen::post::blender::{BlenderPost, discover};
use mcp_meshgen::request::{ResolvedRequest, SizeAxis};
use tokio_util::sync::CancellationToken;

fn blender() -> Option<BlenderPost> {
    match discover(None) {
        Ok(_) => {
            let cache = std::env::temp_dir().join("mcp-meshgen-test-cache");
            let mut post = BlenderPost::new(None, cache);
            post.preview_size = 128;
            post.preview_samples = 4;
            Some(post)
        }
        Err(e) => {
            eprintln!("skipping: {e}");
            None
        }
    }
}

/// A 2 × 1 × 0.5 m box (glTF X, Y up, Z) written as the job's finished GLB.
fn setup(dir: &Path, stem: &str) -> (MeshNames, std::path::PathBuf) {
    let names = reserve(&MeshTarget {
        dir: dir.to_path_buf(),
        stem: stem.into(),
    })
    .unwrap();
    std::fs::write(
        names.partial(&names.glb()),
        mcp_meshgen::glb::box_mesh([2.0, 1.0, 0.5]),
    )
    .unwrap();
    let glb = names.finalize(&names.glb()).unwrap();
    (names, glb)
}

fn request(size_m: Option<f64>, axis: SizeAxis, fbx: bool) -> ResolvedRequest {
    ResolvedRequest {
        model: "m".into(),
        image: "in.png".into(),
        seed: 1,
        quality: "512".into(),
        texture_size: 1024,
        steps: None,
        target_triangles: 200_000,
        obj: false,
        previews: false,
        return_preview: false,
        size_m,
        size_axis: axis,
        fbx,
        keep_loaded: false,
        warnings: vec![],
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-3
}

#[tokio::test]
async fn previews_are_four_transparent_renders() {
    let Some(post) = blender() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (names, glb) = setup(dir.path(), "box");
    let paths = post
        .previews(&glb, &names, &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(paths.len(), 4);
    for path in &paths {
        let img = image::ImageReader::open(names.partial(path))
            .unwrap()
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap()
            .to_rgba8();
        assert_eq!(
            (img.width(), img.height()),
            (128, 128),
            "{}",
            path.display()
        );
        assert_eq!(
            img.get_pixel(0, 0)[3],
            0,
            "corner is transparent: {}",
            path.display()
        );
        assert_eq!(
            img.get_pixel(64, 64)[3],
            255,
            "the box fills the centre: {}",
            path.display()
        );
        let px = img.get_pixel(64, 64);
        assert!(
            px[0] as u32 + px[1] as u32 + px[2] as u32 > 60,
            "lit, not black: {px:?} in {}",
            path.display()
        );
    }
}

#[tokio::test]
async fn engine_copy_scales_the_longest_side_and_sits_on_the_origin() {
    let Some(post) = blender() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (names, glb) = setup(dir.path(), "box");
    let copy = post
        .game_copy(
            &glb,
            &names,
            &request(Some(4.0), SizeAxis::Longest, true),
            &CancellationToken::new(),
        )
        .await
        .unwrap();
    assert!(close(copy.scale, 2.0), "{}", copy.scale);
    let s = &copy.stats;
    assert!(
        close(s.size[0], 4.0) && close(s.size[1], 2.0) && close(s.size[2], 1.0),
        "{:?}",
        s.size
    );
    // glTF is Y up: the bottom is at y = 0 and the footprint is centred.
    assert!(close(s.bounds_min[1], 0.0), "{:?}", s.bounds_min);
    assert!(close(s.bounds_min[0], -s.bounds_max[0]) && close(s.bounds_min[2], -s.bounds_max[2]));
    assert_eq!(s.triangles, 12, "topology untouched");
    assert_eq!(s.textures.len(), 1, "texture carried over");
    let fbx = names.partial(&names.game_fbx());
    let data = std::fs::read(&fbx).unwrap();
    assert!(data.starts_with(b"Kaydara FBX Binary"), "binary FBX");
}

#[tokio::test]
async fn engine_copy_can_scale_by_height() {
    let Some(post) = blender() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (names, glb) = setup(dir.path(), "box");
    let copy = post
        .game_copy(
            &glb,
            &names,
            &request(Some(4.0), SizeAxis::Height, false),
            &CancellationToken::new(),
        )
        .await
        .unwrap();
    assert!(close(copy.scale, 4.0), "{}", copy.scale);
    assert!(close(copy.stats.size[1], 4.0), "{:?}", copy.stats.size);
    assert!(copy.fbx.is_none() && !names.partial(&names.game_fbx()).exists());
}

#[tokio::test]
async fn cancel_stops_blender() {
    let Some(post) = blender() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (names, glb) = setup(dir.path(), "box");
    let cancel = CancellationToken::new();
    cancel.cancel();
    let err = post.previews(&glb, &names, &cancel).await.unwrap_err();
    assert_eq!(err, "cancelled");
}
