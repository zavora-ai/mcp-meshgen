//! Input check (design §4.3; R2): the image must already be cut out. Measured before anything loads.
//!
//! The server never removes backgrounds itself; an input without a real cut-out is rejected with the measured
//! coverage and a pointer to mcp-imagegen `edit_image`.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::config::InputThresholds;
use crate::error::MeshGenError;

/// Refuse files bigger than this outright (a cut-out reference is a few MB at most).
pub const MAX_INPUT_BYTES: u64 = 64 * 1024 * 1024;
/// Longest side decoded; TRELLIS works at 518 px, so anything larger is only resized away.
pub const MAX_INPUT_SIDE: u32 = 8192;

/// What was measured, recorded in the result and sidecar.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InputReport {
    pub path: PathBuf,
    pub format: String,
    pub width: u32,
    pub height: u32,
    pub sha256: String,
    /// Share of fully transparent pixels (alpha 0).
    pub transparent: f64,
    /// Share of opaque pixels (alpha ≥ `opaque_alpha`).
    pub opaque: f64,
}

/// Decode `path` and check it's a real cut-out under `limits`.
pub fn check_cutout(path: &Path, limits: &InputThresholds) -> Result<InputReport, MeshGenError> {
    let invalid = |msg: String| MeshGenError::invalid("image", msg);
    let reject = |reason: String| MeshGenError::InputRejected {
        path: path.to_path_buf(),
        reason,
    };

    let len = std::fs::metadata(path)
        .map_err(|e| invalid(format!("{}: {e}", path.display())))?
        .len();
    if len > MAX_INPUT_BYTES {
        return Err(invalid(format!(
            "{} is {len} bytes; the limit is {MAX_INPUT_BYTES}",
            path.display()
        )));
    }
    let data = std::fs::read(path).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
    let format = match image::guess_format(&data) {
        Ok(image::ImageFormat::Png) => "png",
        Ok(image::ImageFormat::WebP) => "webp",
        Ok(image::ImageFormat::Jpeg) => {
            return Err(reject(
                "it's a JPEG, which has no alpha channel; a cut-out needs a PNG or WebP with transparency".into(),
            ));
        }
        _ => {
            return Err(invalid(format!(
                "{} is not a PNG or WebP image",
                path.display()
            )));
        }
    };

    let mut reader = image::ImageReader::new(Cursor::new(&data))
        .with_guessed_format()
        .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
    let mut decode_limits = image::Limits::default();
    decode_limits.max_image_width = Some(MAX_INPUT_SIDE);
    decode_limits.max_image_height = Some(MAX_INPUT_SIDE);
    reader.limits(decode_limits);
    let img = reader
        .decode()
        .map_err(|e| invalid(format!("{}: unreadable image: {e}", path.display())))?;
    let (width, height) = (img.width(), img.height());

    if !img.color().has_alpha() {
        return Err(reject(format!(
            "the {} has no alpha channel, so there is no cut-out",
            format.to_uppercase()
        )));
    }
    let (transparent, opaque) = alpha_coverage(&img.to_rgba8(), limits.opaque_alpha);

    let pct = |v: f64| format!("{:.1}%", v * 100.0);
    let measured = format!("{} transparent, {} opaque", pct(transparent), pct(opaque));
    if transparent < limits.min_transparent {
        return Err(reject(format!(
            "{measured}; at least {} of the pixels must be fully transparent, so the background is still there",
            pct(limits.min_transparent)
        )));
    }
    if transparent > limits.max_transparent {
        return Err(reject(format!(
            "{measured}; more than {} is transparent, so almost nothing is left to convert",
            pct(limits.max_transparent)
        )));
    }
    if opaque < limits.min_opaque {
        return Err(reject(format!(
            "{measured}; at least {} must be opaque, so the subject is missing or faded",
            pct(limits.min_opaque)
        )));
    }

    Ok(InputReport {
        path: path.to_path_buf(),
        format: format.into(),
        width,
        height,
        sha256: format!("{:x}", Sha256::digest(&data)),
        transparent,
        opaque,
    })
}

/// Shares of fully transparent (alpha 0) and opaque (alpha ≥ `opaque_alpha`) pixels.
fn alpha_coverage(img: &image::RgbaImage, opaque_alpha: u8) -> (f64, f64) {
    let total = (img.width() as u64 * img.height() as u64).max(1) as f64;
    let (mut transparent, mut opaque) = (0u64, 0u64);
    for px in img.pixels() {
        match px[3] {
            0 => transparent += 1,
            a if a >= opaque_alpha => opaque += 1,
            _ => {}
        }
    }
    (transparent as f64 / total, opaque as f64 / total)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: InputThresholds = InputThresholds {
        min_transparent: 0.05,
        max_transparent: 0.95,
        min_opaque: 0.02,
        opaque_alpha: 205,
    };

    /// A 100×100 RGBA image whose first `opaque_rows` rows have alpha `a`, the rest alpha 0.
    fn rgba(dir: &Path, name: &str, rows: u32, a: u8, format: image::ImageFormat) -> PathBuf {
        let img = image::RgbaImage::from_fn(100, 100, |_, y| {
            image::Rgba([180, 140, 60, if y < rows { a } else { 0 }])
        });
        let path = dir.join(name);
        image::DynamicImage::ImageRgba8(img)
            .save_with_format(&path, format)
            .unwrap();
        path
    }

    fn rgb(dir: &Path, name: &str, format: image::ImageFormat) -> PathBuf {
        let img = image::RgbImage::from_pixel(64, 64, image::Rgb([200, 30, 30]));
        let path = dir.join(name);
        image::DynamicImage::ImageRgb8(img)
            .save_with_format(&path, format)
            .unwrap();
        path
    }

    fn reason(path: &Path) -> String {
        let err = check_cutout(path, &LIMITS).unwrap_err();
        assert_eq!(err.code(), "input_rejected", "{err}");
        let msg = err.to_string();
        assert!(msg.contains("edit_image"), "{msg}");
        msg
    }

    #[test]
    fn accepts_a_real_cutout_and_fingerprints_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = rgba(dir.path(), "cut.png", 60, 255, image::ImageFormat::Png);
        let r = check_cutout(&path, &LIMITS).unwrap();
        assert_eq!((r.format.as_str(), r.width, r.height), ("png", 100, 100));
        assert!((r.transparent - 0.40).abs() < 1e-9);
        assert!((r.opaque - 0.60).abs() < 1e-9);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(r.sha256, format!("{:x}", Sha256::digest(&bytes)));
        let webp = rgba(dir.path(), "cut.webp", 50, 255, image::ImageFormat::WebP);
        assert_eq!(check_cutout(&webp, &LIMITS).unwrap().format, "webp");
    }

    #[test]
    fn rejects_images_without_alpha() {
        let dir = tempfile::tempdir().unwrap();
        let msg = reason(&rgb(dir.path(), "flat.png", image::ImageFormat::Png));
        assert!(msg.contains("no alpha channel"), "{msg}");
        let msg = reason(&rgb(dir.path(), "photo.jpg", image::ImageFormat::Jpeg));
        assert!(msg.contains("JPEG"), "{msg}");
    }

    #[test]
    fn rejects_nearly_all_opaque_with_the_numbers() {
        let dir = tempfile::tempdir().unwrap();
        // 98 of 100 rows opaque: 2% transparent.
        let msg = reason(&rgba(
            dir.path(),
            "bg.png",
            98,
            255,
            image::ImageFormat::Png,
        ));
        assert!(msg.contains("2.0% transparent, 98.0% opaque"), "{msg}");
        assert!(msg.contains("background is still there"), "{msg}");
    }

    #[test]
    fn rejects_nearly_all_transparent() {
        let dir = tempfile::tempdir().unwrap();
        let msg = reason(&rgba(
            dir.path(),
            "empty.png",
            3,
            255,
            image::ImageFormat::Png,
        ));
        assert!(msg.contains("97.0% transparent"), "{msg}");
        assert!(msg.contains("almost nothing is left"), "{msg}");
    }

    #[test]
    fn rejects_a_faded_subject() {
        let dir = tempfile::tempdir().unwrap();
        // Half the pixels are semi-transparent (alpha 100), none opaque.
        let msg = reason(&rgba(
            dir.path(),
            "ghost.png",
            50,
            100,
            image::ImageFormat::Png,
        ));
        assert!(msg.contains("0.0% opaque"), "{msg}");
        assert!(msg.contains("subject is missing or faded"), "{msg}");
    }

    #[test]
    fn thresholds_come_from_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = rgba(dir.path(), "bg.png", 98, 255, image::ImageFormat::Png);
        let lenient = InputThresholds {
            min_transparent: 0.01,
            ..LIMITS
        };
        assert!(check_cutout(&path, &lenient).is_ok());
        // The opaque cut-off decides what counts as opaque.
        let semi = rgba(dir.path(), "semi.png", 50, 210, image::ImageFormat::Png);
        assert!(check_cutout(&semi, &LIMITS).is_ok());
        let strict = InputThresholds {
            opaque_alpha: 255,
            ..LIMITS
        };
        assert!(check_cutout(&semi, &strict).is_err());
    }

    #[test]
    fn palette_png_with_transparency_counts_as_alpha() {
        // Indexed PNGs with a tRNS chunk decode with alpha; the worker converts every input to RGBA.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("indexed.png");
        let file = std::fs::File::create(&path).unwrap();
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), 10, 10);
        enc.set_color(png::ColorType::Indexed);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_palette(vec![0, 0, 0, 200, 150, 50]);
        enc.set_trns(vec![0, 255]);
        let mut writer = enc.write_header().unwrap();
        let pixels: Vec<u8> = (0..100).map(|i| u8::from(i >= 50)).collect();
        writer.write_image_data(&pixels).unwrap();
        writer.finish().unwrap();
        let r = check_cutout(&path, &LIMITS).unwrap();
        assert!((r.transparent - 0.5).abs() < 1e-9);
    }

    #[test]
    fn non_images_are_invalid_requests() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fake.png");
        std::fs::write(&path, b"just text").unwrap();
        let err = check_cutout(&path, &LIMITS).unwrap_err();
        assert_eq!(err.code(), "invalid_request");
        assert!(err.to_string().contains("not a PNG or WebP"), "{err}");
    }
}

/// Run by hand on real files: `MESHGEN_INPUTS="a.png:b.png" cargo test real_inputs -- --ignored --nocapture`.
#[cfg(test)]
mod live {
    use super::*;

    #[test]
    #[ignore]
    fn real_inputs() {
        let paths = std::env::var("MESHGEN_INPUTS").unwrap_or_default();
        for p in std::env::split_paths(&paths) {
            match check_cutout(&p, &InputThresholds::default()) {
                Ok(r) => println!(
                    "OK   {} {}x{} transparent {:.1}% opaque {:.1}%",
                    p.display(),
                    r.width,
                    r.height,
                    r.transparent * 100.0,
                    r.opaque * 100.0
                ),
                Err(e) => println!("FAIL {e}"),
            }
        }
    }
}
