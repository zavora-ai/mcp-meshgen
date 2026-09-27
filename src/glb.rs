//! Minimal glTF binary (GLB) reader for result stats (R1), independent of the backend, plus a textured cube writer
//! for tests and the mock backends.

use std::io::Cursor;
use std::path::Path;

use serde::Serialize;
use serde_json::Value;

/// Mesh statistics reported in results and sidecars (R1).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct MeshStats {
    pub vertices: u64,
    pub triangles: u64,
    /// Axis-aligned bounds in the file's own units and axes (glTF: metres, Y-up), node transforms applied.
    pub bounds_min: [f64; 3],
    pub bounds_max: [f64; 3],
    pub size: [f64; 3],
    pub materials: usize,
    /// Embedded texture sizes as `[width, height]`.
    pub textures: Vec<[u32; 2]>,
}

type Mat4 = [[f64; 4]; 4];

const IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Read stats from a `.glb` file.
pub fn stats(path: &Path) -> Result<MeshStats, String> {
    let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    stats_from_bytes(&data)
}

pub fn stats_from_bytes(data: &[u8]) -> Result<MeshStats, String> {
    let (json, bin) = split_chunks(data)?;
    let doc: Value = serde_json::from_slice(json).map_err(|e| format!("bad glTF JSON: {e}"))?;
    let accessors = doc["accessors"].as_array().cloned().unwrap_or_default();
    let meshes = doc["meshes"].as_array().cloned().unwrap_or_default();
    let nodes = doc["nodes"].as_array().cloned().unwrap_or_default();

    let mut stats = MeshStats {
        materials: doc["materials"].as_array().map_or(0, Vec::len),
        ..Default::default()
    };
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];

    // Instances: every node reachable from the scene that references a mesh, with its world matrix.
    let scene = doc["scene"].as_u64().unwrap_or(0) as usize;
    let roots: Vec<usize> = doc["scenes"][scene]["nodes"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|n| n.as_u64().map(|n| n as usize))
                .collect()
        })
        .unwrap_or_else(|| (0..nodes.len()).collect());
    let mut stack: Vec<(usize, Mat4, usize)> =
        roots.into_iter().map(|n| (n, IDENTITY, 0)).collect();
    while let Some((idx, parent, depth)) = stack.pop() {
        if depth > 64 {
            return Err("node hierarchy too deep (cycle?)".into());
        }
        let node = nodes
            .get(idx)
            .ok_or_else(|| format!("missing node {idx}"))?;
        let world = mul(&parent, &local_matrix(node));
        if let Some(mesh) = node["mesh"].as_u64() {
            let mesh = meshes.get(mesh as usize).ok_or("missing mesh")?;
            for prim in mesh["primitives"].as_array().into_iter().flatten() {
                let mode = prim["mode"].as_u64().unwrap_or(4);
                let pos = &accessors[prim["attributes"]["POSITION"]
                    .as_u64()
                    .ok_or("primitive without POSITION")?
                    as usize];
                let count = pos["count"].as_u64().unwrap_or(0);
                stats.vertices += count;
                let index_count = prim["indices"]
                    .as_u64()
                    .map(|i| accessors[i as usize]["count"].as_u64().unwrap_or(0))
                    .unwrap_or(count);
                if mode == 4 {
                    stats.triangles += index_count / 3;
                }
                if let (Some(lo), Some(hi)) = (vec3(&pos["min"]), vec3(&pos["max"])) {
                    for corner in corners(lo, hi) {
                        let p = transform(&world, corner);
                        for k in 0..3 {
                            min[k] = min[k].min(p[k]);
                            max[k] = max[k].max(p[k]);
                        }
                    }
                }
            }
        }
        for child in node["children"].as_array().into_iter().flatten() {
            if let Some(c) = child.as_u64() {
                stack.push((c as usize, world, depth + 1));
            }
        }
    }
    if stats.vertices > 0 && min[0].is_finite() {
        stats.bounds_min = min;
        stats.bounds_max = max;
        stats.size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    }

    let views = doc["bufferViews"].as_array().cloned().unwrap_or_default();
    for image in doc["images"].as_array().into_iter().flatten() {
        let Some(view) = image["bufferView"]
            .as_u64()
            .and_then(|v| views.get(v as usize))
        else {
            continue;
        };
        let start = view["byteOffset"].as_u64().unwrap_or(0) as usize;
        let len = view["byteLength"].as_u64().unwrap_or(0) as usize;
        let Some(bytes) = bin.get(start..start.saturating_add(len)) else {
            continue;
        };
        if let Ok(dims) = image::ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|_| ())
            .and_then(|r| r.into_dimensions().map_err(|_| ()))
        {
            stats.textures.push([dims.0, dims.1]);
        }
    }
    Ok(stats)
}

fn split_chunks(data: &[u8]) -> Result<(&[u8], &[u8]), String> {
    let u32_at = |at: usize| -> Result<u32, String> {
        data.get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .ok_or_else(|| "truncated GLB".to_string())
    };
    if data.get(0..4) != Some(b"glTF") {
        return Err("not a GLB file (bad magic)".into());
    }
    if u32_at(4)? != 2 {
        return Err("unsupported glTF version (need 2)".into());
    }
    let json_len = u32_at(12)? as usize;
    if u32_at(16)? != 0x4E4F_534A {
        return Err("first GLB chunk is not JSON".into());
    }
    let json = data.get(20..20 + json_len).ok_or("truncated JSON chunk")?;
    let bin_at = 20 + json_len;
    let bin = if data.len() >= bin_at + 8 && u32_at(bin_at + 4)? == 0x004E_4942 {
        let len = u32_at(bin_at)? as usize;
        data.get(bin_at + 8..bin_at + 8 + len)
            .ok_or("truncated BIN chunk")?
    } else {
        &[]
    };
    Ok((json, bin))
}

fn vec3(v: &Value) -> Option<[f64; 3]> {
    let a = v.as_array()?;
    Some([
        a.first()?.as_f64()?,
        a.get(1)?.as_f64()?,
        a.get(2)?.as_f64()?,
    ])
}

fn corners(lo: [f64; 3], hi: [f64; 3]) -> impl Iterator<Item = [f64; 3]> {
    (0..8).map(move |i| {
        [
            if i & 1 == 0 { lo[0] } else { hi[0] },
            if i & 2 == 0 { lo[1] } else { hi[1] },
            if i & 4 == 0 { lo[2] } else { hi[2] },
        ]
    })
}

/// Column-major glTF `matrix`, else translation × rotation × scale.
fn local_matrix(node: &Value) -> Mat4 {
    if let Some(m) = node["matrix"].as_array().filter(|m| m.len() == 16) {
        let f: Vec<f64> = m.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
        let mut out = IDENTITY;
        for (col, chunk) in f.chunks(4).enumerate() {
            for (row, v) in chunk.iter().enumerate() {
                out[row][col] = *v;
            }
        }
        return out;
    }
    let t = vec3(&node["translation"]).unwrap_or([0.0; 3]);
    let s = vec3(&node["scale"]).unwrap_or([1.0; 3]);
    let q = node["rotation"]
        .as_array()
        .filter(|r| r.len() == 4)
        .map(|r| [0, 1, 2, 3].map(|i| r[i].as_f64().unwrap_or(0.0)))
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let [x, y, z, w] = q;
    let r = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let mut out = IDENTITY;
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = r[row][col] * s[col];
        }
        out[row][3] = t[row];
    }
    out
}

fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn transform(m: &Mat4, p: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|r| m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3])
}

/// A 1 m cube (24 vertices, 12 triangles, origin at its centre) with one material and a 16×16 base-colour texture.
#[cfg(any(test, feature = "mock"))]
pub fn cube() -> Vec<u8> {
    box_mesh([1.0, 1.0, 1.0])
}

/// A box of `size` metres (glTF axes: X, Y up, Z), centred on the origin, textured like `cube`.
#[cfg(any(test, feature = "mock"))]
pub fn box_mesh(size: [f32; 3]) -> Vec<u8> {
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ([0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ];
    let (mut pos, mut nrm, mut uv, mut idx) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::<u16>::new());
    for (n, u, v) in faces {
        let base = (pos.len() / 3) as u16;
        for (su, sv) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            for k in 0..3 {
                pos.push(0.5 * size[k] * (n[k] + su * u[k] + sv * v[k]));
            }
            nrm.extend_from_slice(&n);
            uv.extend_from_slice(&[(su + 1.0) / 2.0, (1.0 - sv) / 2.0]);
        }
        // Counter-clockwise seen from outside: u × v = n for every face above.
        idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    let texture = {
        let img = image::RgbaImage::from_fn(16, 16, |x, y| {
            if (x / 4 + y / 4) % 2 == 0 {
                image::Rgba([200, 150, 60, 255])
            } else {
                image::Rgba([90, 60, 30, 255])
            }
        });
        let mut out = Vec::new();
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
            .expect("in-memory PNG");
        out
    };

    let mut bin = Vec::new();
    let mut views = Vec::new();
    let mut push = |bytes: &[u8], target: Option<u32>| {
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let mut view =
            serde_json::json!({ "buffer": 0, "byteOffset": bin.len(), "byteLength": bytes.len() });
        if let Some(t) = target {
            view["target"] = t.into();
        }
        bin.extend_from_slice(bytes);
        views.push(view);
    };
    let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    push(&f32s(&pos), Some(34962));
    push(&f32s(&nrm), Some(34962));
    push(&f32s(&uv), Some(34962));
    push(
        &idx.iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
        Some(34963),
    );
    push(&texture, None);
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }

    let doc = serde_json::json!({
        "asset": { "version": "2.0", "generator": "mcp-meshgen test cube" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0, "name": "cube" }],
        "meshes": [{ "primitives": [{
            "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 },
            "indices": 3, "material": 0, "mode": 4
        }]}],
        "materials": [{ "name": "cube", "pbrMetallicRoughness": {
            "baseColorTexture": { "index": 0 }, "metallicFactor": 0.0, "roughnessFactor": 0.8
        }}],
        "textures": [{ "source": 0 }],
        "images": [{ "bufferView": 4, "mimeType": "image/png" }],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 24, "type": "VEC3",
              "min": size.map(|v| -0.5 * v), "max": size.map(|v| 0.5 * v) },
            { "bufferView": 1, "componentType": 5126, "count": 24, "type": "VEC3" },
            { "bufferView": 2, "componentType": 5126, "count": 24, "type": "VEC2" },
            { "bufferView": 3, "componentType": 5123, "count": 36, "type": "SCALAR" }
        ],
        "bufferViews": views,
        "buffers": [{ "byteLength": bin.len() }]
    });
    let mut json = serde_json::to_vec(&doc).expect("json");
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    let total = 12 + 8 + json.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
    out.extend_from_slice(&json);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x004E_4942u32.to_le_bytes());
    out.extend_from_slice(&bin);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    #[test]
    fn box_stats() {
        let s = stats_from_bytes(&box_mesh([2.0, 1.0, 0.5])).unwrap();
        assert!(close(s.size, [2.0, 1.0, 0.5]), "{:?}", s.size);
    }

    #[test]
    fn cube_stats() {
        let s = stats_from_bytes(&cube()).unwrap();
        assert_eq!((s.vertices, s.triangles, s.materials), (24, 12, 1));
        assert!(close(s.bounds_min, [-0.5; 3]) && close(s.bounds_max, [0.5; 3]));
        assert!(close(s.size, [1.0; 3]));
        assert_eq!(s.textures, vec![[16, 16]]);
    }

    /// Re-wrap the cube's JSON with a modified node to check transforms.
    fn with_node(node: Value) -> Vec<u8> {
        let data = cube();
        let (json, bin) = split_chunks(&data).unwrap();
        let mut doc: Value = serde_json::from_slice(json).unwrap();
        doc["nodes"] = serde_json::json!([node, { "mesh": 0 }]);
        doc["scenes"] = serde_json::json!([{ "nodes": [0] }]);
        let mut json = serde_json::to_vec(&doc).unwrap();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let total = 28 + json.len() + bin.len();
        let mut out = b"glTF".to_vec();
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
        out.extend_from_slice(&json);
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x004E_4942u32.to_le_bytes());
        out.extend_from_slice(bin);
        out
    }

    #[test]
    fn node_transforms_apply_through_children() {
        // Parent scales by (2, 1, 3) and moves up 1; the child holds the mesh.
        let s = stats_from_bytes(&with_node(serde_json::json!({
            "children": [1], "scale": [2.0, 1.0, 3.0], "translation": [0.0, 1.0, 0.0]
        })))
        .unwrap();
        assert!(close(s.size, [2.0, 1.0, 3.0]), "{:?}", s.size);
        assert!(close(s.bounds_min, [-1.0, 0.5, -1.5]), "{:?}", s.bounds_min);
    }

    #[test]
    fn rotation_and_matrix_forms() {
        // 90° about X swaps Y and Z extents; a column-major matrix translating +2 in X.
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let rotated = stats_from_bytes(&with_node(serde_json::json!({
            "children": [1], "rotation": [h, 0.0, 0.0, h], "scale": [1.0, 2.0, 1.0]
        })))
        .unwrap();
        assert!(close(rotated.size, [1.0, 1.0, 2.0]), "{:?}", rotated.size);
        let moved = stats_from_bytes(&with_node(serde_json::json!({
            "children": [1],
            "matrix": [1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0, 2.0,0.0,0.0,1.0]
        })))
        .unwrap();
        assert!(
            close(moved.bounds_min, [1.5, -0.5, -0.5]),
            "{:?}",
            moved.bounds_min
        );
    }

    #[test]
    fn rejects_non_glb() {
        assert!(
            stats_from_bytes(b"not a glb at all")
                .unwrap_err()
                .contains("magic")
        );
        assert!(stats_from_bytes(b"glTF").is_err());
    }
}

/// Run by hand on real files: `MESHGEN_GLBS="a.glb:b.glb" cargo test real_glbs -- --ignored --nocapture`.
#[cfg(test)]
mod live {
    #[test]
    #[ignore]
    fn real_glbs() {
        let paths = std::env::var("MESHGEN_GLBS").unwrap_or_default();
        for p in std::env::split_paths(&paths) {
            println!("{}: {:?}", p.display(), super::stats(&p));
        }
    }
}
