"""mcp-meshgen worker for trellis-mac (TRELLIS.2 on Apple Silicon).

Runs inside trellis-mac's venv:  <venv python> -u trellis_mac_worker.py <trellis-mac checkout>
The server embeds this file and writes it to its cache dir, so the checkout stays unmodified.

Protocol: one JSON object per line. Protocol lines go to the real stdout; everything else the libraries print goes to
stderr (the server keeps its tail for error reports).
  server -> worker  {"op": "generate", "id": "...", "image": "...", "pretrained": "microsoft/TRELLIS.2-4B", "seed": 42,
                     "quality": "512", "texture_size": 1024, "steps": null, "decimation_target": 200000,
                     "out_glb": ".../x.glb.partial", "out_obj": null}
                    {"op": "ping"}   {"op": "unload"}   (EOF also ends the worker)
  worker -> server  {"event": "hello", "protocol": 1, ...}
                    {"event": "loading", "id": ...}
                    {"event": "progress", "id": ..., "stage": "sparse_structure|shape|texture|decode|bake|export",
                     "step": 3, "total": 12}
                    {"event": "result", "id": ..., "timings": {...}, "peak_rss_mb": ..., "fixes": [...], ...}
                    {"event": "error", "id": ..., "stage": "...", "kind": "...", "message": "...", "hint": "..."}

Fixes applied at load time (A4), each reported in `fixes` only when it took effect:
  mps_sparse_decoder_cpu  the dense conv3d sparse-structure decoder returns one constant value per voxel on MPS (torch 2.14),
                          so the object decodes empty; it runs on the CPU in fp32 instead (it's small, 64^3).
Background removal is disabled: the server only sends cut-outs, and the default remover (RMBG-2.0) is gated and
non-commercial, so it's never downloaded or loaded.
"""
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

PROTOCOL = 1

# Keep a private handle on the real stdout for protocol lines, then point fd 1 (C code included) and sys.stdout at stderr.
_proto = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

STAGE_BY_DESC = {
    "Sampling sparse structure": "sparse_structure",
    "Sampling shape SLat": "shape",
    "Sampling texture SLat": "texture",
}

WATCHDOG_HINT = (
    "on Apple Silicon this is usually the macOS GPU watchdog killing a long Metal kernel, which leaves empty tensors. "
    "Close other GPU-heavy apps and external displays and retry; or retry with quality 512 / fewer steps. "
    "See https://github.com/shivampkumar/trellis-mac (GPU watchdog notes)"
)


def emit(event, **fields):
    _proto.write(json.dumps({"event": event, **fields}) + "\n")
    _proto.flush()


class WorkerError(Exception):
    def __init__(self, stage, kind, message, hint=""):
        super().__init__(message)
        self.stage, self.kind, self.message, self.hint = stage, kind, message, hint


class Job:
    """Per-request state the patched pipeline methods report into."""

    current = None

    def __init__(self, job_id):
        self.id = job_id
        self.stage = "loading"
        self.timings = {}
        self.fixes = []
        self.warnings = []
        self.details = {}

    def enter(self, stage, step=None, total=None):
        self.stage = stage
        fields = {"id": self.id, "stage": stage}
        if total is not None:
            fields.update(step=step, total=total)
        emit("progress", **fields)

    def add_time(self, name, seconds):
        self.timings[name] = round(self.timings.get(name, 0.0) + seconds, 3)

    def fix(self, name):
        if name not in self.fixes:
            self.fixes.append(name)


# ---------------------------------------------------------------------------------------------------------------------
# Environment and patches (done once per process)

_pipeline = None
_pipeline_id = None


def setup(trellis_dir):
    """Same environment as trellis-mac's generate.py, before torch is imported anywhere."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    try:
        import flex_gemm  # noqa: F401
        os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    except (ImportError, RuntimeError):
        os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
    sys.path.insert(0, str(Path(trellis_dir) / "TRELLIS.2"))
    sys.path.insert(0, str(trellis_dir))
    sys.path.append(str(Path(trellis_dir) / "stubs"))


def progress_tqdm(iterable=None, desc=None, total=None, **_kwargs):
    """Stands in for tqdm inside the samplers: reports each step as a progress event."""
    items = list(iterable) if iterable is not None else []
    n = total or len(items)
    stage = STAGE_BY_DESC.get(desc, desc or "sampling")
    job = Job.current
    t0 = time.time()
    if job:
        job.enter(stage, 0, n)
    for i, item in enumerate(items, 1):
        yield item
        if job:
            job.enter(stage, i, n)
    if job:
        job.add_time(stage, time.time() - t0)


class _NoBackgroundRemoval:
    """Replaces the pipeline's background remover so RMBG-2.0 is never downloaded or loaded."""

    def __init__(self, *args, **kwargs):
        pass

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def __call__(self, image):
        raise WorkerError(
            "loading", "needs_cutout",
            "the input has no transparent background, and background removal is disabled",
            "cut the subject out first, e.g. with mcp-imagegen edit_image",
        )


def patch_pipeline():
    import torch
    import trellis2.pipelines.rembg as rembg_module
    import trellis2.pipelines.samplers.flow_euler as flow_euler
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    # Only the remover classes TRELLIS defines (e.g. BiRefNet), not names the package re-exports from typing etc.
    for name, obj in list(vars(rembg_module).items()):
        if isinstance(obj, type) and getattr(obj, "__module__", "").startswith(rembg_module.__name__):
            setattr(rembg_module, name, _NoBackgroundRemoval)

    flow_euler.tqdm = progress_tqdm
    import tqdm as tqdm_module
    for name, module in list(sys.modules.items()):
        if name.startswith("trellis2") and getattr(module, "tqdm", None) is tqdm_module.tqdm:
            module.tqdm = progress_tqdm

    def sample_sparse_structure(self, cond, resolution, num_samples=1, sampler_params={}):
        # Upstream's method, with the dense decoder moved to the CPU under MPS (fix mps_sparse_decoder_cpu).
        flow_model = self.models["sparse_structure_flow_model"]
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model, noise, **cond, **sampler_params, verbose=True, tqdm_desc="Sampling sparse structure",
        ).samples
        if self.low_vram:
            flow_model.cpu()
        decoder = self.models["sparse_structure_decoder"]
        if z_s.device.type == "mps":
            decoder.cpu()
            decoder.convert_to_fp32()
            decoded = (decoder(z_s.float().cpu()) > 0).to(self.device)
            if Job.current:
                Job.current.fix("mps_sparse_decoder_cpu")
        else:
            if self.low_vram:
                decoder.to(self.device)
            decoded = decoder(z_s) > 0
            if self.low_vram:
                decoder.cpu()
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
        voxels = int(coords.shape[0])
        if Job.current:
            Job.current.details["voxels"] = voxels
        if voxels == 0:
            raise WorkerError(
                "sparse_structure", "empty_structure",
                "the sparse structure decoded to 0 voxels, so there is no object to build",
                "check the cut-out (subject large, centred and fully opaque, clean edges) and try another seed",
            )
        return coords

    Trellis2ImageTo3DPipeline.sample_sparse_structure = sample_sparse_structure

    original_decode = Trellis2ImageTo3DPipeline.decode_latent

    def decode_latent(self, *args, **kwargs):
        job = Job.current
        if job:
            job.enter("decode")
        t0 = time.time()
        out = original_decode(self, *args, **kwargs)
        if job:
            job.add_time("decode", time.time() - t0)
        return out

    Trellis2ImageTo3DPipeline.decode_latent = decode_latent


def load(pretrained):
    """Load (or reuse) the pipeline on MPS."""
    global _pipeline, _pipeline_id
    if _pipeline is not None and _pipeline_id == pretrained:
        return _pipeline
    unload()
    import torch

    if not torch.backends.mps.is_available():
        raise WorkerError(
            "loading", "no_mps", "PyTorch reports no MPS device",
            "trellis-mac needs an Apple Silicon Mac with a recent macOS and PyTorch built with MPS",
        )
    patch_pipeline()
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    try:
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(pretrained)
    except Exception as e:  # missing files surface as various HF / OS errors
        text = f"{type(e).__name__}: {e}"
        if any(s in text for s in ("LocalEntryNotFound", "offline", "OfflineMode", "No such file", "not found", "does not appear")):
            raise WorkerError(
                "loading", "missing_weights", text.splitlines()[0][:400],
                "a weight file isn't in the Hugging Face cache; list_models shows the exact download command",
            )
        raise
    pipeline.to(torch.device("mps"))
    _pipeline, _pipeline_id = pipeline, pretrained
    return pipeline


def unload():
    global _pipeline, _pipeline_id
    if _pipeline is None:
        return
    _pipeline, _pipeline_id = None, None
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------------------------------------------------
# One conversion

def peak_rss_mb():
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak / (1024 * 1024)) if sys.platform == "darwin" else int(peak / 1024)


def bake_and_export(job, mesh_out, req):
    """generate.py's texture bake (Metal via o_voxel, KDTree fallback), simplified to decimation_target first."""
    import numpy as np
    import torch

    verts = mesh_out.vertices.cpu().numpy()
    faces = mesh_out.faces.cpu().numpy()
    job.details["raw_triangles"] = int(faces.shape[0])
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        raise WorkerError("decode", "empty_mesh", "the decoder produced an empty mesh", WATCHDOG_HINT)

    tex_size = int(req["texture_size"])
    target_faces = min(int(req["decimation_target"]), len(faces))
    job.details["target_triangles"] = target_faces
    if not (hasattr(mesh_out, "attrs") and mesh_out.attrs is not None):
        raise WorkerError("decode", "no_texture_voxels", "the decoded mesh carries no texture voxels", "try another seed")

    job.enter("bake")
    t0 = time.time()
    glb = None
    use_metal = False
    try:
        import o_voxel.postprocess

        use_metal = getattr(o_voxel.postprocess, "_BACKEND", None) == "metal" and getattr(o_voxel.postprocess, "_HAS_DR", False)
        if use_metal and not getattr(o_voxel.postprocess, "_HAS_FLEX_GEMM", False):
            import torch.nn.functional as F

            def _gs3d_fix(feats, coords, shape, grid, mode="trilinear"):
                B, C = shape[0], shape[1]
                D, H, W = shape[2], shape[3], shape[4]
                dense = torch.zeros(B, C, D, H, W, dtype=feats.dtype, device=feats.device)
                b = coords[:, 0].long()
                cx, cy, cz = coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()
                dense[b, :, cx, cy, cz] = feats
                g = torch.stack([
                    grid[..., 2] / (W - 1) * 2 - 1,
                    grid[..., 1] / (H - 1) * 2 - 1,
                    grid[..., 0] / (D - 1) * 2 - 1,
                ], dim=-1).reshape(B, 1, 1, -1, 3)
                s = F.grid_sample(dense, g, mode="bilinear", align_corners=True, padding_mode="border")
                M = grid.shape[1]
                return s.reshape(B, C, M).permute(0, 2, 1).reshape(B * M, C)

            o_voxel.postprocess._grid_sample_3d = _gs3d_fix
    except (ImportError, AttributeError):
        use_metal = False

    if use_metal:
        try:
            import fast_simplification

            if len(faces) > target_faces:
                sv, sf = fast_simplification.simplify(verts, faces, 1.0 - target_faces / len(faces))
                v_t = torch.from_numpy(sv).float()
                f_t = torch.from_numpy(sf.astype("int32"))
            else:
                v_t, f_t = mesh_out.vertices.cpu(), mesh_out.faces.cpu()
            glb = o_voxel.postprocess.to_glb(
                vertices=v_t.cpu(), faces=f_t.cpu(),
                attr_volume=mesh_out.attrs.cpu(), coords=mesh_out.coords.cpu(),
                attr_layout=mesh_out.layout, voxel_size=mesh_out.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=target_faces, texture_size=tex_size, verbose=False,
            )
            job.details["bake"] = "metal"
        except RuntimeError as e:
            job.warnings.append(f"Metal bake failed ({str(e).splitlines()[0][:200]}); used the slower KDTree baker")
            glb = None

    if glb is None:
        import trimesh
        from PIL import Image
        from backends.texture_baker import bake_texture, uv_unwrap

        bv, bf = verts, faces
        if len(faces) > target_faces:
            import fast_simplification

            bv, bf = fast_simplification.simplify(verts, faces, 1.0 - target_faces / len(faces))
        nv, nf, uvs, _ = uv_unwrap(bv, bf)
        base, mr, _mask = bake_texture(
            nv, nf, uvs,
            mesh_out.coords.cpu().float().numpy(), mesh_out.attrs.cpu().float().numpy(),
            mesh_out.origin.cpu().float().numpy(), mesh_out.voxel_size,
            texture_size=tex_size,
        )
        glb = trimesh.Trimesh(vertices=nv, faces=nf, process=False)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(base), metallicFactor=0.0, roughnessFactor=0.8,
        )
        if mr is not None:
            material.metallicRoughnessTexture = Image.fromarray(mr)
        glb.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
        job.details["bake"] = "kdtree"
    job.add_time("bake", time.time() - t0)

    job.enter("export")
    t0 = time.time()
    data = glb.export(file_type="glb")
    if not data:
        raise WorkerError("export", "empty_mesh", "the GLB export came out empty", WATCHDOG_HINT)
    Path(req["out_glb"]).write_bytes(data)
    files = []
    if req.get("out_obj"):
        files = export_obj(glb, req["out_obj"])
    job.add_time("export", time.time() - t0)
    return files


def export_obj(mesh, out_obj_partial):
    """Textured OBJ: <stem>.obj, <stem>.mtl and <stem>_<texture> beside it, all written as *.partial.
    Returns the final paths of the extra files (the server renames every partial)."""
    from trimesh.exchange.obj import export_obj as trimesh_export_obj

    final = Path(out_obj_partial[: -len(".partial")] if out_obj_partial.endswith(".partial") else out_obj_partial)
    stem, folder = final.stem, final.parent
    mtl_name = f"{stem}.mtl"
    text, files = trimesh_export_obj(mesh, include_texture=True, return_texture=True, mtl_name=mtl_name)
    renamed = {}
    for name in list(files):
        if name != mtl_name:
            renamed[name] = f"{stem}_{name}"
    extras = []
    for name, data in files.items():
        if name == mtl_name:
            body = data.decode("utf-8") if isinstance(data, bytes) else data
            for old, new in renamed.items():
                body = body.replace(old, new)
            data = body.encode("utf-8")
            target = folder / mtl_name
        else:
            target = folder / renamed[name]
        Path(f"{target}.partial").write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
        extras.append(str(target))
    Path(out_obj_partial).write_text(text)
    return extras


def handle_generate(req):
    job = Job(req.get("id", ""))
    Job.current = job
    t_start = time.time()
    try:
        from PIL import Image

        emit("loading", id=job.id)
        t0 = time.time()
        fresh = _pipeline is None or _pipeline_id != req["pretrained"]
        pipeline = load(req["pretrained"])
        if fresh:
            job.add_time("load", time.time() - t0)

        job.stage = "preprocess"
        image = Image.open(req["image"])
        image.load()
        image = image.convert("RGBA")  # indexed / grey+alpha inputs keep their cut-out
        overrides = {"steps": int(req["steps"])} if req.get("steps") else {}
        job.stage = "sparse_structure"
        outputs = pipeline.run(
            image,
            seed=int(req["seed"]),
            pipeline_type=req["quality"],
            sparse_structure_sampler_params=overrides,
            shape_slat_sampler_params=overrides,
            tex_slat_sampler_params=overrides,
        )
        mesh_out = outputs[0] if isinstance(outputs, list) else outputs
        files = bake_and_export(job, mesh_out, req)
        job.add_time("total", time.time() - t_start)
        emit(
            "result", id=job.id, timings=job.timings, peak_rss_mb=peak_rss_mb(), fixes=job.fixes,
            warnings=job.warnings, details=job.details, files=files,
        )
    except WorkerError as e:
        emit("error", id=job.id, stage=e.stage, kind=e.kind, message=e.message, hint=e.hint)
    except (IndexError, AssertionError) as e:
        msg = str(e)
        if "non-zero size" in msg or "BVH needs at least 8 triangles" in msg:
            emit("error", id=job.id, stage=job.stage, kind="empty_mesh",
                 message=f"an empty tensor reached {job.stage} ({msg[:200]})", hint=WATCHDOG_HINT)
        else:
            report_exception(job, e)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            emit("error", id=job.id, stage=job.stage, kind="out_of_memory", message=str(e).splitlines()[0][:400],
                 hint="close other heavy apps (Unreal, Blender, other models), or use quality 512 and a smaller texture_size")
        else:
            report_exception(job, e)
    except Exception as e:
        report_exception(job, e)
    finally:
        Job.current = None
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


def report_exception(job, e):
    traceback.print_exc(file=sys.stderr)
    last = f"{type(e).__name__}: {e}".splitlines()[0][:400]
    emit("error", id=job.id, stage=job.stage, kind="worker_exception", message=last,
         hint="see the worker log tail below; report it if it repeats")


def main():
    trellis_dir = Path(sys.argv[1] if len(sys.argv) > 1 else os.getcwd()).resolve()
    setup(trellis_dir)
    emit("hello", protocol=PROTOCOL, pid=os.getpid(), python=sys.version.split()[0], trellis_dir=str(trellis_dir))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            emit("error", id="", stage="protocol", kind="bad_request", message=f"not JSON: {e}", hint="")
            continue
        op = msg.get("op")
        if op == "generate":
            handle_generate(msg)
        elif op == "ping":
            emit("pong", id=msg.get("id", ""))
        elif op == "unload":
            unload()
            emit("unloaded", id=msg.get("id", ""))
            break
        else:
            emit("error", id=msg.get("id", ""), stage="protocol", kind="bad_request", message=f"unknown op {op!r}", hint="")
    unload()


if __name__ == "__main__":
    main()
