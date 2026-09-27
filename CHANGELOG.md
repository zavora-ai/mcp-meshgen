# Changelog

## 0.1.0 (unreleased)

First version: image-to-3D over MCP with TRELLIS.2 on Apple Silicon.

- Tools: `generate_mesh` (MCP Task on 2026-07-28 clients, `wait: false` + `get_job` for others), `get_job`, `cancel_job`,
  `list_models`, `unload_models`.
- `trellis_mac` backend: a Python worker embedded in the server runs inside trellis-mac's venv over JSON lines, with stage
  progress, cancel (process-group kill), keep-alive with idle unload, and explained failures (`empty_structure`,
  `empty_mesh`, `gpu_watchdog`, `out_of_memory`, `missing_weights`, `worker_exited` / `worker_killed`).
- Mac fix applied at load: the sparse-structure decoder runs on the CPU under MPS (`mps_sparse_decoder_cpu`). TRELLIS's
  background remover (RMBG-2.0, non-commercial) is stubbed out; inputs must be cut-outs, checked before anything loads.
- Model registry (`models.toml`) with per-model status: `ready`, `missing_files` (exact `hf download` command and size),
  `needs_licence_acceptance` (page to accept), `backend_unavailable` (why and how to fix); licences of every component.
- Machine safety: one local lane, a cross-process GPU lock shared with sister servers, a free-memory preflight that names
  heavy processes (physical footprint on macOS), and the worker exits after each job by default.
- Output: GLB (+ optional textured OBJ), all paths jailed to allowed roots, `*.partial` files renamed on success and deleted
  on cancel or failure, never overwriting; a sidecar with parameters, input SHA-256, mesh stats read from the GLB, timings,
  peak memory, fixes and licences.
- `target_triangles` applied before the texture bake; headless-Blender previews (four views) and an engine copy (`size_m`,
  origin at the bottom centre, Z up, FBX with settings verified for Unreal Engine 5.8).
- Config discovery (`$MCP_MESHGEN_CONFIG_DIR` or the platform config dir), relative paths anchored at the config dir's
  parent, Blender discovery on macOS, Windows and Linux.
