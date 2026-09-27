# mcp-meshgen

Image-to-3D conversion over MCP with pluggable backends. The first backend runs Microsoft's TRELLIS.2 locally on Apple Silicon
through [trellis-mac](https://github.com/shivampkumar/trellis-mac).

Status: **in development** (v0.1.0). Design and tasks: `specs/mesh-gen-mcp/` in the games workspace.

## Verify

```bash
cargo test --all-features
```
