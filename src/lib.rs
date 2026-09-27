//! mcp-meshgen — local image-to-3D conversion over MCP with pluggable backends.
//!
//! See `specs/mesh-gen-mcp/design.md` in the games workspace for the design.

pub mod server;

/// The registry manifest, embedded so the binary works from any working directory.
pub const MANIFEST_TOML: &str = include_str!("../mcp-server.toml");

#[cfg(test)]
mod tests {
    use super::MANIFEST_TOML;

    #[test]
    fn manifest_is_valid() {
        let manifest =
            adk_mcp_sdk::ServerManifest::from_toml(MANIFEST_TOML).expect("manifest parses");
        assert_eq!(
            manifest.version,
            env!("CARGO_PKG_VERSION"),
            "manifest version matches the crate"
        );
        assert!(
            manifest.validate().is_empty(),
            "manifest errors: {:?}",
            manifest.validate()
        );
    }
}
