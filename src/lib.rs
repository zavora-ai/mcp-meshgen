//! mcp-meshgen — local image-to-3D conversion over MCP with pluggable backends.
//!
//! See the README for what it does, how to set it up and how it protects a shared machine.

pub mod backend;
pub mod config;
pub mod error;
pub mod glb;
pub mod gpu_lock;
pub mod heavy;
pub mod input;
pub mod jobs;
pub mod memory;
pub mod output;
pub mod post;
pub mod registry;
pub mod request;
pub mod server;

/// The registry manifest, embedded so the binary works from any working directory.
pub const MANIFEST_TOML: &str = include_str!("../mcp-server.toml");

#[cfg(test)]
mod tests {
    use super::MANIFEST_TOML;

    /// The README's download commands are the ones `list_models` prints for an empty cache.
    #[test]
    fn readme_download_commands_match_the_registry() {
        let readme = include_str!("../README.md");
        let registry =
            crate::registry::ModelRegistry::from_toml(crate::config::MODELS_EXAMPLE).unwrap();
        let empty = tempfile::tempdir().unwrap();
        for model in registry.models() {
            for missing in model.missing_files(empty.path(), empty.path()) {
                assert!(
                    readme.contains(&missing.command),
                    "README lacks: {}",
                    missing.command
                );
            }
        }
    }

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
