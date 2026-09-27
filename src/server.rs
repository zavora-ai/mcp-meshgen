//! MCP surface. Bootstrap stage: `list_models` answers with an empty list until the registry lands (task 3)
//! and the real tools arrive (tasks 11-13).

use adk_mcp_sdk::{HealthCheck, HealthStatus};
use rmcp::{tool, tool_router};

#[derive(Clone, Default)]
pub struct MeshGenServer;

#[tool_router]
impl MeshGenServer {
    /// List image-to-3D models with their status, licences and memory needs. Never loads weights.
    #[tool(
        description = "List image-to-3D models with their status, licences and memory needs; never loads weights"
    )]
    async fn list_models(&self) -> String {
        serde_json::json!({ "models": [], "note": "mcp-meshgen bootstrap: no models configured yet" }).to_string()
    }
}

adk_mcp_sdk::mcp_2026_server! {
    server: MeshGenServer,
    task_tools: [],
    task_ttl_overrides: [],
    approval_tools: [],
    mutating_tools: [],
    destructive_tools: [],
    idempotent_tools: [],
    cache_ttl_ms: 60_000,
    instructions: "Image-to-3D conversion with local models. Call list_models first to see which models are ready and what \
their licences allow. Inputs must already have a transparent background (cut them out with an image editor first).",
}

#[async_trait::async_trait]
impl HealthCheck for MeshGenServer {
    async fn check_health(&self) -> HealthStatus {
        HealthStatus {
            healthy: true,
            message: Some("bootstrap: no backends yet".into()),
            latency_ms: Some(0),
        }
    }
}
