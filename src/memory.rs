//! Memory preflight (design §6; R6), from mcp-imagegen: refuse to load a model that would push the machine into
//! swap, and say which heavy processes to close.

use std::sync::Mutex;

use crate::error::MeshGenError;
use crate::registry::ModelSpec;

pub trait MemoryProbe: Send + Sync {
    /// Memory the OS could hand out now, in MB (free + reclaimable caches).
    fn available_mb(&self) -> u64;
}

/// Real probe backed by `sysinfo`.
pub struct SystemMemory {
    sys: Mutex<sysinfo::System>,
}

impl Default for SystemMemory {
    fn default() -> Self {
        Self {
            sys: Mutex::new(sysinfo::System::new()),
        }
    }
}

impl MemoryProbe for SystemMemory {
    fn available_mb(&self) -> u64 {
        let mut sys = self.sys.lock().unwrap_or_else(|e| e.into_inner());
        sys.refresh_memory();
        // sysinfo's available_memory() reads 0 on macOS under compression; total - used
        // matches vm_stat's free + inactive + purgeable + speculative.
        let bytes = if cfg!(target_os = "macos") || sys.available_memory() == 0 {
            sys.total_memory().saturating_sub(sys.used_memory())
        } else {
            sys.available_memory()
        };
        bytes / (1024 * 1024)
    }
}

/// Fixed value, for tests.
pub struct FixedMemory(pub u64);

impl MemoryProbe for FixedMemory {
    fn available_mb(&self) -> u64 {
        self.0
    }
}

/// Check that `model` fits with `headroom_mb` to spare. `advice` is only called on refusal: it names the heavy
/// processes to close (from the census) or returns an empty string.
pub fn preflight(
    model: &ModelSpec,
    headroom_mb: u64,
    probe: &dyn MemoryProbe,
    advice: impl FnOnce() -> String,
) -> Result<(), MeshGenError> {
    let available_mb = probe.available_mb();
    let need_mb = model.memory_mb + headroom_mb;
    if available_mb >= need_mb {
        return Ok(());
    }
    let census = advice();
    let advice = if census.is_empty() {
        format!("Close other heavy apps, or lower memory_headroom_mb (now {headroom_mb} MB).")
    } else {
        format!("{census} Or lower memory_headroom_mb (now {headroom_mb} MB).")
    };
    Err(MeshGenError::InsufficientMemory {
        model: model.id.clone(),
        need_mb,
        available_mb,
        advice,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MODELS_EXAMPLE;
    use crate::registry::ModelRegistry;

    fn trellis() -> ModelSpec {
        ModelRegistry::from_toml(MODELS_EXAMPLE)
            .unwrap()
            .get("trellis2-4b")
            .unwrap()
            .clone()
    }

    #[test]
    fn passes_with_enough_memory() {
        // 15000 needed + 2000 headroom
        let called = std::cell::Cell::new(false);
        let ok = preflight(&trellis(), 2000, &FixedMemory(17_000), || {
            called.set(true);
            String::new()
        });
        assert!(ok.is_ok());
        assert!(!called.get(), "the census only runs on refusal");
    }

    #[test]
    fn refusal_names_what_to_close() {
        let err = preflight(&trellis(), 2000, &FixedMemory(9_000), || {
            "Close Unreal Editor (6100 MB), ComfyUI (4800 MB).".into()
        })
        .unwrap_err();
        assert_eq!(err.code(), "insufficient_memory");
        let msg = err.to_string();
        assert!(msg.contains("~17000 MB"), "{msg}");
        assert!(msg.contains("9000 MB available"), "{msg}");
        assert!(
            msg.contains("Close Unreal Editor (6100 MB), ComfyUI (4800 MB)."),
            "{msg}"
        );
        assert!(msg.contains("memory_headroom_mb (now 2000 MB)"), "{msg}");
    }

    #[test]
    fn refusal_without_census_still_says_what_to_do() {
        let err = preflight(&trellis(), 500, &FixedMemory(100), String::new).unwrap_err();
        assert!(err.to_string().contains("Close other heavy apps"), "{err}");
    }

    #[test]
    fn system_probe_reports_something() {
        assert!(SystemMemory::default().available_mb() > 0);
    }
}
