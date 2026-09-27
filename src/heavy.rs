//! Heavy-process census (design §6; R6): when memory is short, name what to close.

use std::collections::BTreeMap;
use std::sync::Mutex;

use serde::Serialize;

/// Processes below this are not worth naming.
const MIN_REPORT_MB: u64 = 200;
/// With no configured match, name this many of the largest processes instead.
const FALLBACK_COUNT: usize = 3;
const FALLBACK_MIN_MB: u64 = 1_000;

#[derive(Debug, Clone, PartialEq)]
pub struct ProcessInfo {
    pub name: String,
    /// Physical footprint on macOS (GPU allocations included), resident size elsewhere.
    pub resident_mb: u64,
}

pub trait ProcessLister: Send + Sync {
    fn processes(&self) -> Vec<ProcessInfo>;
}

/// Real lister backed by `sysinfo`.
pub struct SystemProcesses {
    sys: Mutex<sysinfo::System>,
}

impl Default for SystemProcesses {
    fn default() -> Self {
        Self {
            sys: Mutex::new(sysinfo::System::new()),
        }
    }
}

impl ProcessLister for SystemProcesses {
    fn processes(&self) -> Vec<ProcessInfo> {
        use sysinfo::{ProcessRefreshKind, ProcessesToUpdate};
        let mut sys = self.sys.lock().unwrap_or_else(|e| e.into_inner());
        sys.refresh_processes_specifics(
            ProcessesToUpdate::All,
            true,
            ProcessRefreshKind::nothing().with_memory(),
        );
        let own = std::process::id();
        sys.processes()
            .iter()
            .filter(|(pid, _)| pid.as_u32() != own)
            .map(|(pid, p)| ProcessInfo {
                name: p.name().to_string_lossy().into_owned(),
                resident_mb: footprint_bytes(pid.as_u32()).unwrap_or_else(|| p.memory())
                    / (1024 * 1024),
            })
            .collect()
    }
}

/// macOS: the physical footprint (Activity Monitor's "Memory"), which counts GPU/Metal allocations that resident size
/// misses on unified memory: Unreal Editor measured 9958 MB footprint against 1627 MB resident. `None` where it can't be
/// read (another user's process) or on other platforms, so the caller falls back to resident size.
#[cfg(target_os = "macos")]
fn footprint_bytes(pid: u32) -> Option<u64> {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v2>::zeroed();
    // SAFETY: `info` is a properly sized, writable rusage_info_v2 for flavor RUSAGE_INFO_V2.
    let rc = unsafe {
        libc::proc_pid_rusage(
            i32::try_from(pid).ok()?,
            libc::RUSAGE_INFO_V2,
            info.as_mut_ptr().cast::<libc::rusage_info_t>(),
        )
    };
    // SAFETY: proc_pid_rusage filled the struct when it returned 0.
    (rc == 0).then(|| unsafe { info.assume_init() }.ri_phys_footprint)
}

#[cfg(not(target_os = "macos"))]
fn footprint_bytes(_pid: u32) -> Option<u64> {
    None
}

/// Fixed list, for tests.
pub struct FixedProcesses(pub Vec<ProcessInfo>);

impl ProcessLister for FixedProcesses {
    fn processes(&self) -> Vec<ProcessInfo> {
        self.0.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HeavyProcess {
    /// The configured name that matched (all its processes summed), or the process name in the fallback.
    pub name: String,
    pub resident_mb: u64,
    pub count: usize,
}

/// Processes matching `patterns` (case-insensitive substring), summed per pattern, largest first. With no match,
/// the few largest processes over 1 GB, so the message still points somewhere.
pub fn census(processes: &[ProcessInfo], patterns: &[String]) -> Vec<HeavyProcess> {
    let mut matched: BTreeMap<&str, (u64, usize)> = BTreeMap::new();
    for p in processes {
        let name = p.name.to_lowercase();
        if let Some(pattern) = patterns
            .iter()
            .find(|pat| !pat.is_empty() && name.contains(&pat.to_lowercase()))
        {
            let entry = matched.entry(pattern.as_str()).or_default();
            entry.0 += p.resident_mb;
            entry.1 += 1;
        }
    }
    let mut heavy: Vec<HeavyProcess> = matched
        .into_iter()
        .filter(|(_, (mb, _))| *mb >= MIN_REPORT_MB)
        .map(|(name, (resident_mb, count))| HeavyProcess {
            name: name.to_string(),
            resident_mb,
            count,
        })
        .collect();
    if heavy.is_empty() {
        heavy = processes
            .iter()
            .filter(|p| p.resident_mb >= FALLBACK_MIN_MB)
            .map(|p| HeavyProcess {
                name: p.name.clone(),
                resident_mb: p.resident_mb,
                count: 1,
            })
            .collect();
        heavy.sort_by_key(|h| std::cmp::Reverse(h.resident_mb));
        heavy.truncate(FALLBACK_COUNT);
    } else {
        heavy.sort_by_key(|h| std::cmp::Reverse(h.resident_mb));
    }
    heavy
}

/// "Close UnrealEditor (6100 MB), ComfyUI (4800 MB)." or an empty string when nothing is worth naming.
pub fn advice(heavy: &[HeavyProcess]) -> String {
    if heavy.is_empty() {
        return String::new();
    }
    let list: Vec<String> = heavy
        .iter()
        .map(|h| {
            if h.count > 1 {
                format!("{} ({} MB in {} processes)", h.name, h.resident_mb, h.count)
            } else {
                format!("{} ({} MB)", h.name, h.resident_mb)
            }
        })
        .collect();
    format!("Close {}.", list.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(name: &str, mb: u64) -> ProcessInfo {
        ProcessInfo {
            name: name.into(),
            resident_mb: mb,
        }
    }

    fn patterns() -> Vec<String> {
        [
            "UnrealEditor",
            "Blender",
            "ComfyUI",
            "sd-server",
            "mcp-imagegen",
        ]
        .map(String::from)
        .to_vec()
    }

    #[test]
    fn matches_sums_and_sorts() {
        let procs = [
            p("UnrealEditor", 6100),
            p("UnrealEditor-Cmd", 900),
            p("comfyui-backend", 4800),
            p("Blender", 150),
            p("Safari", 3000),
            p("mcp-imagegen", 40),
        ];
        let heavy = census(&procs, &patterns());
        assert_eq!(
            heavy,
            vec![
                HeavyProcess {
                    name: "UnrealEditor".into(),
                    resident_mb: 7000,
                    count: 2
                },
                HeavyProcess {
                    name: "ComfyUI".into(),
                    resident_mb: 4800,
                    count: 1
                },
            ]
        );
        assert_eq!(
            advice(&heavy),
            "Close UnrealEditor (7000 MB in 2 processes), ComfyUI (4800 MB)."
        );
    }

    #[test]
    fn falls_back_to_the_largest_processes() {
        let procs = [
            p("Safari", 3000),
            p("Xcode", 5000),
            p("Finder", 90),
            p("Mail", 1200),
            p("Slack", 1100),
        ];
        let heavy = census(&procs, &patterns());
        let names: Vec<&str> = heavy.iter().map(|h| h.name.as_str()).collect();
        assert_eq!(names, ["Xcode", "Safari", "Mail"]);
    }

    #[test]
    fn nothing_heavy_means_no_advice() {
        let heavy = census(&[p("Finder", 90)], &patterns());
        assert!(heavy.is_empty());
        assert_eq!(advice(&heavy), "");
    }

    #[test]
    fn empty_patterns_never_match_everything() {
        let heavy = census(&[p("Safari", 300)], &[String::new()]);
        assert!(heavy.is_empty(), "{heavy:?}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn footprint_is_readable_for_our_own_process() {
        let bytes = footprint_bytes(std::process::id()).expect("own footprint");
        assert!(bytes > 0);
    }

    #[test]
    fn system_lister_sees_processes() {
        let procs = SystemProcesses::default().processes();
        assert!(!procs.is_empty());
        assert!(procs.iter().any(|p| p.resident_mb > 0));
    }
}
