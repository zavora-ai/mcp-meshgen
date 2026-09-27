//! Cross-process lock for local GPU jobs (design §6; A2).
//!
//! An exclusive advisory lock on `<cache>/zavora-mcp/local-gpu.lock` (std `File::try_lock`: `flock` on Unix,
//! `LockFileEx` on Windows) held for a whole local job, so two servers never load big models at once. The OS drops
//! the lock when the holder exits, even on a crash. Who holds it is written to `local-gpu.owner` beside it (best
//! effort: Windows won't let other processes read a locked file).

use std::fs::{File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::error::MeshGenError;

const POLL: Duration = Duration::from_millis(500);

/// Who holds the lock, for "waiting for …" messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Holder {
    pub server: String,
    pub pid: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job: Option<String>,
    pub since_unix: u64,
}

impl Holder {
    pub fn this_process(job: Option<&str>) -> Self {
        Self {
            server: env!("CARGO_PKG_NAME").to_string(),
            pid: std::process::id(),
            job: job.map(str::to_string),
            since_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// "mcp-imagegen (pid 4242)"
    pub fn describe(&self) -> String {
        format!("{} (pid {})", self.server, self.pid)
    }
}

#[derive(Debug, Clone)]
pub struct GpuLock {
    path: PathBuf,
}

/// Held lock; released (and the owner file removed) on drop.
#[derive(Debug)]
pub struct GpuLockGuard {
    file: File,
    owner_path: PathBuf,
}

impl Drop for GpuLockGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.owner_path);
        let _ = self.file.unlock();
    }
}

impl GpuLock {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn owner_path(&self) -> PathBuf {
        self.path.with_extension("owner")
    }

    /// Take the lock if it's free.
    pub fn try_acquire(&self, holder: &Holder) -> Result<Option<GpuLockGuard>, MeshGenError> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| MeshGenError::io(dir, e))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&self.path)
            .map_err(|e| MeshGenError::io(&self.path, e))?;
        match file.try_lock() {
            Ok(()) => {
                let owner_path = self.owner_path();
                if let Ok(json) = serde_json::to_vec(holder) {
                    let _ = std::fs::write(&owner_path, json);
                }
                Ok(Some(GpuLockGuard { file, owner_path }))
            }
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Error(e)) => Err(MeshGenError::io(&self.path, e)),
        }
    }

    /// Wait for the lock, polling, until it's free or `cancel` fires. `on_wait` is called once, when the first
    /// attempt finds it taken, with the current holder if known.
    pub async fn acquire(
        &self,
        holder: &Holder,
        cancel: &CancellationToken,
        on_wait: impl FnOnce(Option<Holder>),
    ) -> Result<GpuLockGuard, MeshGenError> {
        let mut on_wait = Some(on_wait);
        loop {
            if let Some(guard) = self.try_acquire(holder)? {
                return Ok(guard);
            }
            if let Some(notify) = on_wait.take() {
                notify(self.current_holder());
            }
            tokio::select! {
                _ = cancel.cancelled() => return Err(MeshGenError::Cancelled),
                _ = tokio::time::sleep(POLL) => {}
            }
        }
    }

    /// The holder recorded in the owner file, if any (may be stale when the lock is actually free).
    pub fn current_holder(&self) -> Option<Holder> {
        let text = std::fs::read(self.owner_path()).ok()?;
        serde_json::from_slice(&text).ok()
    }
}

/// "waiting for another local model (mcp-imagegen (pid 4242))"
pub fn waiting_message(holder: Option<&Holder>) -> String {
    waiting_message_for(holder.map(Holder::describe).as_deref())
}

/// Same, from an already-described holder.
pub fn waiting_message_for(holder: Option<&str>) -> String {
    match holder {
        Some(h) => format!("waiting for another local model ({h})"),
        None => "waiting for another local model".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    fn lock_in(dir: &Path) -> GpuLock {
        GpuLock::new(dir.join("zavora-mcp").join("local-gpu.lock"))
    }

    #[test]
    fn second_handle_is_blocked_until_release() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (lock_in(dir.path()), lock_in(dir.path()));
        let me = Holder::this_process(Some("job-1"));
        let guard = a.try_acquire(&me).unwrap().expect("free lock");
        assert!(b.try_acquire(&me).unwrap().is_none(), "held by a");
        assert_eq!(b.current_holder(), Some(me.clone()));
        drop(guard);
        assert!(
            b.current_holder().is_none(),
            "owner file removed on release"
        );
        assert!(b.try_acquire(&me).unwrap().is_some());
    }

    #[tokio::test]
    async fn acquire_waits_then_proceeds() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (lock_in(dir.path()), lock_in(dir.path()));
        let other = Holder {
            server: "mcp-imagegen".into(),
            pid: 4242,
            job: None,
            since_unix: 0,
        };
        let guard = a.try_acquire(&other).unwrap().unwrap();
        let seen = Arc::new(Mutex::new(None));
        let seen2 = seen.clone();
        let waiter = tokio::spawn(async move {
            b.acquire(
                &Holder::this_process(None),
                &CancellationToken::new(),
                move |h| {
                    *seen2.lock().unwrap() = Some(waiting_message(h.as_ref()));
                },
            )
            .await
            .map(|_| ())
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!waiter.is_finished());
        assert_eq!(
            seen.lock().unwrap().as_deref(),
            Some("waiting for another local model (mcp-imagegen (pid 4242))")
        );
        drop(guard);
        tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .expect("acquired after release")
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn waiting_can_be_cancelled() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (lock_in(dir.path()), lock_in(dir.path()));
        let me = Holder::this_process(None);
        let _guard = a.try_acquire(&me).unwrap().unwrap();
        let cancel = CancellationToken::new();
        let c = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            c.cancel();
        });
        let err = b.acquire(&me, &cancel, |_| {}).await.unwrap_err();
        assert_eq!(err.code(), "cancelled");
    }

    #[test]
    fn waiting_message_without_a_known_holder() {
        assert_eq!(waiting_message(None), "waiting for another local model");
    }
}
