//! Per-user single-instance lock for the v4l2loopback producer.
//!
//! `v4l2loopback exclusive_caps=1` already prevents two writers from
//! producing into the same device, but the gstreamer error you get when
//! that happens is cryptic. This wraps a file lock under
//! `$XDG_RUNTIME_DIR/linux-broadcast.lock` (with a per-config-dir fallback)
//! and turns it into a clean "already broadcasting" message — primarily so
//! that an autostarted headless session and a manual GUI launch don't end
//! up fighting in surprising ways.
//!
//! The lock is acquired *only* when the pipeline starts (not on app
//! launch), so the GUI itself is always free to open and let the user
//! tweak settings while another instance owns the device.

use anyhow::{Context, Result};
use directories::BaseDirs;
use fs2::FileExt;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::PathBuf;

fn lock_path() -> Result<PathBuf> {
    let base = BaseDirs::new().context("no XDG base dirs")?;
    if let Some(rt) = base.runtime_dir() {
        return Ok(rt.join("linux-broadcast.lock"));
    }
    // Fallback: per-user config dir (always writable, persistent — but
    // the lock semantics are about presence of the held fd, not file
    // contents, so persistence is harmless).
    let dir = base.config_dir().join("linux-broadcast");
    std::fs::create_dir_all(&dir).with_context(|| format!("mkdir -p {}", dir.display()))?;
    Ok(dir.join("linux-broadcast.lock"))
}

/// Holds the OS-level exclusive lock. Drop the value to release.
pub struct InstanceLock {
    _file: File,
}

impl InstanceLock {
    /// `Ok(Some(lock))` on success, `Ok(None)` if another instance already
    /// holds the lock, `Err` only on unrelated I/O errors.
    pub fn try_acquire() -> Result<Option<Self>> {
        let path = lock_path()?;
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("open {}", path.display()))?;
        match FileExt::try_lock_exclusive(&file) {
            Ok(()) => Ok(Some(Self { _file: file })),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e).with_context(|| format!("flock {}", path.display())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use tempfile::TempDir;

    fn with_temp_runtime<F: FnOnce(&std::path::Path)>(f: F) {
        let tmp = TempDir::new().unwrap();
        let prior_rt = std::env::var_os("XDG_RUNTIME_DIR");
        let prior_cfg = std::env::var_os("XDG_CONFIG_HOME");
        std::env::set_var("XDG_RUNTIME_DIR", tmp.path());
        // Also set XDG_CONFIG_HOME so the fallback path is sandboxed too.
        std::env::set_var("XDG_CONFIG_HOME", tmp.path());
        f(tmp.path());
        match prior_rt {
            Some(v) => std::env::set_var("XDG_RUNTIME_DIR", v),
            None => std::env::remove_var("XDG_RUNTIME_DIR"),
        }
        match prior_cfg {
            Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
    }

    #[test]
    #[serial]
    fn single_acquire_succeeds() {
        with_temp_runtime(|_| {
            let lock = InstanceLock::try_acquire().unwrap();
            assert!(lock.is_some(), "fresh acquire should succeed");
        });
    }

    /// flock(2) on Linux is per-OFD (open file description). Two
    /// `try_acquire` calls in the same process produce two separate
    /// OFDs, and their flock holds do NOT contend with each other —
    /// that's a kernel-level guarantee, not something our code controls.
    /// Real cross-process contention is what production cares about.
    /// We exercise that here by spawning a child via `flock(1)` from
    /// util-linux; if the binary is missing we skip the test rather
    /// than fail.
    #[test]
    #[serial]
    fn second_acquire_returns_none_while_held_by_other_process() {
        with_temp_runtime(|_| {
            let path = lock_path().unwrap();
            // Make sure the file exists so `flock(1)` opens it cleanly.
            std::fs::OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(false)
                .open(&path)
                .unwrap();

            let mut child = match std::process::Command::new("flock")
                .args(["-n", "-x"])
                .arg(&path)
                .args(["-c", "sleep 5"])
                .spawn()
            {
                Ok(c) => c,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    eprintln!("skipping: `flock(1)` not on PATH");
                    return;
                }
                Err(e) => panic!("spawn flock: {e}"),
            };
            // Give the child a moment to actually acquire the lock.
            std::thread::sleep(std::time::Duration::from_millis(150));

            let attempt = InstanceLock::try_acquire().unwrap();
            assert!(attempt.is_none(), "expected None while child holds flock");

            let _ = child.kill();
            let _ = child.wait();
        });
    }

    #[test]
    #[serial]
    fn dropped_lock_releases() {
        with_temp_runtime(|_| {
            let lock = InstanceLock::try_acquire().unwrap();
            assert!(lock.is_some());
            drop(lock);
            // After drop, a fresh acquire still works (this is trivially
            // true same-process, but the test doubles as a regression
            // guard if the impl ever changes to retain locks globally).
            let again = InstanceLock::try_acquire().unwrap();
            assert!(again.is_some());
        });
    }
}
