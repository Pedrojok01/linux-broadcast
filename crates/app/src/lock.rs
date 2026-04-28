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
