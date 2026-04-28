//! Detect which processes are reading our `/dev/video10` virtual camera.
//!
//! Strategy: walk `/proc/<pid>/fd/*`, follow each symlink, and count those
//! whose target equals our sink device path. This is what `lsof` and
//! `fuser` do internally — robust, portable, no new dependencies, and
//! cheap (~1–3 ms per poll on a typical system).
//!
//! Why poll instead of an fsnotify-style hook: the kernel does not fire
//! `inotify`/`fanotify` events on character-device opens, and the
//! v4l2loopback driver does not expose a sysfs "consumer count" attribute
//! (some forks do, but mainline does not). Userspace polling is the only
//! portable signal.
//!
//! The watcher runs on a background thread and emits a snapshot on a
//! crossbeam channel whenever the consumer set changes. Identical
//! snapshots are deduplicated upstream.

use crossbeam_channel::{Receiver, Sender};
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// A single process that has `/dev/video10` open. Equality + hashing is
/// by PID alone; the friendly name is best-effort metadata for the GUI.
#[derive(Debug, Clone)]
pub struct Consumer {
    pub pid: u32,
    pub name: String,
}

impl PartialEq for Consumer {
    fn eq(&self, other: &Self) -> bool {
        self.pid == other.pid
    }
}
impl Eq for Consumer {}
impl std::hash::Hash for Consumer {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.pid.hash(h);
    }
}

/// Walk `/proc/*/fd/*` once and return every PID (other than `exclude`)
/// that has `target_device` open.
pub fn current_consumers(target_device: &str, exclude_pid: u32) -> Vec<Consumer> {
    let target = PathBuf::from(target_device);
    let mut out = Vec::new();

    let proc_dir = match fs::read_dir("/proc") {
        Ok(d) => d,
        Err(_) => return out,
    };

    for entry in proc_dir.flatten() {
        // Only numeric PIDs.
        let name = entry.file_name();
        let pid_str = match name.to_str() {
            Some(s) => s,
            None => continue,
        };
        let pid: u32 = match pid_str.parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
        if pid == exclude_pid {
            continue;
        }

        let fd_dir = entry.path().join("fd");
        let fds = match fs::read_dir(&fd_dir) {
            Ok(d) => d,
            // EACCES on root-owned PIDs we can't see is normal.
            Err(_) => continue,
        };
        let mut matched = false;
        for fd in fds.flatten() {
            // readlink — non-existent / racy entries are silently skipped.
            if let Ok(link) = fs::read_link(fd.path()) {
                if link == target {
                    matched = true;
                    break;
                }
            }
        }
        if matched {
            out.push(Consumer {
                pid,
                name: read_comm(pid).unwrap_or_else(|| format!("pid-{pid}")),
            });
        }
    }

    out
}

/// `/proc/<pid>/comm` is the kernel-supplied process name (15 chars max,
/// no path). Cheap and stable enough for a status footer.
fn read_comm(pid: u32) -> Option<String> {
    fs::read_to_string(format!("/proc/{pid}/comm"))
        .ok()
        .map(|s| s.trim_end().to_string())
        .filter(|s| !s.is_empty())
}

/// Background poller. Drop the watcher to stop the thread.
pub struct Watcher {
    rx: Receiver<Vec<Consumer>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Watcher {
    /// Start polling `target_device` every `interval`, sending fresh
    /// snapshots on the returned receiver whenever the consumer set
    /// changes (PID set comparison; process renames don't fire events).
    pub fn start(target_device: String, exclude_pid: u32, interval: Duration) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded::<Vec<Consumer>>();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop);

        let handle = thread::Builder::new()
            .name("lb-consumer-watch".into())
            .spawn(move || run(target_device, exclude_pid, interval, tx, stop_for_thread))
            .expect("spawn lb-consumer-watch");

        Self {
            rx,
            stop,
            handle: Some(handle),
        }
    }

    /// Receiver of consumer-set snapshots. The first message lands ~one
    /// poll interval after start.
    pub fn events(&self) -> &Receiver<Vec<Consumer>> {
        &self.rx
    }
}

impl Drop for Watcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn run(
    target: String,
    exclude_pid: u32,
    interval: Duration,
    tx: Sender<Vec<Consumer>>,
    stop: Arc<AtomicBool>,
) {
    let mut last: HashSet<u32> = HashSet::new();
    let mut sent_initial = false;
    while !stop.load(Ordering::Relaxed) {
        let started = Instant::now();
        let now = current_consumers(&target, exclude_pid);
        let now_set: HashSet<u32> = now.iter().map(|c| c.pid).collect();
        if !sent_initial || now_set != last {
            sent_initial = true;
            if tx.send(now.clone()).is_err() {
                return; // receiver dropped
            }
            last = now_set;
        }
        // Sleep the remainder of the interval (poll cost is variable).
        let elapsed = started.elapsed();
        if elapsed < interval {
            thread::sleep(interval - elapsed);
        }
    }
}
