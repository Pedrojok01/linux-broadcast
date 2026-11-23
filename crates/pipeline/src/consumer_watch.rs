//! Detect which processes are reading our `/dev/video10` virtual camera.
//!
//! The consumer list comes from walking `/proc/<pid>/fd/*` and matching each
//! symlink target against the sink device path, as `lsof` and `fuser` do.
//! v4l2loopback exposes no reader count, so this walk is the only portable
//! way to name the readers.
//!
//! The walk costs one `readlink` per open fd on the whole system, and some
//! daemons hold hundreds of thousands (Docker Desktop's `virtiofsd`: ~180k,
//! ~0.5 s per walk). So the thread sleeps on inotify open/close events on
//! the device node and walks only when one fires, plus a slow rescan for
//! opens inotify can't see. Without inotify it falls back to polling.
//!
//! Snapshots go out on a crossbeam channel only when the PID set changes.

use crossbeam_channel::{Receiver, Sender};
use rustix::event::{EventfdFlags, PollFd, PollFlags, Timespec, eventfd, poll};
use rustix::fd::OwnedFd;
use rustix::fs::inotify;
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Full rescan cadence while inotify works. Only catches opens inotify can't
/// see: through another inode of the device, e.g. a container's own `/dev`.
const RESCAN_INTERVAL: Duration = Duration::from_secs(30);

/// Rescan cadence without inotify. The full interval is slept after each
/// walk, so a slow walk can't run back-to-back.
const POLL_INTERVAL: Duration = Duration::from_millis(800);

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
            if let Ok(link) = fs::read_link(fd.path())
                && link == target
            {
                matched = true;
                break;
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

/// Background watcher thread. Drop the watcher to stop it.
pub struct Watcher {
    rx: Receiver<Vec<Consumer>>,
    /// eventfd that wakes the thread to exit.
    stop: Arc<OwnedFd>,
    handle: Option<JoinHandle<()>>,
}

impl Watcher {
    /// Start watching `target_device`, sending a snapshot on the returned
    /// receiver whenever the consumer set changes (PID set comparison;
    /// process renames don't fire events).
    pub fn start(target_device: String, exclude_pid: u32) -> io::Result<Self> {
        let stop = Arc::new(eventfd(0, EventfdFlags::CLOEXEC)?);
        let (tx, rx) = crossbeam_channel::unbounded::<Vec<Consumer>>();
        let thread_stop = Arc::clone(&stop);
        let handle = thread::Builder::new()
            .name("lb-consumer-watch".into())
            .spawn(move || run(&target_device, exclude_pid, &tx, &thread_stop))?;
        Ok(Self {
            rx,
            stop,
            handle: Some(handle),
        })
    }

    /// Receiver of consumer-set snapshots. The first message lands as soon
    /// as the first walk finishes.
    pub fn events(&self) -> &Receiver<Vec<Consumer>> {
        &self.rx
    }
}

impl Drop for Watcher {
    fn drop(&mut self) {
        let _ = rustix::io::write(&*self.stop, &1u64.to_ne_bytes());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn run(target: &str, exclude_pid: u32, tx: &Sender<Vec<Consumer>>, stop: &OwnedFd) {
    // Watch before the first walk, so an open racing it still triggers a rescan.
    let opens = watch_opens(target)
        .inspect_err(|e| log::warn!("no inotify on {target} ({e}); polling instead"))
        .ok();
    let rescan_after = if opens.is_some() {
        RESCAN_INTERVAL
    } else {
        POLL_INTERVAL
    };
    let mut last: Option<HashSet<u32>> = None;
    loop {
        let now = current_consumers(target, exclude_pid);
        let pids: HashSet<u32> = now.iter().map(|c| c.pid).collect();
        if last.as_ref() != Some(&pids) {
            if tx.send(now).is_err() {
                return; // receiver dropped
            }
            last = Some(pids);
        }
        if wait(opens.as_ref(), stop, rescan_after) == Wake::Stop {
            return;
        }
    }
}

/// Non-blocking inotify fd that turns readable whenever any process opens
/// or closes `target`.
fn watch_opens(target: &str) -> io::Result<OwnedFd> {
    let fd = inotify::init(inotify::CreateFlags::CLOEXEC | inotify::CreateFlags::NONBLOCK)?;
    let events = inotify::WatchFlags::OPEN
        | inotify::WatchFlags::CLOSE_WRITE
        | inotify::WatchFlags::CLOSE_NOWRITE;
    inotify::add_watch(&fd, target, events)?;
    Ok(fd)
}

#[derive(Debug, PartialEq, Eq)]
enum Wake {
    Stop,
    DeviceActivity,
    Timeout,
}

/// Block until `stop` is signalled, the device is opened or closed, or
/// `timeout` passes. Device events are drained, so a burst of opens costs
/// one walk.
fn wait(opens: Option<&OwnedFd>, stop: &OwnedFd, timeout: Duration) -> Wake {
    let mut fds = vec![PollFd::new(stop, PollFlags::IN)];
    if let Some(fd) = opens {
        fds.push(PollFd::new(fd, PollFlags::IN));
    }
    let timeout = Timespec::try_from(timeout).ok();
    // EINTR (the only realistic error) just means "walk again".
    if poll(&mut fds, timeout.as_ref()).unwrap_or(0) == 0 {
        return Wake::Timeout;
    }
    if fds[0].revents().contains(PollFlags::IN) {
        return Wake::Stop;
    }
    if let Some(fd) = opens {
        let mut buf = [0u8; 4096];
        while matches!(rustix::io::read(fd, &mut buf[..]), Ok(n) if n > 0) {}
    }
    Wake::DeviceActivity
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use std::time::Instant;

    fn wait_for(
        rx: &Receiver<Vec<Consumer>>,
        timeout: Duration,
        pred: impl Fn(&[Consumer]) -> bool,
    ) -> bool {
        let deadline = Instant::now() + timeout;
        while let Some(left) = deadline.checked_duration_since(Instant::now()) {
            match rx.recv_timeout(left) {
                Ok(snapshot) if pred(&snapshot) => return true,
                Ok(_) => {}
                Err(_) => return false,
            }
        }
        false
    }

    /// An open and the matching close are both reported within seconds,
    /// far below `RESCAN_INTERVAL`, so inotify is what drives detection.
    #[test]
    fn reports_open_and_close_without_waiting_for_rescan() {
        let dir = tempfile::tempdir().unwrap();
        let device = dir.path().canonicalize().unwrap().join("video");
        fs::write(&device, b"").unwrap();
        let target = device.to_str().unwrap().to_string();
        let watcher = Watcher::start(target.clone(), std::process::id()).unwrap();
        let rx = watcher.events();
        assert!(wait_for(rx, Duration::from_secs(5), <[_]>::is_empty));

        // The shell opens the file on fd 3, then execs `sleep` in place.
        let mut child = Command::new("sh")
            .args(["-c", "exec 3<\"$0\"; exec sleep 30"])
            .arg(&target)
            .spawn()
            .unwrap();
        let pid = child.id();
        let opened = wait_for(rx, Duration::from_secs(5), |s| {
            s.iter().any(|c| c.pid == pid)
        });
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(opened, "open not reported");
        assert!(
            wait_for(rx, Duration::from_secs(5), <[_]>::is_empty),
            "close not reported"
        );
    }

    #[test]
    fn wait_times_out_when_nothing_happens() {
        let dir = tempfile::tempdir().unwrap();
        let device = dir.path().join("video");
        fs::write(&device, b"").unwrap();
        let opens = watch_opens(device.to_str().unwrap()).unwrap();
        let stop = eventfd(0, EventfdFlags::CLOEXEC).unwrap();
        let t = Instant::now();
        assert_eq!(
            wait(Some(&opens), &stop, Duration::from_millis(200)),
            Wake::Timeout
        );
        assert!(t.elapsed() >= Duration::from_millis(200));
    }

    #[test]
    fn drop_stops_the_thread_promptly() {
        let watcher = Watcher::start("/nonexistent/video".into(), std::process::id()).unwrap();
        assert!(wait_for(watcher.events(), Duration::from_secs(5), |_| true));
        let t = Instant::now();
        drop(watcher);
        assert!(
            t.elapsed() < Duration::from_millis(500),
            "took {:?}",
            t.elapsed()
        );
    }
}
