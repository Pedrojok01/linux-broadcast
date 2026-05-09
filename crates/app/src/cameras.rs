//! Enumerate physical webcam nodes for the camera dropdown.
//!
//! We list `/dev/video*` and read each device's friendly name from
//! `/sys/class/video4linux/<n>/name`. That's a non-disruptive probe
//! (no `open(2)`, no LED) and avoids fighting another process for
//! exclusive access while the user is just *picking* a camera.
//!
//! Limitation: the kernel exposes every numbered subdevice a webcam
//! claims (capture, metadata, control), so a single physical webcam
//! often shows up as several entries. Filtering to capture-only would
//! require a `VIDIOC_QUERYCAP` ioctl on each node, which means opening
//! the device — and that *will* fight other consumers and may flicker
//! the LED. We've judged the extra entries acceptable; the user can
//! pick the right one and we persist their choice.
//!
//! `sink_device` is excluded so the virtual cam (`/dev/video10`)
//! never appears as a selectable input.

use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraEntry {
    pub path: String,
    pub label: String,
}

/// Enumerate `/dev/video*` capture-capable nodes, excluding `sink_device`.
///
/// We probe each `/dev/videoN` for a friendly name via
/// `/sys/class/video4linux/videoN/name` rather than opening the device —
/// cheaper and non-disruptive. The same physical webcam often exposes
/// several numbered nodes (capture, metadata, control) and they all show
/// up here; filtering to capture-only would require a v4l2 ioctl probe.
pub fn enumerate(sink_device: &str) -> Vec<CameraEntry> {
    let mut out = Vec::new();
    let glob = match glob::glob("/dev/video*") {
        Ok(g) => g,
        Err(_) => return out,
    };
    for entry in glob.flatten() {
        let path = entry.to_string_lossy().to_string();
        if path == sink_device {
            continue;
        }
        let basename = entry
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("video?")
            .to_string();
        let sys_name = PathBuf::from("/sys/class/video4linux")
            .join(&basename)
            .join("name");
        let pretty = std::fs::read_to_string(&sys_name)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "Unknown camera".to_string());
        out.push(CameraEntry {
            path,
            label: format!("{} — {}", basename, pretty),
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}
