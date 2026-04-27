use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraEntry {
    pub path: String,
    pub label: String,
}

/// Enumerate `/dev/video*` capture-capable nodes, excluding `sink_device`.
///
/// We probe each `/dev/videoN` for a friendly name via
/// `/sys/class/video4linux/videoN/name`; we *don't* try to open the device
/// (cheap + non-disruptive). The same physical webcam often exposes several
/// numbered nodes (capture, metadata, control); for v1 we list them all and
/// let the user pick — Phase 3 can filter to capture-only via a v4l2 ioctl
/// if it gets noisy.
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
