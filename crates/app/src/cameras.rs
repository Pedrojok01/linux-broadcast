//! Enumerate physical webcam nodes for the camera dropdown.
//!
//! We list `/dev/video*` and read each device's friendly name from
//! `/sys/class/video4linux/<n>/name`. That's a non-disruptive probe
//! (no `open(2)`, no LED) and avoids fighting another process for
//! exclusive access while the user is just *picking* a camera.
//!
//! A webcam claims several numbered nodes (capture, metadata). The sysfs
//! `index` attribute numbers them per physical device, and the capture
//! node is index 0, so the rest are dropped without opening anything.
//!
//! `sink_device` is excluded so the virtual cam (`/dev/video10`)
//! never appears as a selectable input.

use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraEntry {
    pub path: String,
    pub label: String,
}

/// Enumerate `/dev/video*` capture nodes, excluding `sink_device`, in
/// numeric order (`video2` before `video10`). Cheap enough to call each
/// time the picker opens, so hot-plugged cameras show up.
pub fn enumerate(sink_device: &str) -> Vec<CameraEntry> {
    enumerate_in(
        Path::new("/dev"),
        Path::new("/sys/class/video4linux"),
        sink_device,
    )
}

fn enumerate_in(dev_dir: &Path, sys_dir: &Path, sink_device: &str) -> Vec<CameraEntry> {
    let Ok(entries) = std::fs::read_dir(dev_dir) else {
        return Vec::new();
    };
    let mut out: Vec<(u32, CameraEntry)> = entries
        .flatten()
        .filter_map(|e| {
            let basename = e.file_name().into_string().ok()?;
            let n: u32 = basename.strip_prefix("video")?.parse().ok()?;
            let path = dev_dir.join(&basename).to_string_lossy().into_owned();
            let sys = sys_dir.join(&basename);
            let read = |attr: &str| {
                std::fs::read_to_string(sys.join(attr))
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            };
            // Missing `index` (old kernels) keeps the node rather than hiding a camera.
            if path == sink_device || read("index").is_some_and(|i| i != "0") {
                return None;
            }
            let pretty = read("name").unwrap_or_else(|| "Unknown camera".to_string());
            Some((
                n,
                CameraEntry {
                    path,
                    label: format!("{basename} — {pretty}"),
                },
            ))
        })
        .collect();
    out.sort_by_key(|(n, _)| *n);
    out.into_iter().map(|(_, c)| c).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn node(dev: &Path, sys: &Path, name: &str, index: Option<&str>, label: &str) {
        std::fs::write(dev.join(name), "").unwrap();
        let dir = sys.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("name"), format!("{label}\n")).unwrap();
        if let Some(i) = index {
            std::fs::write(dir.join("index"), format!("{i}\n")).unwrap();
        }
    }

    #[test]
    fn lists_capture_nodes_in_numeric_order_without_sink_or_metadata() {
        let (dev, sys) = (TempDir::new().unwrap(), TempDir::new().unwrap());
        let (d, s) = (dev.path(), sys.path());
        node(d, s, "video0", Some("0"), "C920");
        node(d, s, "video1", Some("1"), "C920");
        node(d, s, "video2", Some("0"), "Integrated");
        node(d, s, "video10", Some("0"), "LinuxBroadcast");
        node(d, s, "video11", None, "Old kernel");
        std::fs::write(d.join("vhost-net"), "").unwrap();

        let sink = d.join("video10").to_string_lossy().into_owned();
        let labels: Vec<_> = enumerate_in(d, s, &sink)
            .into_iter()
            .map(|c| c.label)
            .collect();
        assert_eq!(
            labels,
            [
                "video0 — C920",
                "video2 — Integrated",
                "video11 — Old kernel"
            ]
        );
    }
}
