use anyhow::{anyhow, Context, Result};
use directories::ProjectDirs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const QUALIFIER: &str = "io";
const ORG: &str = "Pedrojok01";
const APP: &str = "linux-broadcast";

#[derive(Debug, Clone)]
pub struct LibraryEntry {
    pub path: PathBuf,
    pub label: String,
}

pub fn storage_dir() -> Option<PathBuf> {
    ProjectDirs::from(QUALIFIER, ORG, APP).map(|d| d.data_dir().join("backgrounds"))
}

fn ensure_dir() -> Result<PathBuf> {
    let dir = storage_dir().context("no data dir on this platform")?;
    std::fs::create_dir_all(&dir).with_context(|| format!("mkdir -p {}", dir.display()))?;
    Ok(dir)
}

/// Copy a freshly-picked image into the library, returning its new in-library
/// path. If a file with the same name already exists, a numeric suffix is
/// appended.
pub fn import(src: &Path) -> Result<PathBuf> {
    let dir = ensure_dir()?;
    let ext = src
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("img")
        .to_lowercase();
    let stem = src
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("background")
        .to_string();
    let mut candidate = dir.join(format!("{stem}.{ext}"));
    let mut n = 1;
    while candidate.exists() {
        candidate = dir.join(format!("{stem}-{n}.{ext}"));
        n += 1;
    }
    std::fs::copy(src, &candidate)
        .with_context(|| format!("copy {} → {}", src.display(), candidate.display()))?;
    Ok(candidate)
}

/// List every image file currently in the library, newest first.
pub fn list() -> Vec<LibraryEntry> {
    let Some(dir) = storage_dir() else {
        return Vec::new();
    };
    let read = match std::fs::read_dir(&dir) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let mut out: Vec<(LibraryEntry, SystemTime)> = read
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            if !is_image(&p) {
                return None;
            }
            let label = p
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("background")
                .to_string();
            let mtime = e.metadata().and_then(|m| m.modified()).ok()?;
            Some((LibraryEntry { path: p, label }, mtime))
        })
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1));
    out.into_iter().map(|(e, _)| e).collect()
}

pub fn remove(path: &Path) -> Result<()> {
    let dir = storage_dir().context("no data dir on this platform")?;
    if !path.starts_with(&dir) {
        return Err(anyhow!(
            "refusing to delete path outside library: {}",
            path.display()
        ));
    }
    std::fs::remove_file(path)
        .with_context(|| format!("rm {}", path.display()))?;
    Ok(())
}

fn is_image(p: &Path) -> bool {
    matches!(
        p.extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp" | "bmp")
    )
}
