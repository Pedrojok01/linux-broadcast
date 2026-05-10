//! User-managed library of replacement-background images.
//!
//! Imported files are *copied* into
//! `~/.local/share/linux-broadcast/backgrounds/` (XDG data dir, resolved
//! via `directories::ProjectDirs`) so the library survives across
//! launches even when the user's source files move or get deleted.
//! `import` collision-renames with a numeric suffix to avoid clobbering.
//!
//! The library is the source of truth for which images appear in the
//! sidebar grid; the active selection is persisted as a path in
//! `Config::background_path` and re-read on next launch.
//!
//! Pixel decoding lives here (the `image` crate) rather than in the
//! pipeline crate so that `lb_pipeline` stays free of image-format
//! deps. The compositor receives a finished `Background::Image { rgba,
//! width, height }`.

use anyhow::{Context, Result, anyhow};
use directories::ProjectDirs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const QUALIFIER: &str = "io";
const ORG: &str = "linux-broadcast";
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
    import_into(&dir, src)
}

fn import_into(dir: &Path, src: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).with_context(|| format!("mkdir -p {}", dir.display()))?;
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
    list_in(&dir)
}

fn list_in(dir: &Path) -> Vec<LibraryEntry> {
    let read = match std::fs::read_dir(dir) {
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
    out.sort_by_key(|e| std::cmp::Reverse(e.1));
    out.into_iter().map(|(e, _)| e).collect()
}

pub fn remove(path: &Path) -> Result<()> {
    let dir = storage_dir().context("no data dir on this platform")?;
    remove_from(&dir, path)
}

fn remove_from(dir: &Path, path: &Path) -> Result<()> {
    if !path.starts_with(dir) {
        return Err(anyhow!(
            "refusing to delete path outside library: {}",
            path.display()
        ));
    }
    std::fs::remove_file(path).with_context(|| format!("rm {}", path.display()))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Build a 1×1 PNG so `import` has a real image to copy. `is_image`
    /// only checks the extension, so the bytes don't have to be valid.
    fn write_fake_png(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, b"\x89PNG\r\n\x1a\nfake").unwrap();
        path
    }

    #[test]
    fn import_then_list_returns_entry() {
        let lib = TempDir::new().unwrap();
        let src_dir = TempDir::new().unwrap();
        let src = write_fake_png(src_dir.path(), "wave.png");
        let dest = import_into(lib.path(), &src).unwrap();
        assert!(dest.exists());

        let entries = list_in(lib.path());
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, dest);
        assert_eq!(entries[0].label, "wave");
    }

    #[test]
    fn remove_clears_entry() {
        let lib = TempDir::new().unwrap();
        let src_dir = TempDir::new().unwrap();
        let src = write_fake_png(src_dir.path(), "studio.png");
        let dest = import_into(lib.path(), &src).unwrap();
        assert_eq!(list_in(lib.path()).len(), 1);

        remove_from(lib.path(), &dest).unwrap();
        assert!(!dest.exists());
        assert!(list_in(lib.path()).is_empty());
    }
}
