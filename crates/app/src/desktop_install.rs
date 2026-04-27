//! Install a `.desktop` entry + an icon under `~/.local/share/` so the
//! Wayland compositor (KDE / GNOME) can resolve our window's `app_id`
//! to a real icon.
//!
//! Idempotent — runs on every launch but only writes when files are missing
//! or stale (binary path changed).

use anyhow::{Context, Result};
use directories::BaseDirs;
use std::path::{Path, PathBuf};

const APP_ID: &str = "io.Pedrojok01.LinuxBroadcast";
const ICON_NAME: &str = "io.Pedrojok01.LinuxBroadcast";

pub fn ensure_desktop_entry() -> Result<()> {
    let base = BaseDirs::new().context("no XDG base dirs")?;
    let data = base.data_dir(); // ~/.local/share

    install_icon(data)?;
    install_desktop_file(data)?;
    Ok(())
}

fn install_icon(data: &Path) -> Result<()> {
    let dir = data.join("icons/hicolor/64x64/apps");
    let path = dir.join(format!("{ICON_NAME}.png"));
    if path.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("mkdir -p {}", dir.display()))?;
    let icon = crate::icon::build();
    let img: image::ImageBuffer<image::Rgba<u8>, _> =
        image::ImageBuffer::from_raw(icon.width, icon.height, icon.rgba.clone())
            .context("icon buffer")?;
    img.save(&path)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn install_desktop_file(data: &Path) -> Result<()> {
    let dir = data.join("applications");
    let path = dir.join(format!("{APP_ID}.desktop"));
    let exe = std::env::current_exe()
        .ok()
        .and_then(|p| p.canonicalize().ok())
        .unwrap_or_else(|| PathBuf::from("linux-broadcast"));

    let want = format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Version=1.0\n\
         Name=LinuxBroadcast\n\
         GenericName=Virtual Webcam\n\
         Comment=Background-replacement virtual webcam\n\
         Exec={}\n\
         Icon={}\n\
         Terminal=false\n\
         StartupNotify=true\n\
         StartupWMClass={}\n\
         Categories=AudioVideo;Video;Utility;\n",
        exe.display(),
        ICON_NAME,
        APP_ID,
    );

    // Skip rewrite when the file already matches — avoids touching mtime.
    if let Ok(existing) = std::fs::read_to_string(&path) {
        if existing == want {
            return Ok(());
        }
    }
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("mkdir -p {}", dir.display()))?;
    std::fs::write(&path, want).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}
