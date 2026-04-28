//! Install a `.desktop` entry + an icon under `~/.local/share/` so the
//! Wayland compositor (KDE / GNOME) can resolve our window's `app_id`
//! to a real icon.
//!
//! Idempotent — runs on every launch but only writes when files are missing
//! or stale (binary path changed).
//!
//! Skipped entirely when the `.deb` package has already dropped a system
//! entry under `/usr/share/applications/LinuxBroadcast.desktop`. That
//! avoids a duplicate menu entry and keeps the system version (which uses
//! the canonical `/usr/bin/linux-broadcast` Exec path) authoritative.

use anyhow::{Context, Result};
use directories::BaseDirs;
use std::path::{Path, PathBuf};

const APP_ID: &str = "LinuxBroadcast";
const ICON_NAME: &str = "LinuxBroadcast";
const SYSTEM_DESKTOP: &str = "/usr/share/applications/LinuxBroadcast.desktop";

pub fn ensure_desktop_entry() -> Result<()> {
    let base = BaseDirs::new().context("no XDG base dirs")?;
    let data = base.data_dir(); // ~/.local/share

    if Path::new(SYSTEM_DESKTOP).exists() {
        // .deb-installed entry covers the menu. Actively remove any stale
        // per-user .desktop / icon left over from earlier dev launches —
        // XDG resolution puts $XDG_DATA_HOME ahead of /usr/share, so a
        // leftover with a now-broken Exec= would override the system
        // entry and the menu launcher would fail.
        remove_user_entry(data);
        return Ok(());
    }

    install_icon(data)?;
    install_desktop_file(data)?;
    Ok(())
}

fn remove_user_entry(data: &Path) {
    let desktop = data.join("applications").join(format!("{APP_ID}.desktop"));
    let icon = data
        .join("icons/hicolor/64x64/apps")
        .join(format!("{ICON_NAME}.png"));
    for path in [&desktop, &icon] {
        match std::fs::remove_file(path) {
            Ok(_) => log::info!("removed stale per-user entry {}", path.display()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => log::warn!("rm {}: {e:#}", path.display()),
        }
    }
}

fn install_icon(data: &Path) -> Result<()> {
    let dir = data.join("icons/hicolor/64x64/apps");
    let path = dir.join(format!("{ICON_NAME}.png"));
    if path.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("mkdir -p {}", dir.display()))?;
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

    // Quote the Exec path. Per the XDG Desktop Entry spec, paths with
    // spaces must be enclosed in double quotes; without this a dev build
    // sitting under "Coding/Projets Perso/" gets truncated at the first
    // space and the launcher fails with "Could not find the program".
    let exec_field = quote_exec(&exe);

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
        exec_field, ICON_NAME, APP_ID,
    );

    // Skip rewrite when the file already matches — avoids touching mtime.
    if let Ok(existing) = std::fs::read_to_string(&path) {
        if existing == want {
            return Ok(());
        }
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("mkdir -p {}", dir.display()))?;
    std::fs::write(&path, want).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// XDG Desktop Entry spec quoting for Exec=. Paths with spaces *must* be
/// wrapped in double quotes; embedded `"`, `` ` ``, `$`, `\` need a
/// preceding backslash. We also escape these inside un-quoted paths
/// defensively, but in practice unix paths only need the space-handling.
pub(crate) fn quote_exec(path: &Path) -> String {
    let s = path.to_string_lossy();
    let needs_quoting = s
        .chars()
        .any(|c| c.is_whitespace() || matches!(c, '"' | '`' | '$' | '\\'));
    if !needs_quoting {
        return s.into_owned();
    }
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        if matches!(c, '"' | '`' | '$' | '\\') {
            out.push('\\');
        }
        out.push(c);
    }
    out.push('"');
    out
}
