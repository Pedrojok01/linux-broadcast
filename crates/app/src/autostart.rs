//! Opt-in XDG autostart entry.
//!
//! When the user flips "Start on login" in the sidebar we drop a tiny
//! `.desktop` file under `~/.config/autostart/` that runs the headless
//! pipeline at every login. That keeps `/dev/video10` populated *before*
//! Zoom / Meet / Signal / Firefox start probing for cameras — so the
//! virtual cam just shows up in their dropdown alongside the physical one,
//! no manual launch needed.
//!
//! Off by default. Toggling the switch off removes the file. This module
//! never touches anything outside `~/.config/autostart/`.
//!
//! Filename is `LinuxBroadcast-autostart.desktop` (distinct from the menu
//! entry `LinuxBroadcast.desktop`) so the user can identify it at a glance
//! and so disabling autostart doesn't blow away the menu entry.
//!
//! Note: every desktop honours XDG autostart slightly differently. GNOME
//! and KDE respect both `Hidden=false` and `X-GNOME-Autostart-enabled=true`;
//! XFCE / Cinnamon ignore the GNOME-specific key and rely on `Hidden`. We
//! set both for portability.

use anyhow::{Context, Result};
use directories::BaseDirs;
use std::path::{Path, PathBuf};

const FILENAME: &str = "LinuxBroadcast-autostart.desktop";

fn autostart_path() -> Result<PathBuf> {
    let base = BaseDirs::new().context("no XDG base dirs")?;
    Ok(autostart_path_in(base.config_dir()))
}

fn autostart_path_in(config_dir: &Path) -> PathBuf {
    config_dir.join("autostart").join(FILENAME)
}

/// Render the .desktop body. `exec_path` is the absolute path to the
/// linux-broadcast binary; we pass `--headless` so no window pops up.
/// The path is XDG-quoted so a dev build sitting under a directory with
/// spaces (e.g. "Coding/Projets Perso/") still parses correctly.
fn desktop_body(exec_path: &Path) -> String {
    format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Version=1.0\n\
         Name=LinuxBroadcast (autostart)\n\
         Comment=Run the LinuxBroadcast virtual camera in the background at login\n\
         Exec={} --headless\n\
         Icon=LinuxBroadcast\n\
         Terminal=false\n\
         StartupNotify=false\n\
         X-GNOME-Autostart-enabled=true\n\
         Hidden=false\n\
         Categories=AudioVideo;Video;Utility;\n",
        crate::desktop_install::quote_exec(exec_path),
    )
}

/// Write `~/.config/autostart/LinuxBroadcast-autostart.desktop` if absent
/// or stale. Idempotent — re-run is cheap.
pub fn install(exec_path: &Path) -> Result<()> {
    install_at(&autostart_path()?, exec_path)
}

fn install_at(path: &Path, exec_path: &Path) -> Result<()> {
    let body = desktop_body(exec_path);
    if let Ok(existing) = std::fs::read_to_string(path)
        && existing == body
    {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("mkdir -p {}", parent.display()))?;
    }
    std::fs::write(path, body).with_context(|| format!("write {}", path.display()))?;
    log::info!("autostart enabled → {}", path.display());
    Ok(())
}

/// Remove the autostart entry if present. Missing-file is success.
pub fn uninstall() -> Result<()> {
    uninstall_at(&autostart_path()?)
}

fn uninstall_at(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(_) => {
            log::info!("autostart disabled (removed {})", path.display());
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("rm {}", path.display())),
    }
}

pub fn is_installed() -> bool {
    autostart_path().map(|p| p.exists()).unwrap_or(false)
}

/// Bring the on-disk state in line with `desired`. Used at startup so the
/// toggle stays truthful even if the user wiped `~/.config/autostart/` by
/// hand or copied a config from another machine.
pub fn reconcile(desired: bool, exec_path: &Path) -> Result<()> {
    let on_disk = is_installed();
    match (desired, on_disk) {
        (true, _) => install(exec_path),
        (false, true) => uninstall(),
        (false, false) => Ok(()),
    }
}

#[cfg(test)]
fn reconcile_at(path: &Path, desired: bool, exec_path: &Path) -> Result<()> {
    let on_disk = path.exists();
    match (desired, on_disk) {
        (true, false) | (true, true) => install_at(path, exec_path),
        (false, true) => uninstall_at(path),
        (false, false) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn reconcile_true_creates_desktop_entry() {
        let tmp = TempDir::new().unwrap();
        let path = autostart_path_in(tmp.path());
        let exec = PathBuf::from("/usr/local/bin/linux-broadcast");
        reconcile_at(&path, true, &exec).unwrap();
        assert!(path.exists(), "expected {} to exist", path.display());
        let body = std::fs::read_to_string(&path).unwrap();
        assert!(body.contains("Exec="));
        assert!(body.contains("--headless"));
    }

    #[test]
    fn reconcile_false_removes_it() {
        let tmp = TempDir::new().unwrap();
        let path = autostart_path_in(tmp.path());
        let exec = PathBuf::from("/usr/local/bin/linux-broadcast");
        reconcile_at(&path, true, &exec).unwrap();
        assert!(path.exists());

        reconcile_at(&path, false, &exec).unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn reconcile_idempotent() {
        let tmp = TempDir::new().unwrap();
        let path = autostart_path_in(tmp.path());
        let exec = PathBuf::from("/usr/local/bin/linux-broadcast");
        reconcile_at(&path, true, &exec).unwrap();
        let mtime1 = std::fs::metadata(&path).unwrap().modified().unwrap();

        // Sleep just enough to see a different mtime if a write
        // happened. 1.1s is generous; some FSes only have second
        // resolution.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        reconcile_at(&path, true, &exec).unwrap();
        let mtime2 = std::fs::metadata(&path).unwrap().modified().unwrap();

        assert_eq!(mtime1, mtime2, "second reconcile must be a no-op");
    }
}
