//! Persisted user configuration.
//!
//! TOML at `~/.config/linux-broadcast/config.toml` (path resolved via
//! `directories::ProjectDirs` so it follows XDG on Linux). Loaded once
//! at startup and rewritten each time a user-facing toggle flips.
//!
//! Every field is annotated with `#[serde(default)]` at the struct
//! level so older configs missing newly-added keys still load — that's
//! how we add settings (`auto_frame`, `show_preview`, …) without
//! migrations or breaking existing installs.
//!
//! `Model` and `Mode` here mirror their pipeline-side counterparts
//! (`lb_pipeline::ModelKind`, `lb_pipeline::Background` discriminant)
//! but with `serde` impls. Keeping them separate is deliberate: the
//! pipeline crate stays free of `serde` and `toml` dependencies, so
//! it's still usable as a plain library by downstream code.

use anyhow::{Context, Result};
use directories::ProjectDirs;
use lb_pipeline::ModelKind;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Mirrors `lb_pipeline::ModelKind` but with `serde` impls for the config
/// file. Kept separate so the pipeline crate doesn't need a serde dep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Model {
    /// MediaPipe Selfie Segmentation (binary, ~450 KB).
    #[default]
    SelfieBinary,
    /// MediaPipe Selfie Multiclass (~16 MB) — segments hair / face / body /
    /// clothes / others separately, gives sharper edges.
    SelfieMulticlass,
    /// Robust Video Matting (MobileNetV3, ~15 MB) — recurrent, frame-rate
    /// alpha matte. Best edges; slowest.
    Rvm,
}

impl Model {
    pub fn label(self) -> &'static str {
        match self {
            Model::SelfieBinary => "Selfie (binary, fast)",
            Model::SelfieMulticlass => "Selfie multiclass (sharper)",
            Model::Rvm => "RVM (best edges, slower)",
        }
    }
    pub fn into_kind(self) -> ModelKind {
        match self {
            Model::SelfieBinary => ModelKind::SelfieBinary,
            Model::SelfieMulticlass => ModelKind::SelfieMulticlass,
            Model::Rvm => ModelKind::Rvm,
        }
    }
    pub const ALL: &'static [Model] = &[Model::SelfieBinary, Model::SelfieMulticlass, Model::Rvm];
}

const QUALIFIER: &str = "io";
const ORG: &str = "linux-broadcast";
const APP: &str = "linux-broadcast";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Pass the camera through with no segmentation or blur.
    None,
    /// Blur the original frame and put the user on top.
    #[default]
    Blur,
    /// Replace the background with the saved image at `background_path`.
    Replace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub source_device: String,
    pub sink_device: String,
    pub width: u32,
    pub height: u32,
    pub framerate: u32,
    pub mode: Mode,
    pub blur_strength: f32,
    pub background_path: Option<PathBuf>,
    pub model: Model,
    /// When true, the GUI keeps an XDG-autostart `.desktop` in
    /// `~/.config/autostart/` so the headless pipeline runs at login and
    /// `/dev/video10` is populated before any conferencing app starts.
    /// Off by default — flipping the toggle in the sidebar writes/removes
    /// the autostart file.
    pub start_on_login: bool,
    /// When true, the GUI's preview pane renders live composited frames.
    /// Off → preview pane shows a static placeholder; the pipeline stops
    /// forwarding frames to the GUI (saves a per-frame RGBA clone). The
    /// broadcast itself is unaffected — consumers of `/dev/video10` see
    /// the same picture either way.
    pub show_preview: bool,
    /// When true, the pipeline runs the auto-frame stage: a smoothed
    /// virtual-PTZ crop driven by the segmentation mask keeps the user
    /// centered (Meet-style). Forces segmentation even in `Mode::None`
    /// because the bbox needs a mask.
    pub auto_frame: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            source_device: "/dev/video0".to_string(),
            sink_device: "/dev/video10".to_string(),
            width: 1280,
            height: 900,
            framerate: 30,
            mode: Mode::Blur,
            blur_strength: 0.62,
            background_path: None,
            model: Model::default(),
            start_on_login: false,
            show_preview: true,
            auto_frame: false,
        }
    }
}

impl Config {
    pub fn config_dir() -> Option<PathBuf> {
        ProjectDirs::from(QUALIFIER, ORG, APP).map(|d| d.config_dir().to_path_buf())
    }

    pub fn file_path() -> Option<PathBuf> {
        Self::config_dir().map(|d| d.join("config.toml"))
    }

    pub fn load() -> Self {
        match Self::try_load() {
            Ok(cfg) => cfg,
            Err(e) => {
                log::warn!("config load failed ({e:#}); using defaults");
                Self::default()
            }
        }
    }

    fn try_load() -> Result<Self> {
        let path = Self::file_path().context("no config dir on this platform")?;
        if !path.exists() {
            return Ok(Self::default());
        }
        let s =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        toml::from_str(&s).with_context(|| format!("parse {}", path.display()))
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::file_path().context("no config dir on this platform")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("mkdir -p {}", parent.display()))?;
        }
        let s = toml::to_string_pretty(self).context("serialize config")?;
        std::fs::write(&path, s).with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    /// True when the configured background image actually resolves.
    pub fn background_image_path(&self) -> Option<&Path> {
        self.background_path.as_deref().filter(|p| p.exists())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use tempfile::TempDir;

    /// Point ProjectDirs at a fresh tempdir for the duration of the test.
    /// Must run #[serial] — env vars are process-global.
    fn with_temp_xdg<F: FnOnce(&Path)>(f: F) {
        let tmp = TempDir::new().unwrap();
        // Save the prior value (if any) so we don't leak state into
        // sibling tests that happen to also touch XDG_CONFIG_HOME.
        let prior = std::env::var_os("XDG_CONFIG_HOME");
        std::env::set_var("XDG_CONFIG_HOME", tmp.path());
        f(tmp.path());
        match prior {
            Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
    }

    #[test]
    #[serial]
    fn default_load_with_redirected_xdg() {
        with_temp_xdg(|_| {
            let cfg = Config::load();
            let default = Config::default();
            assert_eq!(cfg.source_device, default.source_device);
            assert_eq!(cfg.sink_device, default.sink_device);
            assert_eq!(cfg.width, default.width);
            assert_eq!(cfg.height, default.height);
            assert_eq!(cfg.framerate, default.framerate);
            assert_eq!(cfg.mode, default.mode);
            assert_eq!(cfg.model, default.model);
            assert_eq!(cfg.show_preview, default.show_preview);
        });
    }

    #[test]
    #[serial]
    fn roundtrip_save_then_load() {
        with_temp_xdg(|_| {
            let cfg = Config {
                width: 1920,
                height: 1080,
                framerate: 60,
                mode: Mode::Replace,
                blur_strength: 0.123,
                background_path: Some(PathBuf::from("/tmp/some_bg.png")),
                model: Model::Rvm,
                start_on_login: true,
                show_preview: false,
                ..Config::default()
            };
            cfg.save().unwrap();

            let loaded = Config::load();
            assert_eq!(loaded.width, 1920);
            assert_eq!(loaded.height, 1080);
            assert_eq!(loaded.framerate, 60);
            assert_eq!(loaded.mode, Mode::Replace);
            assert!((loaded.blur_strength - 0.123).abs() < 1e-6);
            assert_eq!(
                loaded.background_path,
                Some(PathBuf::from("/tmp/some_bg.png"))
            );
            assert_eq!(loaded.model, Model::Rvm);
            assert!(loaded.start_on_login);
            assert!(!loaded.show_preview);
        });
    }

    #[test]
    #[serial]
    fn forward_compat_missing_field() {
        // A user upgrading from an older build has a TOML without
        // `start_on_login` / `show_preview`. Must load with defaults
        // filled in.
        with_temp_xdg(|root| {
            let dir = root.join("linux-broadcast");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join("config.toml"),
                "source_device = \"/dev/video0\"\nwidth = 1280\nheight = 720\n",
            )
            .unwrap();

            let cfg = Config::load();
            assert_eq!(cfg.width, 1280);
            assert_eq!(cfg.height, 720);
            // Defaulted fields:
            assert_eq!(cfg.framerate, Config::default().framerate);
            assert_eq!(cfg.show_preview, Config::default().show_preview);
            assert_eq!(cfg.start_on_login, Config::default().start_on_login);
        });
    }

    #[test]
    #[serial]
    fn unknown_field_does_not_panic() {
        // Older or experimental builds may write extra keys. Loader must
        // silently ignore them rather than failing the whole config.
        with_temp_xdg(|root| {
            let dir = root.join("linux-broadcast");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join("config.toml"),
                "width = 640\nheight = 480\nfuture_field = 42\nspeculative = \"hi\"\n",
            )
            .unwrap();

            let cfg = Config::load();
            assert_eq!(cfg.width, 640);
            assert_eq!(cfg.height, 480);
        });
    }
}
