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

/// Mirrors `lb_pipeline::ModelKind` with `serde` impls for the config
/// file. Kept separate so the pipeline crate doesn't need a serde dep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Model {
    /// MediaPipe Selfie Multiclass — 256×256, per-frame. Low-CPU fallback.
    SelfieMulticlass,
    /// Robust Video Matting (MobileNetV3) — recurrent, frame-res alpha.
    #[default]
    #[serde(alias = "selfie_binary")]
    Rvm,
}

impl Model {
    pub fn label(self) -> &'static str {
        match self {
            Model::SelfieMulticlass => "Selfie multiclass (low CPU)",
            Model::Rvm => "RVM (best edges)",
        }
    }
    pub fn into_kind(self) -> ModelKind {
        match self {
            Model::SelfieMulticlass => ModelKind::SelfieMulticlass,
            Model::Rvm => ModelKind::Rvm,
        }
    }
    pub const ALL: &'static [Model] = &[Model::Rvm, Model::SelfieMulticlass];
}

/// Accepted output size range, per side, and the top framerate.
const MIN_DIM: u32 = 64;
const MAX_DIM: u32 = 4096;
const MAX_FPS: u32 = 60;

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
            // 16:9 — must match common webcam aspect so the source isn't
            // stretched into a non-standard output ratio. See the
            // aspect-preserving crop in `lb_pipeline::build_source_pipeline`.
            width: 1280,
            height: 720,
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
        let Some(path) = Self::file_path() else {
            log::warn!("no config dir on this platform; using defaults");
            return Self::default();
        };
        Self::load_from(&path)
    }

    fn load_from(path: &Path) -> Self {
        let mut cfg = match Self::try_load_from(path) {
            Ok(cfg) => cfg,
            Err(e) => {
                // The next save overwrites the file, so keep the user's copy.
                let backup = path.with_extension("toml.bad");
                let kept = std::fs::copy(path, &backup).is_ok();
                log::warn!(
                    "config load failed ({e:#}); using defaults{}",
                    if kept {
                        format!(", old file kept at {}", backup.display())
                    } else {
                        String::new()
                    }
                );
                Self::default()
            }
        };
        cfg.sanitize();
        cfg
    }

    fn try_load_from(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let s =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        toml::from_str(&s).with_context(|| format!("parse {}", path.display()))
    }

    /// Bring hand-edited values back into what the pipeline can run: a
    /// zero framerate or height panics it, and YUY2 needs an even width.
    fn sanitize(&mut self) {
        let d = Self::default();
        let dims_ok = |v: u32| (MIN_DIM..=MAX_DIM).contains(&v);
        if !dims_ok(self.width) || !dims_ok(self.height) {
            log::warn!(
                "config: resolution {}×{} out of range, using {}×{}",
                self.width,
                self.height,
                d.width,
                d.height
            );
            (self.width, self.height) = (d.width, d.height);
        }
        if !self.width.is_multiple_of(2) || !self.height.is_multiple_of(2) {
            let (w, h) = (self.width & !1, self.height & !1);
            log::warn!(
                "config: odd resolution {}×{}, using {w}×{h}",
                self.width,
                self.height
            );
            (self.width, self.height) = (w, h);
        }
        if !(1..=MAX_FPS).contains(&self.framerate) {
            let fps = if self.framerate == 0 {
                d.framerate
            } else {
                MAX_FPS
            };
            log::warn!(
                "config: framerate {} out of range, using {fps}",
                self.framerate
            );
            self.framerate = fps;
        }
        if !(0.0..=1.0).contains(&self.blur_strength) {
            let b = if self.blur_strength.is_finite() {
                self.blur_strength.clamp(0.0, 1.0)
            } else {
                d.blur_strength
            };
            log::warn!(
                "config: blur_strength {} out of range, using {b}",
                self.blur_strength
            );
            self.blur_strength = b;
        }
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::file_path().context("no config dir on this platform")?;
        self.save_to(&path)
    }

    fn save_to(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("mkdir -p {}", parent.display()))?;
        }
        let s = toml::to_string_pretty(self).context("serialize config")?;
        // Write then rename, so a crash mid-write never truncates the file.
        let tmp = path.with_extension("toml.tmp");
        std::fs::write(&tmp, s).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("rename to {}", path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn config_file(dir: &Path) -> PathBuf {
        dir.join("config.toml")
    }

    /// Regression guard: the default output resolution must be 16:9. A
    /// non-16:9 default (it was once 1280×900) makes the source graph
    /// stretch the 16:9 webcam capture vertically — people look distorted
    /// in conferencing apps.
    #[test]
    fn default_resolution_is_16_9() {
        let d = Config::default();
        assert_eq!(
            d.width * 9,
            d.height * 16,
            "default {}×{} is not 16:9",
            d.width,
            d.height,
        );
    }

    #[test]
    fn default_load_with_redirected_xdg() {
        let tmp = TempDir::new().unwrap();
        let cfg = Config::load_from(&config_file(tmp.path()));
        let default = Config::default();
        assert_eq!(cfg.source_device, default.source_device);
        assert_eq!(cfg.sink_device, default.sink_device);
        assert_eq!(cfg.width, default.width);
        assert_eq!(cfg.height, default.height);
        assert_eq!(cfg.framerate, default.framerate);
        assert_eq!(cfg.mode, default.mode);
        assert_eq!(cfg.model, default.model);
        assert_eq!(cfg.show_preview, default.show_preview);
    }

    #[test]
    fn roundtrip_save_then_load() {
        let tmp = TempDir::new().unwrap();
        let path = config_file(tmp.path());
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
        cfg.save_to(&path).unwrap();

        let loaded = Config::load_from(&path);
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
    }

    #[test]
    fn forward_compat_missing_field() {
        // A user upgrading from an older build has a TOML without
        // `start_on_login` / `show_preview`. Must load with defaults
        // filled in.
        let tmp = TempDir::new().unwrap();
        let path = config_file(tmp.path());
        std::fs::write(
            &path,
            "source_device = \"/dev/video0\"\nwidth = 1280\nheight = 720\n",
        )
        .unwrap();

        let cfg = Config::load_from(&path);
        assert_eq!(cfg.width, 1280);
        assert_eq!(cfg.height, 720);
        // Defaulted fields:
        assert_eq!(cfg.framerate, Config::default().framerate);
        assert_eq!(cfg.show_preview, Config::default().show_preview);
        assert_eq!(cfg.start_on_login, Config::default().start_on_login);
    }

    #[test]
    fn sanitize_fixes_values_the_pipeline_cannot_run() {
        let tmp = TempDir::new().unwrap();
        let path = config_file(tmp.path());
        std::fs::write(
            &path,
            "width = 1281\nheight = 721\nframerate = 0\nblur_strength = 7.5\n",
        )
        .unwrap();
        let cfg = Config::load_from(&path);
        assert_eq!((cfg.width, cfg.height), (1280, 720));
        assert_eq!(cfg.framerate, Config::default().framerate);
        assert_eq!(cfg.blur_strength, 1.0);

        std::fs::write(&path, "width = 1280\nheight = 0\nframerate = 240\n").unwrap();
        let cfg = Config::load_from(&path);
        let d = Config::default();
        assert_eq!((cfg.width, cfg.height), (d.width, d.height));
        assert_eq!(cfg.framerate, MAX_FPS);
    }

    #[test]
    fn unparsable_file_is_backed_up_before_defaults() {
        let tmp = TempDir::new().unwrap();
        let path = config_file(tmp.path());
        std::fs::write(&path, "width = \"wide\"\n").unwrap();
        let cfg = Config::load_from(&path);
        assert_eq!(cfg.width, Config::default().width);
        let backup = std::fs::read_to_string(tmp.path().join("config.toml.bad")).unwrap();
        assert_eq!(backup, "width = \"wide\"\n");
    }

    #[test]
    fn save_leaves_no_temp_file() {
        let tmp = TempDir::new().unwrap();
        let path = config_file(tmp.path());
        Config::default().save_to(&path).unwrap();
        Config::default().save_to(&path).unwrap();
        let names: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, ["config.toml"]);
    }

    #[test]
    fn unknown_field_does_not_panic() {
        // Older or experimental builds may write extra keys. Loader must
        // silently ignore them rather than failing the whole config.
        let tmp = TempDir::new().unwrap();
        let path = config_file(tmp.path());
        std::fs::write(
            &path,
            "width = 640\nheight = 480\nfuture_field = 42\nspeculative = \"hi\"\n",
        )
        .unwrap();

        let cfg = Config::load_from(&path);
        assert_eq!(cfg.width, 640);
        assert_eq!(cfg.height, 480);
    }
}
