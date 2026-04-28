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
    /// When true, the camera is held on regardless of consumer count.
    /// Default off; the pipeline's lazy mode opens the camera only while
    /// a real consumer is reading `/dev/video10`. Useful for streamer /
    /// rehearsal flows where the camera LED should stay lit.
    pub force_on: bool,
    /// When true, the GUI's preview pane renders live composited frames.
    /// Off → preview pane shows a static placeholder; the pipeline stops
    /// forwarding frames to the GUI (saves a per-frame RGBA clone). The
    /// broadcast itself is unaffected — consumers of `/dev/video10` see
    /// the same picture either way.
    pub show_preview: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            source_device: "/dev/video0".to_string(),
            sink_device: "/dev/video10".to_string(),
            width: 1280,
            height: 800,
            framerate: 30,
            mode: Mode::Blur,
            blur_strength: 0.62,
            background_path: None,
            model: Model::default(),
            start_on_login: false,
            force_on: false,
            show_preview: true,
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
