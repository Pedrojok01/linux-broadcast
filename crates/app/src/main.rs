mod autostart;
mod backgrounds;
mod cameras;
mod config;
mod desktop_install;
mod icon;
mod lock;
mod theme;
mod tray;
mod ui;

use anyhow::Result;

use crate::lock::InstanceLock;

/// Bundled MediaPipe Selfie Segmentation ONNX (binary, ~450 KB).
/// Sourced from `onnx-community/mediapipe_selfie_segmentation` on Hugging Face.
pub(crate) const MODEL_BINARY_ONNX: &[u8] = include_bytes!("../../../models/selfie_segmenter.onnx");

/// Bundled MediaPipe Selfie Multiclass ONNX (6 classes: bg/hair/body/face/clothes/other,
/// ~16 MB). Converted via tf2onnx from the official MediaPipe TFLite.
pub(crate) const MODEL_MULTICLASS_ONNX: &[u8] =
    include_bytes!("../../../models/selfie_multiclass.onnx");

/// Bundled Robust Video Matting ONNX (MobileNetV3 backbone, fp32, ~15 MB).
/// Sourced from PeterL1n/RobustVideoMatting v1.0.0 release.
pub(crate) const MODEL_RVM_ONNX: &[u8] = include_bytes!("../../../models/rvm.onnx");

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    if std::env::var_os("LB_DUMP_ICON").is_some() {
        let icon = icon::build();
        let img: image::ImageBuffer<image::Rgba<u8>, _> =
            image::ImageBuffer::from_raw(icon.width, icon.height, icon.rgba.clone())
                .expect("icon buffer");
        img.save("/tmp/lb-icon.png")?;
        println!("wrote /tmp/lb-icon.png ({}×{})", icon.width, icon.height);
        return Ok(());
    }

    // Process-level single-instance lock. With lazy mode, the pipeline
    // can sit idle (no v4l2sink contention) for arbitrarily long, so we
    // need a real lock that covers the whole LB process — otherwise two
    // LB instances would both poll for consumers and race when one
    // arrives. Held until `main` returns.
    let _instance_lock = match InstanceLock::try_acquire()? {
        Some(l) => l,
        None => {
            log::info!("another LinuxBroadcast instance is already running; exiting");
            return Ok(());
        }
    };

    // Headless mode = same eframe loop, but starts with the window
    // hidden in the tray and auto-starts the pipeline. Same UX as the
    // old separate code path (autostart on login, no window flash), but
    // a single, testable code path. Triggered by `--headless` (used by
    // the autostart .desktop) or `LB_HEADLESS=1` (kept for back-compat).
    let argv: Vec<String> = std::env::args().collect();
    let headless =
        argv.iter().any(|a| a == "--headless") || std::env::var_os("LB_HEADLESS").is_some();

    if !headless {
        if let Err(e) = desktop_install::ensure_desktop_entry() {
            log::warn!("desktop entry install: {e:#}");
        }
    }

    ui::run(headless)
}
