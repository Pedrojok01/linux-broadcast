mod backgrounds;
mod cameras;
mod config;
mod desktop_install;
mod icon;
mod theme;
mod ui;

use anyhow::Result;

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

    if std::env::var_os("LB_HEADLESS").is_some() {
        return ui::run_headless();
    }
    if let Err(e) = desktop_install::ensure_desktop_entry() {
        log::warn!("desktop entry install: {e:#}");
    }
    ui::run_gui()
}
