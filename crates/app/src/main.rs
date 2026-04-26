use anyhow::Result;
use lb_pipeline::{Background, Pipeline, PipelineConfig};

/// Bundled MediaPipe Selfie Segmentation ONNX model (~450 KB, general variant).
/// Sourced from `onnx-community/mediapipe_selfie_segmentation` on Hugging Face.
const MODEL_ONNX: &[u8] = include_bytes!("../../../models/selfie_segmenter.onnx");

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Phase 1 hardcoded config. Phase 2 will replace this with the GUI.
    let cfg = PipelineConfig {
        source_device: std::env::var("LB_SOURCE").unwrap_or_else(|_| "/dev/video0".into()),
        sink_device: std::env::var("LB_SINK").unwrap_or_else(|_| "/dev/video10".into()),
        width: 1280,
        height: 720,
        framerate: 30,
        background: Background::Blur,
    };

    log::info!(
        "starting pipeline {} → {} ({}x{}@{}fps, blur background)",
        cfg.source_device,
        cfg.sink_device,
        cfg.width,
        cfg.height,
        cfg.framerate,
    );

    let pipeline = Pipeline::start(cfg, MODEL_ONNX)?;
    pipeline.run_until_done()?;
    Ok(())
}
