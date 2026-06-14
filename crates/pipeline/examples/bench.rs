//! Headless throughput bench for the feeder loop.
//!
//! Starts the real pipeline (same code path as the app) with a chosen
//! background mode and framing, then idles for a fixed duration while a
//! separate consumer reads the loopback. The per-stage numbers come from
//! the feeder's own profiler — run with `LB_PROFILE=1`.
//!
//! A consumer must read `/dev/video10` to push the feeder Live, e.g.:
//!   ffmpeg -nostdin -loglevel error -f v4l2 -i /dev/video10 -t 25 -f null -
//!
//! Env knobs:
//!   LB_BENCH_BG       blur | image | none   (default blur)
//!   LB_BENCH_FRAMING  0 | 1                  (default 0)
//!   LB_BENCH_SECS     run duration, seconds  (default 30)
//!   LB_PROFILE        any value → emit timing lines
//!   LB_INFER_INTERVAL inference cadence, ≥1  (default 1)

use std::time::Duration;

use lb_pipeline::{Background, Pipeline, PipelineConfig};

static RVM: &[u8] = include_bytes!("../../../models/rvm.onnx");
static MULTICLASS: &[u8] = include_bytes!("../../../models/selfie_multiclass.onnx");

fn main() -> anyhow::Result<()> {
    let bg_kind = std::env::var("LB_BENCH_BG").unwrap_or_else(|_| "blur".into());
    let framing = std::env::var("LB_BENCH_FRAMING")
        .map(|v| v == "1")
        .unwrap_or(false);
    let secs = std::env::var("LB_BENCH_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30);
    let interval = std::env::var("LB_INFER_INTERVAL").unwrap_or_else(|_| "1".into());

    let width = 1280u32;
    let height = 720u32;
    let background = match bg_kind.as_str() {
        "none" => Background::None,
        "image" => {
            // Synthetic gradient standing in for a user background image —
            // exercises the Image composite path (asymmetric blend + alpha
            // decontamination) without pulling in an image decoder.
            let mut rgba = vec![0u8; (width * height * 4) as usize];
            for y in 0..height {
                for x in 0..width {
                    let i = ((y * width + x) * 4) as usize;
                    rgba[i] = (x * 255 / width) as u8;
                    rgba[i + 1] = (y * 255 / height) as u8;
                    rgba[i + 2] = 128;
                    rgba[i + 3] = 255;
                }
            }
            Background::Image {
                rgba,
                width,
                height,
            }
        }
        _ => Background::Blur {
            strength: Background::DEFAULT_BLUR_STRENGTH,
        },
    };

    eprintln!(
        "[bench] bg={bg_kind} framing={framing} interval={interval} secs={secs} — \
         start an ffmpeg consumer on /dev/video10 now"
    );

    let cfg = PipelineConfig {
        width,
        height,
        framing,
        background,
        ..Default::default()
    };

    let pipeline = Pipeline::start(cfg, MULTICLASS, RVM)?;
    std::thread::sleep(Duration::from_secs(secs));
    pipeline.stop();
    eprintln!("[bench] done");
    Ok(())
}
