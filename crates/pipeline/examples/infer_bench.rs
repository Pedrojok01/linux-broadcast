//! Headless inference microbenchmark: times `Segmenter::segment()` for RVM
//! on a synthetic frame, isolating inference cost from the v4l2 pipeline.
//! Used to compare CPU against the `cuda` feature's GPU execution provider
//! without a webcam, `/dev/video10`, or a consumer.
//!
//!   cargo run --release -p lb-pipeline --example infer_bench --features cuda
//!   LB_FORCE_CPU=1 cargo run --release -p lb-pipeline --example infer_bench --features cuda
//!
//! Env knobs:
//!   LB_BENCH_FRAMES  measured frames     (default 200)
//!   LB_BENCH_WARMUP  warmup frames       (default 20)
//!   LB_BENCH_W/H     frame size          (default 1280x720)

use std::time::Instant;

use lb_pipeline::{ModelKind, Segmenter};

static RVM: &[u8] = include_bytes!("../../../models/rvm.onnx");

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let w = env_usize("LB_BENCH_W", 1280);
    let h = env_usize("LB_BENCH_H", 720);
    let frames = env_usize("LB_BENCH_FRAMES", 200);
    let warmup = env_usize("LB_BENCH_WARMUP", 20);

    // Centered bright block on a dark field so RVM has a plausible
    // foreground to matte.
    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            let fg = x > w / 3 && x < 2 * w / 3 && y > h / 4;
            let v = if fg { 200 } else { 30 };
            rgba[i] = v;
            rgba[i + 1] = v;
            rgba[i + 2] = v;
            rgba[i + 3] = 255;
        }
    }

    let mut seg = Segmenter::from_bytes(ModelKind::Rvm, RVM)?;

    eprintln!("[infer_bench] model=RVM {w}x{h} warmup={warmup} frames={frames}");
    for _ in 0..warmup {
        seg.segment(&rgba, w, h)?;
    }

    let mut times = Vec::with_capacity(frames);
    for _ in 0..frames {
        let t = Instant::now();
        let _mask = seg.segment(&rgba, w, h)?;
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let p50 = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];
    eprintln!(
        "[infer_bench] segment ms: avg={avg:.2} p50={p50:.2} min={min:.2} max={max:.2}  \
         → {:.1} fps (inference only)",
        1000.0 / avg
    );
    Ok(())
}
