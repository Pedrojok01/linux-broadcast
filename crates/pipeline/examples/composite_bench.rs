//! Headless microbenchmark for the composite stage. Times
//! `Compositor::composite()` per background mode on synthetic frames +
//! mask, isolating composite cost from inference and v4l2. GPU-independent
//! (composite is CPU), so it measures the stage we optimize directly.
//!
//!   cargo run --release -p lb-pipeline --example composite_bench
//!
//! Env: LB_BENCH_FRAMES (default 200), LB_BENCH_WARMUP (default 30),
//!      LB_BENCH_W/H (default 1280x720).

use std::time::Instant;

use lb_pipeline::compositor::Framing;
use lb_pipeline::{Background, Compositor, Mask};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn synth_frame(w: usize, h: usize) -> Vec<u8> {
    let mut f = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            f[i] = (x * 255 / w) as u8;
            f[i + 1] = (y * 255 / h) as u8;
            f[i + 2] = 128;
            f[i + 3] = 255;
        }
    }
    f
}

// Soft-edged centered ellipse foreground mask, RVM-style frame-resolution f32.
fn synth_mask(w: usize, h: usize) -> Mask {
    let (cx, cy) = (w as f32 / 2.0, h as f32 * 0.55);
    let (rx, ry) = (w as f32 * 0.22, h as f32 * 0.42);
    let mut data = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f32 - cx) / rx;
            let dy = (y as f32 - cy) / ry;
            let d = (dx * dx + dy * dy).sqrt();
            data[y * w + x] = (1.5 - d).clamp(0.0, 1.0);
        }
    }
    Mask {
        data,
        width: w as u32,
        height: h as u32,
    }
}

#[allow(clippy::too_many_arguments)]
fn bench(
    label: &str,
    comp: &mut Compositor,
    pristine: &[u8],
    w: u32,
    h: u32,
    mask: &Mask,
    bg: &Background,
    framing: Option<Framing>,
    warmup: usize,
    frames: usize,
) {
    let mut frame = pristine.to_vec();
    for _ in 0..warmup {
        frame.copy_from_slice(pristine);
        comp.composite(&mut frame, w, h, mask, bg, framing).unwrap();
    }
    let mut times = Vec::with_capacity(frames);
    for _ in 0..frames {
        frame.copy_from_slice(pristine);
        let t = Instant::now();
        comp.composite(&mut frame, w, h, mask, bg, framing).unwrap();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let p50 = times[times.len() / 2];
    let max = times[times.len() - 1];
    println!("[composite_bench] {label:<28} avg={avg:6.2} p50={p50:6.2} max={max:6.2} ms");
}

fn main() {
    let w = env_usize("LB_BENCH_W", 1280);
    let h = env_usize("LB_BENCH_H", 720);
    let frames = env_usize("LB_BENCH_FRAMES", 200);
    let warmup = env_usize("LB_BENCH_WARMUP", 30);

    let pristine = synth_frame(w, h);
    let mask = synth_mask(w, h);

    // Source image larger than frame to exercise scale-to-cover (cached).
    let img = synth_frame(1920, 1080);
    let image_bg = Background::Image {
        rgba: img,
        width: 1920,
        height: 1080,
    };
    let blur_bg = Background::Blur {
        strength: Background::DEFAULT_BLUR_STRENGTH,
    };
    let framing = Framing {
        src_anchor_x: w as f32 / 2.0,
        src_anchor_y: h as f32 * 0.25,
        dst_anchor_x: w as f32 / 2.0,
        dst_anchor_y: h as f32 * 0.25,
        zoom: 1.15,
    };

    println!("[composite_bench] {w}x{h} warmup={warmup} frames={frames}");
    let mut comp = Compositor::new();
    bench(
        "image, no framing",
        &mut comp,
        &pristine,
        w as u32,
        h as u32,
        &mask,
        &image_bg,
        None,
        warmup,
        frames,
    );
    bench(
        "image, framing ON",
        &mut comp,
        &pristine,
        w as u32,
        h as u32,
        &mask,
        &image_bg,
        Some(framing),
        warmup,
        frames,
    );
    bench(
        "blur, no framing",
        &mut comp,
        &pristine,
        w as u32,
        h as u32,
        &mask,
        &blur_bg,
        None,
        warmup,
        frames,
    );
    bench(
        "blur, framing ON",
        &mut comp,
        &pristine,
        w as u32,
        h as u32,
        &mask,
        &blur_bg,
        Some(framing),
        warmup,
        frames,
    );
}
