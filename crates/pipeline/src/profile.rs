//! Lightweight per-stage timing for the feeder's Live loop.
//!
//! Active only when `LB_PROFILE` is set in the environment. Accumulates
//! per-stage durations over a rolling window and prints one summary line
//! to stderr every `WINDOW_FRAMES` composited frames: effective output
//! FPS plus mean/max for each pipeline stage. When disabled, `record`
//! returns on the first branch so the only cost is the timestamps the
//! caller already takes.

use std::time::{Duration, Instant};

/// Frames per reporting window. ~90 frames is 3–6 s of Live at the rates
/// we see, enough to average out scheduler jitter without making the
/// output sparse.
const WINDOW_FRAMES: u64 = 90;

#[derive(Default)]
struct StageAcc {
    sum: Duration,
    max: Duration,
}

impl StageAcc {
    fn add(&mut self, d: Duration) {
        self.sum += d;
        if d > self.max {
            self.max = d;
        }
    }
    fn reset(&mut self) {
        self.sum = Duration::ZERO;
        self.max = Duration::ZERO;
    }
    fn mean_ms(&self, n: u64) -> f64 {
        if n == 0 {
            0.0
        } else {
            self.sum.as_secs_f64() * 1000.0 / n as f64
        }
    }
    fn max_ms(&self) -> f64 {
        self.max.as_secs_f64() * 1000.0
    }
}

pub(crate) struct Profiler {
    pub enabled: bool,
    interval: u64,
    win_start: Instant,
    frames: u64,
    infer_frames: u64,
    pull: StageAcc,
    segment: StageAcc,
    composite: StageAcc,
    push: StageAcc,
    total: StageAcc,
}

impl Profiler {
    pub fn new(interval: u64) -> Self {
        Self {
            enabled: std::env::var_os("LB_PROFILE").is_some(),
            interval,
            win_start: Instant::now(),
            frames: 0,
            infer_frames: 0,
            pull: StageAcc::default(),
            segment: StageAcc::default(),
            composite: StageAcc::default(),
            push: StageAcc::default(),
            total: StageAcc::default(),
        }
    }

    /// Fold one Live tick into the current window. `segment` is `None` on
    /// ticks that reused the cached mask (no inference ran). Emits a
    /// summary line once the window fills.
    pub fn record(
        &mut self,
        pull: Duration,
        segment: Option<Duration>,
        composite: Duration,
        push: Duration,
        total: Duration,
    ) {
        if !self.enabled {
            return;
        }
        self.frames += 1;
        self.pull.add(pull);
        if let Some(s) = segment {
            self.infer_frames += 1;
            self.segment.add(s);
        }
        self.composite.add(composite);
        self.push.add(push);
        self.total.add(total);
        if self.frames >= WINDOW_FRAMES {
            self.report();
        }
    }

    fn report(&mut self) {
        let wall = self.win_start.elapsed().as_secs_f64();
        let fps = if wall > 0.0 {
            self.frames as f64 / wall
        } else {
            0.0
        };
        // infer_rate: how many inferences ran per second (the cap on mask freshness).
        let infer_rate = if wall > 0.0 {
            self.infer_frames as f64 / wall
        } else {
            0.0
        };
        eprintln!(
            "[lb-profile] interval={} fps={:.1} infer/s={:.1} | total={:.1}/{:.1} pull={:.2}/{:.2} segment={:.1}/{:.1}(n={}) composite={:.2}/{:.2} push={:.2}/{:.2} | {}f/{:.2}s (mean/max ms)",
            self.interval,
            fps,
            infer_rate,
            self.total.mean_ms(self.frames),
            self.total.max_ms(),
            self.pull.mean_ms(self.frames),
            self.pull.max_ms(),
            self.segment.mean_ms(self.infer_frames),
            self.segment.max_ms(),
            self.infer_frames,
            self.composite.mean_ms(self.frames),
            self.composite.max_ms(),
            self.push.mean_ms(self.frames),
            self.push.max_ms(),
            self.frames,
            wall,
        );
        self.frames = 0;
        self.infer_frames = 0;
        self.pull.reset();
        self.segment.reset();
        self.composite.reset();
        self.push.reset();
        self.total.reset();
        self.win_start = Instant::now();
    }
}
