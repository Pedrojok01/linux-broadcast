//! Lazy producer state machine and feeder thread.
//!
//! Owns:
//! - the `Segmenter`, `Compositor`, and `MaskSmoother`,
//! - the source GStreamer graph (`v4l2src → … → appsink`) which is
//!   built when going Live and torn down when going Idle, releasing
//!   the physical camera so the LED actually goes off,
//! - the `appsrc` of the always-running sink graph; we push composited
//!   buffers into it while Live, and push nothing while Idle.
//!
//! The feeder runs in one thread. It ticks at ~33 ms while Live (paced
//! by `try_pull_sample` on the source appsink), and ~100 ms while Idle
//! (a plain sleep — no frames flow downstream when nobody's reading).
//!
//! Transitions are debounced to absorb two real-world quirks:
//! - Browsers (Chrome / Firefox) briefly open `/dev/video10` to probe
//!   capabilities (ENUM_FMT) without intending to stream. A short
//!   activation debounce hides those probes from the camera LED.
//! - In-call camera-switcher pickers close-and-reopen the device when
//!   the user toggles between cameras. A short deactivation debounce
//!   prevents the pipeline from thrashing.

use std::collections::HashSet;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossbeam_channel::{select, tick, Receiver, Sender};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;

use crate::compositor::{Background, Compositor, Framing};
use crate::consumer_watch::Consumer;
use crate::framing::{BBoxSmoother, FG_ZOOM, TOP_HEADROOM};
use crate::pipeline::{
    log_negotiated_input, Command, PipelineConfig, PipelineState, PreviewFrame, SourceBuilder,
    NS_PER_SEC,
};
use crate::segmenter::{ModelKind, Segmenter};
use crate::temporal::{MaskSmoother, DEFAULT_ALPHA, RVM_ALPHA};

/// How long a consumer must remain present before we light the camera.
/// Sized to reject browser capability-probes (typically <500 ms) without
/// adding noticeable latency to a real Meet/Zoom open.
pub(crate) const ACTIVATION_DEBOUNCE: Duration = Duration::from_millis(2_000);

/// How long after the last consumer leaves before we release the camera.
/// Absorbs the close-and-reopen flicker of in-call camera-switcher
/// pickers (Chrome's "Camera (LinuxBroadcast)" dropdown).
pub(crate) const DEACTIVATION_DEBOUNCE: Duration = Duration::from_millis(3_000);

/// Idle-loop tick: how often we check for incoming control / watcher
/// events when no frames are flowing. 100 ms is fast enough for the
/// state machine to feel responsive without burning power.
const IDLE_TICK: Duration = Duration::from_millis(100);

/// While Idle (camera released), the feeder still pushes a still-frame
/// to the sink at this rate so any consumer that opens `/dev/video10`
/// sees a steady stream. Picked to be slow enough that the memcpy cost
/// is negligible, fast enough that no WebRTC consumer trips its
/// "no frames" timeout (typically 1–3 s).
const IDLE_PUSH_INTERVAL: Duration = Duration::from_millis(200);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum State {
    Idle,
    Activating { since: Instant },
    Live,
    Deactivating { since: Instant },
}

/// Inputs that drive the state machine, computed once per tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Demand {
    pub consumers_present: bool,
    /// True while the GUI's preview pane is visible. Treated as a
    /// synthetic consumer so opening the preview lights the camera even
    /// when no real client is attached to `/dev/video10`.
    pub gui_preview_active: bool,
}
impl Demand {
    pub(crate) fn signal(&self) -> bool {
        self.consumers_present || self.gui_preview_active
    }
}

/// Pure state-transition function. No side effects. Extracted from
/// `Feeder::step_state` so the debounce logic can be exercised on a
/// synthetic `Instant` timeline without booting GStreamer.
pub(crate) fn next_state(state: State, demand: Demand, now: Instant) -> State {
    let signal = demand.signal();
    match (state, signal) {
        (State::Idle, true) => State::Activating { since: now },
        (State::Idle, false) => State::Idle,

        (State::Activating { since }, true) => {
            if now.duration_since(since) >= ACTIVATION_DEBOUNCE {
                State::Live
            } else {
                State::Activating { since }
            }
        }
        (State::Activating { .. }, false) => State::Idle,

        (State::Live, true) => State::Live,
        (State::Live, false) => State::Deactivating { since: now },

        (State::Deactivating { .. }, true) => State::Live,
        (State::Deactivating { since }, false) => {
            if now.duration_since(since) >= DEACTIVATION_DEBOUNCE {
                State::Idle
            } else {
                State::Deactivating { since }
            }
        }
    }
}

/// Spawn the feeder thread. Returns the join handle plus a stop-flag the
/// caller (Pipeline) can flip to ask the thread to wind down promptly;
/// the thread also exits naturally on `Command::Stop` and on cmd-channel
/// disconnect.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_feeder(
    cfg: PipelineConfig,
    binary_onnx: &'static [u8],
    multiclass_onnx: &'static [u8],
    rvm_onnx: &'static [u8],
    sink_appsrc: gst_app::AppSrc,
    cmd_rx: Receiver<Command>,
    watcher_rx: Receiver<Vec<Consumer>>,
    state_pub: Arc<Mutex<PipelineState>>,
    stop_flag: Arc<AtomicBool>,
    source_builder: SourceBuilder,
) -> Result<std::thread::JoinHandle<()>> {
    let segmenter = Segmenter::from_bytes(
        cfg.model,
        match cfg.model {
            ModelKind::SelfieBinary => binary_onnx,
            ModelKind::SelfieMulticlass => multiclass_onnx,
            ModelKind::Rvm => rvm_onnx,
        },
    )
    .context("load segmentation model")?;

    let cfg_for_thread = cfg.clone();
    let handle = std::thread::Builder::new()
        .name("lb-feeder".into())
        .spawn(move || {
            let mut feeder = Feeder {
                cfg: cfg_for_thread,
                segmenter,
                compositor: Compositor::new(),
                smoother: MaskSmoother::new(match cfg.model {
                    ModelKind::Rvm => RVM_ALPHA,
                    _ => DEFAULT_ALPHA,
                }),
                bbox_smoother: BBoxSmoother::new(),
                framing_enabled: cfg.framing,
                background: cfg.background.clone(),
                preview_tx: cfg.preview_tx.clone(),
                sink_appsrc,
                source: None,
                source_builder,
                consumers: Vec::new(),
                preview_enabled: true,
                gui_preview_active: false,
                state: State::Idle,
                state_pub,
                frame_idx: 0,
                last_state_log: None,
                start_failure_count: 0,
                start_failure_backoff_until: None,
                last_published_pids: HashSet::new(),
                idle_frame: None,
                last_idle_push: None,
            };
            feeder.run(cmd_rx, watcher_rx, stop_flag);
        })
        .context("spawn lb-feeder")?;
    Ok(handle)
}

struct Feeder {
    cfg: PipelineConfig,
    segmenter: Segmenter,
    compositor: Compositor,
    smoother: MaskSmoother,
    /// Smoothed virtual-PTZ crop driver. Held even when `framing_enabled`
    /// is false so toggling at runtime doesn't drop history more than the
    /// explicit `reset()` we run on disable.
    bbox_smoother: BBoxSmoother,
    /// Toggle for the auto-frame stage. When true, segmentation runs on
    /// every frame regardless of background mode (the smoother needs a
    /// mask to compute the bbox), and the compositor's crop+rescale stage
    /// runs after compositing.
    framing_enabled: bool,
    background: Background,
    preview_tx: Option<Sender<PreviewFrame>>,
    sink_appsrc: gst_app::AppSrc,
    /// Builder for the source GST graph. Production wires
    /// `pipeline::build_source_pipeline`; tests substitute a synthetic
    /// builder. Called on every Live engagement.
    source_builder: SourceBuilder,
    /// Source GST graph + its appsink; only `Some` while Live.
    source: Option<(gst::Pipeline, gst_app::AppSink)>,
    consumers: Vec<Consumer>,
    /// Mirrors the GUI's "Show preview" toggle. When false, frames are
    /// not forwarded to `preview_tx` (skips a 1280×720 RGBA clone per
    /// frame). Defaults to true so any standalone consumer of the
    /// pipeline (tests, future headless tooling) gets frames by default.
    preview_enabled: bool,
    /// True while the GUI's preview pane is visible (window visible AND
    /// `show_preview` toggle on). Acts as a synthetic consumer in the
    /// lazy state machine. Defaults to false so headless launches and
    /// any non-GUI caller don't keep the camera lit by accident.
    gui_preview_active: bool,
    state: State,
    state_pub: Arc<Mutex<PipelineState>>,
    frame_idx: u64,
    last_state_log: Option<&'static str>,
    /// Number of consecutive `start_source` failures. Drives an
    /// exponential backoff so a wedged camera doesn't produce a 10 Hz
    /// log flood. Reset to 0 on success.
    start_failure_count: u32,
    /// While `Some(t)` and `now < t`, the state machine refuses to leave
    /// Idle even if demand is asserted. Cleared on natural expiry or on
    /// a successful `start_source`.
    start_failure_backoff_until: Option<Instant>,
    /// PIDs included in the most recent `state_pub` write. Used to skip
    /// redundant publishes when nothing the GUI cares about has changed.
    last_published_pids: HashSet<u32>,
    /// Last successfully composited frame. Re-pushed at `IDLE_PUSH_INTERVAL`
    /// while the lazy state is not Live, so consumers of `/dev/video10`
    /// always see *something* instead of an empty stream — making the
    /// virtual cam robust against portal-mediated consumers our `/proc`
    /// watcher can't observe (PipeWire-backed Slack / Chromium / …).
    /// `None` only on the very first tick, before any composite has run;
    /// `pump_idle_frame` falls back to a freshly-built black RGBA frame
    /// in that case.
    idle_frame: Option<Vec<u8>>,
    /// Last time `pump_idle_frame` actually pushed a buffer. Used to
    /// throttle to `IDLE_PUSH_INTERVAL` regardless of how often the
    /// outer select wakes up.
    last_idle_push: Option<Instant>,
}

impl Feeder {
    fn run(
        &mut self,
        cmd_rx: Receiver<Command>,
        watcher_rx: Receiver<Vec<Consumer>>,
        stop_flag: Arc<AtomicBool>,
    ) {
        // Idle ticker — fires regularly so the state machine wakes even
        // when no commands or consumer events arrive (we still need to
        // observe debounce windows expiring).
        let idler = tick(IDLE_TICK);

        while !stop_flag.load(Ordering::Relaxed) {
            // 1. Multiplex control inputs without blocking on any one
            //    channel for too long — we want to come back and pump
            //    frames from the source appsink quickly when Live.
            select! {
                recv(cmd_rx) -> msg => match msg {
                    Ok(Command::Stop) => return self.shutdown(),
                    Ok(cmd) => self.handle_command(cmd),
                    Err(_) => return self.shutdown(),
                },
                recv(watcher_rx) -> msg => match msg {
                    Ok(c) => self.consumers = c,
                    Err(_) => {
                        // Watcher exited unexpectedly — keep the pipeline
                        // running but we won't transition out of Idle
                        // without force-on / preview signals.
                        log::warn!("consumer watcher disconnected");
                    }
                },
                recv(idler) -> _ => {}
            }

            // 2. Step the state machine. Transitions may build / drop
            //    the source GST graph as a side-effect.
            let now = Instant::now();
            self.step_state(now);

            // 3. Frame pumping. Live: pull a real frame from the source
            //    appsink, segment + composite, push to the sink. Otherwise:
            //    re-push the last live frame (or black) at a low rate so
            //    `/dev/video10` always has output for any consumer that
            //    opens it — the lazy state still controls whether
            //    `/dev/video0` is held, but virtual-cam visibility is no
            //    longer gated on consumer detection.
            if matches!(self.state, State::Live) {
                self.pump_one_frame();
            } else {
                self.pump_idle_frame();
            }
        }
        self.shutdown();
    }

    fn handle_command(&mut self, cmd: Command) {
        match cmd {
            Command::SetBackground(bg) => self.background = bg,
            Command::SetPreviewEnabled(v) => self.preview_enabled = v,
            Command::SetGuiPreviewActive(v) => self.gui_preview_active = v,
            Command::SetFraming(v) => {
                self.framing_enabled = v;
                if !v {
                    // Drop the smoother state so the next enable starts at
                    // the live bbox instead of panning from a stale rect.
                    self.bbox_smoother.reset();
                }
            }
            Command::Stop => unreachable!("handled in run()"),
        }
    }

    /// Step the debounced state machine and execute side effects
    /// (build/drop the source pipeline, publish state). Idempotent
    /// inside a single state.
    fn step_state(&mut self, now: Instant) {
        // Backoff gate: a recent `start_source` failure pins us in Idle
        // until the timer expires, regardless of demand. Without this,
        // a wedged camera produced a Deactivating ↔ Live retry loop at
        // tick rate.
        if let Some(t) = self.start_failure_backoff_until {
            if now < t {
                let prev = self.state;
                self.state = State::Idle;
                self.publish_state(prev);
                return;
            }
            self.start_failure_backoff_until = None;
        }

        let demand = Demand {
            consumers_present: !self.consumers.is_empty(),
            gui_preview_active: self.gui_preview_active,
        };

        let next = next_state(self.state, demand, now);

        // Side effects only fire when the *category* of state changes
        // (Idle/Activating/Live/Deactivating), not on every tick.
        let entered_live = matches!(next, State::Live) && !matches!(self.state, State::Live);
        let exited_live = !matches!(next, State::Live) && matches!(self.state, State::Live);

        if entered_live {
            match self.start_source() {
                Ok(()) => {
                    self.start_failure_count = 0;
                    self.start_failure_backoff_until = None;
                }
                Err(e) => {
                    log::error!("could not start source pipeline: {e:#}");
                    // Exponential backoff: 2, 4, 8, 16, 32 s capped.
                    self.start_failure_count = self.start_failure_count.saturating_add(1).min(5);
                    let delay = Duration::from_secs(2u64.pow(self.start_failure_count));
                    self.start_failure_backoff_until = Some(now + delay);
                    log::warn!(
                        "source-pipeline retry suppressed for {:?} (failure #{})",
                        delay,
                        self.start_failure_count,
                    );
                    let prev = self.state;
                    self.state = State::Idle;
                    self.publish_state(prev);
                    return;
                }
            }
        }
        if exited_live {
            self.stop_source();
            // RVM / smoother carry inter-frame state; reset so the next
            // engagement starts clean instead of with stale gradients.
            // The bg plate is dropped too — lighting and camera
            // position may have changed during the idle window, and
            // the EMA reconverges within ~1 s of being live again.
            self.segmenter.reset();
            self.smoother.reset();
            self.bbox_smoother.reset();
            self.compositor.reset_bg_plate();
        }

        let prev = self.state;
        self.state = next;
        self.publish_state(prev);
    }

    fn start_source(&mut self) -> Result<()> {
        let (pipeline, appsink) = (self.source_builder)(&self.cfg)?;
        pipeline
            .set_state(gst::State::Playing)
            .context("source pipeline → Playing")?;
        log::info!(
            "source camera engaged ({} → composite → /dev/video10)",
            self.cfg.source_device
        );
        log_negotiated_input(&appsink);
        self.source = Some((pipeline, appsink));
        Ok(())
    }

    fn stop_source(&mut self) {
        if let Some((pipeline, _)) = self.source.take() {
            let _ = pipeline.set_state(gst::State::Null);
            // `set_state(Null)` returns `Async` for any pipeline that
            // owns a real device — it queues the state change but the
            // v4l2src fd may still be live for a few frames. Without
            // this synchronous wait, dropping `pipeline` here lets the
            // next `start_source` race the kernel and get EBUSY on
            // /dev/video0. 1 s is generous; v4l2 release is normally
            // a few ms.
            let _ = pipeline.state(gst::ClockTime::from_seconds(1));
            log::info!("source camera released ({})", self.cfg.source_device);
        }
    }

    fn shutdown(&mut self) {
        self.stop_source();
        // Push EOS into the sink so any consumers see a clean end.
        let _ = self.sink_appsrc.end_of_stream();
    }

    /// Refresh the public `PipelineState` and log a transition line.
    /// Skips the mutex write and clone when neither the state-machine
    /// state nor the consumer set has changed since the last publish —
    /// avoids ~30 redundant lock+clone ops/sec while Live.
    fn publish_state(&mut self, prev: State) {
        let state_changed = std::mem::discriminant(&prev) != std::mem::discriminant(&self.state);
        let pids_now: HashSet<u32> = self.consumers.iter().map(|c| c.pid).collect();
        let consumers_changed = pids_now != self.last_published_pids;
        if !state_changed && !consumers_changed {
            return;
        }

        let label: &'static str = match self.state {
            State::Idle => "idle",
            State::Activating { .. } => "activating",
            State::Live => "live",
            State::Deactivating { .. } => "deactivating",
        };
        if self.last_state_log != Some(label) {
            let consumer_summary = self
                .consumers
                .iter()
                .map(|c| format!("{}({})", c.name, c.pid))
                .collect::<Vec<_>>()
                .join(",");
            log::info!(
                "lazy state → {label} (consumers=[{consumer_summary}], gui_preview={})",
                self.gui_preview_active,
            );
            self.last_state_log = Some(label);
        }
        let public = match self.state {
            State::Live => PipelineState::Live {
                consumers: self.consumers.clone(),
            },
            // Activating/Deactivating both look "Idle-ish" to the GUI; we
            // only flip the public state when frames are actually flowing.
            _ => PipelineState::Idle,
        };
        if let Ok(mut slot) = self.state_pub.lock() {
            *slot = public;
        }
        self.last_published_pids = pids_now;
    }

    /// Pull one sample from the source appsink (with a short timeout),
    /// run segmentation + composite, push the result into the sink
    /// appsrc, and stash a copy as the new idle still-frame. No-op if
    /// the pull times out (camera lag).
    fn pump_one_frame(&mut self) {
        let appsink = match self.source.as_ref() {
            Some((_, s)) => s.clone(),
            None => return,
        };

        let sample = match appsink.try_pull_sample(gst::ClockTime::from_mseconds(33)) {
            Some(s) => s,
            None => return,
        };
        let buffer = match sample.buffer() {
            Some(b) => b,
            None => return,
        };
        let map = match buffer.map_readable() {
            Ok(m) => m,
            Err(_) => return,
        };

        let mut frame_rgba = map.as_slice().to_vec();
        let frame_w = self.cfg.width;
        let frame_h = self.cfg.height;
        let bg = self.background.clone();

        // Auto-frame is a no-op without a background plane to fill the
        // strip vacated by the shifted silhouette, so we tie it to a
        // non-None background and keep the original true-passthrough
        // short-circuit otherwise.
        if !matches!(bg, Background::None) {
            match self
                .segmenter
                .segment(&frame_rgba, frame_w as usize, frame_h as usize)
            {
                Ok(mut mask) => {
                    self.smoother.smooth(&mut mask.data);
                    let framing = if self.framing_enabled {
                        self.bbox_smoother.update(&mask).map(|anchor| {
                            // Place head at TOP_HEADROOM, but never higher
                            // than what's needed to keep the source bottom
                            // mapped to (at least) the output bottom — at
                            // low zoom the silhouette doesn't fully cover
                            // the frame, and a fixed top anchor would leave
                            // a band of the background plane at the bottom.
                            let src_y = anchor.top_y_norm * frame_h as f32;
                            let h = frame_h as f32;
                            let min_dst_y = h - (h - src_y) * FG_ZOOM;
                            let dst_y = (h * TOP_HEADROOM).max(min_dst_y);
                            let f = Framing {
                                src_anchor_x: anchor.cx_norm * frame_w as f32,
                                src_anchor_y: src_y,
                                dst_anchor_x: frame_w as f32 * 0.5,
                                dst_anchor_y: dst_y,
                                zoom: FG_ZOOM,
                            };
                            if self.frame_idx == 1
                                || self.frame_idx % (self.cfg.framerate as u64 * 2) == 0
                            {
                                log::debug!(
                                    "auto-frame: cx={:.3} top_y={:.3} zoom={:.2}",
                                    anchor.cx_norm,
                                    anchor.top_y_norm,
                                    f.zoom,
                                );
                            }
                            f
                        })
                    } else {
                        None
                    };
                    if let Err(e) = self.compositor.composite(
                        &mut frame_rgba,
                        frame_w,
                        frame_h,
                        &mask,
                        &bg,
                        framing,
                    ) {
                        log::error!("composite: {e:#}");
                        return;
                    }
                }
                Err(e) => {
                    log::error!("segment: {e:#}");
                    return;
                }
            }
        } else {
            // Passthrough: keep recurrent state clean for the next
            // non-None engagement.
            self.smoother.reset();
            self.segmenter.reset();
        }

        if self.preview_enabled {
            if let Some(tx) = &self.preview_tx {
                // try_send so we silently drop when the GUI lags.
                let _ = tx.try_send(PreviewFrame {
                    width: frame_w,
                    height: frame_h,
                    rgba: frame_rgba.clone(),
                });
            }
        }

        let pushed = self.push_to_sink(&frame_rgba);
        if pushed && (self.frame_idx == 1 || self.frame_idx % (self.cfg.framerate as u64) == 0) {
            log::debug!(
                "live frame #{} ({}x{} RGBA) → /dev/video sink",
                self.frame_idx,
                frame_w,
                frame_h,
            );
        }

        // Save the freshly composited frame so the Idle path can keep
        // re-pushing it after the camera is released. Doing this *after*
        // push_to_sink avoids paying the clone cost when the push fails
        // (e.g. sink in the middle of a state change).
        if pushed {
            self.idle_frame = Some(frame_rgba);
            // Reset the idle-push throttle so the very next tick after
            // we drop out of Live re-pushes immediately — no visible
            // gap on the consumer side.
            self.last_idle_push = None;
        }
    }

    /// Push a still-frame to the sink while the lazy state is not Live.
    /// Throttled to `IDLE_PUSH_INTERVAL`. Uses the last live composite
    /// when available, falls back to a freshly-built black RGBA frame
    /// (cached in `self.idle_frame` so subsequent ticks reuse it).
    fn pump_idle_frame(&mut self) {
        let now = Instant::now();
        if let Some(last) = self.last_idle_push {
            if now.duration_since(last) < IDLE_PUSH_INTERVAL {
                return;
            }
        }

        if self.idle_frame.is_none() {
            let pixels = (self.cfg.width as usize) * (self.cfg.height as usize);
            let mut black = Vec::with_capacity(pixels * 4);
            for _ in 0..pixels {
                black.extend_from_slice(&[0, 0, 0, 255]);
            }
            self.idle_frame = Some(black);
        }

        // Take the buffer out, push, put it back. Avoids cloning the
        // ~3.7 MB Vec every tick while still letting `push_to_sink`
        // borrow `&mut self`.
        let frame = self.idle_frame.take().expect("idle_frame just populated");
        let _ = self.push_to_sink(&frame);
        self.idle_frame = Some(frame);
        self.last_idle_push = Some(now);
    }

    /// Wrap an RGBA frame in a fresh GStreamer buffer with the next PTS,
    /// push it to the sink appsrc, bump `frame_idx`. Returns false if
    /// allocation, mapping, or the push itself failed — caller decides
    /// whether to retry / log.
    fn push_to_sink(&mut self, rgba: &[u8]) -> bool {
        let fps = self.cfg.framerate as u64;
        let mut out = match gst::Buffer::with_size(rgba.len()) {
            Ok(b) => b,
            Err(_) => return false,
        };
        {
            let out_mut = match out.get_mut() {
                Some(m) => m,
                None => return false,
            };
            let pts = gst::ClockTime::from_nseconds(self.frame_idx * NS_PER_SEC / fps);
            out_mut.set_pts(pts);
            out_mut.set_duration(gst::ClockTime::from_nseconds(NS_PER_SEC / fps));
            let mut wmap = match out_mut.map_writable() {
                Ok(w) => w,
                Err(_) => return false,
            };
            wmap.copy_from_slice(rgba);
        }
        self.frame_idx += 1;
        if let Err(e) = self.sink_appsrc.push_buffer(out) {
            log::warn!("sink appsrc push_buffer: {e}");
            return false;
        }
        true
    }
}

/// Helper used by tests + the orchestrator to dedupe consumer-set
/// updates when polling produces an identical PID set.
#[allow(dead_code)]
pub(crate) fn pids(consumers: &[Consumer]) -> HashSet<u32> {
    consumers.iter().map(|c| c.pid).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demand(consumers: bool, preview: bool) -> Demand {
        Demand {
            consumers_present: consumers,
            gui_preview_active: preview,
        }
    }

    /// Drive `next_state` through a sequence of (Δt, Demand) steps,
    /// returning the final state. Useful for asserting end-state across
    /// a synthetic timeline without booting any GStreamer machinery.
    fn run(start: State, steps: &[(Duration, Demand)]) -> State {
        let mut now = Instant::now();
        let mut state = start;
        for (dt, d) in steps {
            now += *dt;
            state = next_state(state, *d, now);
        }
        state
    }

    #[test]
    fn idle_with_no_demand_stays_idle() {
        let off = demand(false, false);
        let final_state = run(State::Idle, &[(Duration::from_secs(1), off); 10]);
        assert!(matches!(final_state, State::Idle));
    }

    #[test]
    fn brief_consumer_under_2s_never_lights_camera() {
        // Demand on for 1.5 s, then off. Must never reach Live.
        let on = demand(true, false);
        let off = demand(false, false);
        let mut state = State::Idle;
        let mut now = Instant::now();
        // 1.5 seconds of demand asserted.
        let end_on = now + Duration::from_millis(1500);
        while now < end_on {
            now += Duration::from_millis(50);
            state = next_state(state, on, now);
            assert!(
                !matches!(state, State::Live),
                "must not reach Live within activation debounce"
            );
        }
        // Then demand off — must transition straight back to Idle.
        for _ in 0..20 {
            now += Duration::from_millis(50);
            state = next_state(state, off, now);
            assert!(!matches!(state, State::Live));
        }
        assert!(matches!(state, State::Idle));
    }

    #[test]
    fn consumer_held_2s_reaches_live() {
        // With demand asserted continuously, we should reach Live at
        // exactly the activation-debounce boundary.
        let on = demand(true, false);
        let t0 = Instant::now();
        let mut state = State::Idle;

        // Just before the boundary: still Activating.
        let before = t0 + ACTIVATION_DEBOUNCE - Duration::from_millis(50);
        state = next_state(state, on, t0);
        state = next_state(state, on, before);
        assert!(matches!(state, State::Activating { .. }));

        // At the boundary: Live.
        let at = t0 + ACTIVATION_DEBOUNCE;
        state = next_state(state, on, at);
        assert!(matches!(state, State::Live));
    }

    #[test]
    fn live_then_no_demand_for_3s_returns_idle() {
        let on = demand(true, false);
        let off = demand(false, false);
        let t0 = Instant::now();

        // 3 s off → Idle.
        let mut s = State::Live;
        s = next_state(s, off, t0);
        assert!(matches!(s, State::Deactivating { .. }));
        s = next_state(s, off, t0 + DEACTIVATION_DEBOUNCE);
        assert!(matches!(s, State::Idle));

        // 2.5 s off then on → must stay Live.
        let t1 = Instant::now();
        let mut s2 = State::Live;
        s2 = next_state(s2, off, t1);
        s2 = next_state(s2, off, t1 + Duration::from_millis(2500));
        assert!(matches!(s2, State::Deactivating { .. }));
        s2 = next_state(s2, on, t1 + Duration::from_millis(2600));
        assert!(matches!(s2, State::Live));
    }

    #[test]
    fn flicker_during_deactivation_stays_live() {
        // Live, demand toggles off/on/off/on at 1 s each — never crosses
        // the 3 s deactivation boundary, so must never leave Live.
        let on = demand(true, false);
        let off = demand(false, false);
        let t0 = Instant::now();
        let mut s = State::Live;
        s = next_state(s, off, t0);
        s = next_state(s, on, t0 + Duration::from_secs(1));
        assert!(matches!(s, State::Live));
        s = next_state(s, off, t0 + Duration::from_secs(2));
        s = next_state(s, on, t0 + Duration::from_secs(3));
        assert!(matches!(s, State::Live));
    }

    #[test]
    fn gui_preview_alone_holds_live() {
        // Validates the 0c wiring: preview pane visible counts as a
        // synthetic consumer in the demand signal.
        let preview = demand(false, true);
        let t0 = Instant::now();
        let mut s = State::Idle;
        s = next_state(s, preview, t0);
        s = next_state(s, preview, t0 + ACTIVATION_DEBOUNCE);
        assert!(matches!(s, State::Live));
    }
}
