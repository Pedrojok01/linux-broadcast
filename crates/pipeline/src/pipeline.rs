//! Public pipeline facade.
//!
//! The pipeline is split into:
//! - a **sink** GStreamer graph (`appsrc → videoconvert → v4l2sink`) that
//!   stays in PLAYING for the entire lifetime of the process. It owns
//!   `/dev/video10` so the virtual cam never blinks out of conferencing
//!   apps' device lists.
//! - a **source** GStreamer graph (`v4l2src → … → appsink`) that is
//!   built on demand when a real consumer attaches and torn down again
//!   on idle. This is the only branch that actually opens
//!   `/dev/video0`, so dropping it lets the kernel close the camera and
//!   the LED goes off.
//!
//! Both graphs are glued together by a single feeder thread (in
//! [`crate::lazy`]) which also owns the segmenter / compositor / EMA
//! smoother and the lazy-mode state machine. Callers never see this
//! split — the public [`Pipeline`] handle behaves like a single object.

use anyhow::{anyhow, Context, Result};
use crossbeam_channel::Sender;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

use crate::compositor::Background;
use crate::consumer_watch::{Consumer, Watcher};
use crate::lazy::spawn_feeder;
use crate::segmenter::ModelKind;

pub(crate) const NS_PER_SEC: u64 = 1_000_000_000;

/// How often the consumer watcher polls `/proc/*/fd`. Tuned against the
/// activation debounce: we want at least one poll cycle inside the
/// debounce window so a real consumer is observed before the timeout.
const WATCH_POLL_INTERVAL: Duration = Duration::from_millis(800);

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Source v4l2 device, e.g. `/dev/video0`.
    pub source_device: String,
    /// Sink v4l2loopback device, e.g. `/dev/video10`. The `.deb` ships
    /// conffiles in `/etc/modprobe.d/` + `/etc/modules-load.d/` that
    /// guarantee this exists at boot; source builds need a manual
    /// `modprobe v4l2loopback`.
    pub sink_device: String,
    pub width: u32,
    pub height: u32,
    pub framerate: u32,
    pub background: Background,
    /// Which segmentation model to load.
    pub model: ModelKind,
    /// If `Some`, each composited frame is forwarded (RGBA) to this
    /// sender for the GUI's preview pane. Bounded; sends use `try_send`
    /// and silently drop when the GUI is slow.
    pub preview_tx: Option<Sender<PreviewFrame>>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            source_device: "/dev/video0".to_string(),
            sink_device: "/dev/video10".to_string(),
            width: 1280,
            height: 720,
            framerate: 30,
            background: Background::Blur {
                strength: Background::DEFAULT_BLUR_STRENGTH,
            },
            model: ModelKind::SelfieBinary,
            preview_tx: None,
        }
    }
}

/// A composited RGBA frame published to the GUI for preview.
#[derive(Debug, Clone)]
pub struct PreviewFrame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Live commands from the GUI thread to the running pipeline.
#[derive(Debug, Clone)]
pub enum Command {
    SetBackground(Background),
    /// Pin the camera on regardless of consumer count. Sidebar toggle.
    SetForceOn(bool),
    /// While false, the feeder skips forwarding frames to the GUI's
    /// preview channel, avoiding the per-frame RGBA clone. The
    /// downstream broadcast (`/dev/video10`) is unaffected.
    SetPreviewEnabled(bool),
    Stop,
}

/// Public pipeline state, polled by the GUI for the footer indicator.
#[derive(Debug, Clone, Default)]
pub enum PipelineState {
    /// Camera released. `/dev/video10` exists but no frames are flowing.
    #[default]
    Idle,
    /// Camera engaged; one or more consumers are reading
    /// `/dev/video10` (or the `force_on` / GUI preview heartbeat is
    /// asserting demand).
    Live { consumers: Vec<Consumer> },
}

pub struct Pipeline {
    /// Held only for its `Drop` side-effect (signalling its stop flag and
    /// joining the watcher thread). Listed first so Rust's
    /// declaration-order field drop runs the watcher's stop+join before
    /// the rest of the pipeline tears down.
    _watcher: Watcher,
    sink_pipeline: gst::Pipeline,
    cmd_tx: Sender<Command>,
    state: Arc<Mutex<PipelineState>>,
    feeder: Option<std::thread::JoinHandle<()>>,
    stop_flag: Arc<AtomicBool>,
}

impl Pipeline {
    /// Build the sink graph, start the consumer watcher and the feeder
    /// thread, and return. The physical camera is **not** opened until
    /// either a consumer attaches to `/dev/video10` or the caller asserts
    /// demand via [`Command::SetForceOn`].
    pub fn start(
        cfg: PipelineConfig,
        binary_onnx: &'static [u8],
        multiclass_onnx: &'static [u8],
        rvm_onnx: &'static [u8],
    ) -> Result<Self> {
        gst::init().context("gst::init")?;

        // 1. Sink graph (always-on owner of /dev/video10).
        let (sink_pipeline, sink_appsrc) = build_sink_pipeline(&cfg)?;
        install_bus_logger(&sink_pipeline, "sink");
        sink_pipeline.set_state(gst::State::Playing).map_err(|e| {
            let bus_errs = drain_bus_errors(&sink_pipeline);
            if bus_errs.is_empty() {
                anyhow!("set sink pipeline to Playing: {e}")
            } else {
                anyhow!("set sink pipeline to Playing: {e}\n{}", bus_errs.join("\n"))
            }
        })?;

        // 2. Channels.
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<Command>();
        let state = Arc::new(Mutex::new(PipelineState::default()));
        let stop_flag = Arc::new(AtomicBool::new(false));

        // 3. Consumer watcher — excludes our own PID so we don't see
        //    ourselves as a consumer (the v4l2sink holds /dev/video10
        //    open as the *producer*, but `/proc/<our pid>/fd` lists it
        //    too).
        let watcher = Watcher::start(
            cfg.sink_device.clone(),
            std::process::id(),
            WATCH_POLL_INTERVAL,
        );
        let watcher_rx = watcher.events().clone();

        // 4. Feeder thread.
        let feeder = spawn_feeder(
            cfg,
            binary_onnx,
            multiclass_onnx,
            rvm_onnx,
            sink_appsrc,
            cmd_rx,
            watcher_rx,
            Arc::clone(&state),
            Arc::clone(&stop_flag),
        )?;

        Ok(Self {
            sink_pipeline,
            cmd_tx,
            state,
            feeder: Some(feeder),
            stop_flag,
            _watcher: watcher,
        })
    }

    pub fn cmd_sender(&self) -> Sender<Command> {
        self.cmd_tx.clone()
    }

    /// Snapshot the current public state. Cheap; backed by an
    /// `Arc<Mutex<…>>` updated by the feeder on every state-machine
    /// transition.
    pub fn state(&self) -> PipelineState {
        self.state.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Block until EOS or error on the sink bus. Used by `--headless`.
    pub fn run_until_done(&self) -> Result<()> {
        let bus = self
            .sink_pipeline
            .bus()
            .ok_or_else(|| anyhow!("sink pipeline has no bus"))?;
        for msg in bus.iter_timed(gst::ClockTime::NONE) {
            match msg.view() {
                gst::MessageView::Eos(_) => break,
                gst::MessageView::Error(err) => {
                    return Err(anyhow!(
                        "{}: {} ({:?})",
                        err.src().map(|s| s.path_string()).unwrap_or_default(),
                        err.error(),
                        err.debug()
                    ));
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        let _ = self.cmd_tx.send(Command::Stop);
        let _ = self.sink_pipeline.set_state(gst::State::Null);
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        let _ = self.cmd_tx.send(Command::Stop);
        // Join the feeder *before* tearing the sink down so its final
        // `push_buffer` calls don't race a NULL appsrc and produce a
        // burst of "sink appsrc push_buffer" warnings.
        if let Some(h) = self.feeder.take() {
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let _ = h.join();
                let _ = tx.send(());
            });
            let _ = rx.recv_timeout(Duration::from_secs(2));
        }
        let _ = self.sink_pipeline.set_state(gst::State::Null);
    }
}

// ---------------------------------------------------------------------------
//  Graph builders — used by `Pipeline::start` (sink) and `lazy::Feeder`
//  (source, on every Live engagement).
// ---------------------------------------------------------------------------

/// Build the sink graph. The returned `AppSrc` is what the feeder pushes
/// composited frames into; the pipeline must be set to PLAYING by the
/// caller (we don't do it here so the caller can attach a bus handler
/// first).
pub(crate) fn build_sink_pipeline(
    cfg: &PipelineConfig,
) -> Result<(gst::Pipeline, gst_app::AppSrc)> {
    let pipeline = gst::Pipeline::new();

    // Caps emitted by appsrc — RGBA at our target size with an *explicit*
    // framerate so the downstream videoconvert can match input/output
    // fps (otherwise gst_video_converter_new asserts fps_n equality).
    let appsrc_caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGBA")
        .field("width", cfg.width as i32)
        .field("height", cfg.height as i32)
        .field("framerate", gst::Fraction::new(cfg.framerate as i32, 1))
        .build();
    let appsrc = gst_app::AppSrc::builder()
        .caps(&appsrc_caps)
        .format(gst::Format::Time)
        .is_live(true)
        .build();

    let sink_convert = gst::ElementFactory::make("videoconvert").build()?;
    let sink_caps = gst::Caps::builder("video/x-raw")
        .field("format", "YUY2") // most v4l2loopback consumers prefer YUY2
        .field("width", cfg.width as i32)
        .field("height", cfg.height as i32)
        .field("framerate", gst::Fraction::new(cfg.framerate as i32, 1))
        .build();
    let sink_capsfilter = gst::ElementFactory::make("capsfilter")
        .property("caps", &sink_caps)
        .build()?;
    let v4l2sink = gst::ElementFactory::make("v4l2sink")
        .property("device", &cfg.sink_device)
        .property("sync", false)
        // async=false → don't block in PAUSED waiting for a preroll
        // buffer. With lazy mode the appsrc may go long stretches with
        // no buffers at all (Idle); blocking on preroll would deadlock
        // the pipeline transition to PLAYING.
        .property("async", false)
        .build()?;

    pipeline.add_many([
        appsrc.upcast_ref(),
        &sink_convert,
        &sink_capsfilter,
        &v4l2sink,
    ])?;
    gst::Element::link_many([
        appsrc.upcast_ref(),
        &sink_convert,
        &sink_capsfilter,
        &v4l2sink,
    ])?;

    Ok((pipeline, appsrc))
}

/// Build the source graph. Returned `AppSink` is pulled by the feeder.
/// The caller is expected to set the pipeline to PLAYING immediately;
/// see `lazy::Feeder::start_source`.
pub(crate) fn build_source_pipeline(
    cfg: &PipelineConfig,
) -> Result<(gst::Pipeline, gst_app::AppSink)> {
    let pipeline = gst::Pipeline::new();

    // Let v4l2src negotiate whatever format the camera prefers (YUYV,
    // MJPEG-decoded, NV12, …). videoscale + videoconvert then bridge
    // the camera-native caps to our RGBA target. Constraining only at
    // the appsink end avoids 30/1-vs-30000/1001 framerate-fraction
    // mismatches that fail v4l2src negotiation.
    let src = gst::ElementFactory::make("v4l2src")
        .property("device", &cfg.source_device)
        .property("do-timestamp", true)
        .build()?;
    let src_scale = gst::ElementFactory::make("videoscale").build()?;
    let src_convert = gst::ElementFactory::make("videoconvert").build()?;
    let appsink_caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGBA")
        .field("width", cfg.width as i32)
        .field("height", cfg.height as i32)
        .build();
    let src_capsfilter = gst::ElementFactory::make("capsfilter")
        .property("caps", &appsink_caps)
        .build()?;
    let appsink = gst_app::AppSink::builder()
        .caps(&appsink_caps)
        .max_buffers(2)
        .drop(true)
        .sync(false)
        .enable_last_sample(false)
        .build();
    appsink.set_sync(false);

    pipeline.add_many([
        &src,
        &src_scale,
        &src_convert,
        &src_capsfilter,
        appsink.upcast_ref(),
    ])?;
    gst::Element::link_many([
        &src,
        &src_scale,
        &src_convert,
        &src_capsfilter,
        appsink.upcast_ref(),
    ])?;

    install_bus_logger(&pipeline, "source");

    Ok((pipeline, appsink))
}

/// Read negotiated input caps from a source-pipeline appsink (only
/// meaningful once the pipeline has reached PLAYING). Logs a one-shot
/// info line; cheap to call multiple times. The feeder calls this once
/// per Live engagement.
pub(crate) fn log_negotiated_input(appsink: &gst_app::AppSink) {
    let pad = match appsink.static_pad("sink") {
        Some(p) => p,
        None => return,
    };
    let caps = match pad.current_caps() {
        Some(c) => c,
        None => return,
    };
    if let Ok(info) = gst_video::VideoInfo::from_caps(&caps) {
        log::info!(
            "negotiated input: {}x{} @ {}/{} fps, format {:?}",
            info.width(),
            info.height(),
            info.fps().numer(),
            info.fps().denom(),
            info.format(),
        );
    }
}

fn install_bus_logger(pipeline: &gst::Pipeline, tag: &'static str) {
    if let Some(bus) = pipeline.bus() {
        bus.set_sync_handler(move |_, msg| {
            match msg.view() {
                gst::MessageView::Error(err) => log::error!(
                    "GST[{tag}] {} :: {} (debug: {})",
                    err.src()
                        .map(|s| s.path_string().to_string())
                        .unwrap_or_default(),
                    err.error(),
                    err.debug().unwrap_or_default(),
                ),
                gst::MessageView::Warning(w) => log::warn!(
                    "GST[{tag}] {} :: {}",
                    w.src()
                        .map(|s| s.path_string().to_string())
                        .unwrap_or_default(),
                    w.error(),
                ),
                _ => {}
            }
            gst::BusSyncReply::Pass
        });
    }
}

fn drain_bus_errors(pipeline: &gst::Pipeline) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(bus) = pipeline.bus() {
        while let Some(msg) = bus.pop() {
            if let gst::MessageView::Error(err) = msg.view() {
                out.push(format!(
                    "  ↳ {} :: {} (debug: {})",
                    err.src()
                        .map(|s| s.path_string().to_string())
                        .unwrap_or_default(),
                    err.error(),
                    err.debug().unwrap_or_default(),
                ));
            }
        }
    }
    out
}
