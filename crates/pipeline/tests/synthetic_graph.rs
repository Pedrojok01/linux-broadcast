//! Synthetic-graph integration tests.
//!
//! These tests build a `Pipeline` against substitute source/sink
//! builders so the pipeline can run without `/dev/video0` or
//! `v4l2loopback`. The source becomes `videotestsrc → videoconvert →
//! capsfilter(RGBA, WxH) → appsink` and the sink becomes `appsrc →
//! capsfilter(RGBA, WxH, fps) → appsink` — the feeder still does
//! segmentation (skipped for `Background::None`) and pushes to the sink
//! exactly the way it would in production.
//!
//! Requires `libgstreamer1.0-dev` + `gstreamer1.0-plugins-good` for
//! `videotestsrc`. No v4l2loopback needed.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::Result;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use lb_pipeline::pipeline::{SinkBuilder, SourceBuilder, SourceCodec};
use lb_pipeline::{Background, Command, Pipeline, PipelineConfig, PipelineState};

// Real model bytes — these tests only run with `Background::None`, so
// the feeder short-circuits before invoking the segmenter, but
// Segmenter::from_bytes still runs at thread spawn (loads the ONNX
// session) and needs valid bytes. We embed the same models the app
// embeds; the path is workspace-relative.
const MODEL_MULTICLASS_ONNX: &[u8] = include_bytes!("../../../models/selfie_multiclass.onnx");
const MODEL_RVM_ONNX: &[u8] = include_bytes!("../../../models/rvm.onnx");

fn cfg() -> PipelineConfig {
    PipelineConfig {
        source_device: "synthetic".to_string(),
        sink_device: "synthetic".to_string(),
        width: 320,
        height: 240,
        framerate: 30,
        background: Background::None,
        model: lb_pipeline::ModelKind::SelfieMulticlass,
        preview_tx: None,
        framing: false,
    }
}

/// `videotestsrc → videoconvert → capsfilter(RGBA, WxH) → appsink`.
/// The codec arg is ignored — `videotestsrc` always produces raw frames.
fn synthetic_source() -> SourceBuilder {
    Arc::new(|cfg: &PipelineConfig, _codec: SourceCodec| source_graph(cfg, -1, None))
}

/// The synthetic source graph. `num_buffers` > 0 makes it EOS after that
/// many frames; `framerate` pins the source rate (default 30 fps).
fn source_graph(
    cfg: &PipelineConfig,
    num_buffers: i32,
    framerate: Option<gst::Fraction>,
) -> Result<(gst::Pipeline, gst_app::AppSink)> {
    let pipeline = gst::Pipeline::new();
    let src = gst::ElementFactory::make("videotestsrc")
        .property_from_str("pattern", "ball")
        .property("is-live", true)
        .property("num-buffers", num_buffers)
        .build()?;
    let convert = gst::ElementFactory::make("videoconvert").build()?;
    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGBA")
        .field("width", cfg.width as i32)
        .field("height", cfg.height as i32)
        .field_if_some("framerate", framerate)
        .build();
    let capsfilter = gst::ElementFactory::make("capsfilter")
        .property("caps", &caps)
        .build()?;
    let appsink = gst_app::AppSink::builder()
        .caps(&caps)
        .max_buffers(2)
        .drop(true)
        .sync(false)
        .build();
    pipeline.add_many([
        &src,
        &convert,
        &capsfilter,
        appsink.upcast_ref::<gst::Element>(),
    ])?;
    gst::Element::link_many([
        &src,
        &convert,
        &capsfilter,
        appsink.upcast_ref::<gst::Element>(),
    ])?;
    Ok((pipeline, appsink))
}

/// `appsrc → capsfilter(RGBA, WxH, fps) → appsink`. The terminal appsink
/// is what the test pulls from to verify frames arrive end-to-end. We
/// stash a clone of it on the side via a closure-captured Mutex so the
/// caller can pull samples after `Pipeline::start_with_builders` returns.
fn synthetic_sink(captured_sink: Arc<Mutex<Option<gst_app::AppSink>>>) -> SinkBuilder {
    Arc::new(
        move |cfg: &PipelineConfig| -> Result<(gst::Pipeline, gst_app::AppSrc)> {
            let pipeline = gst::Pipeline::new();
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
            let appsink = gst_app::AppSink::builder()
                .caps(&appsrc_caps)
                .max_buffers(8)
                .drop(false)
                .sync(false)
                .build();
            pipeline.add_many([
                appsrc.upcast_ref::<gst::Element>(),
                appsink.upcast_ref::<gst::Element>(),
            ])?;
            gst::Element::link_many([
                appsrc.upcast_ref::<gst::Element>(),
                appsink.upcast_ref::<gst::Element>(),
            ])?;
            *captured_sink.lock().unwrap() = Some(appsink);
            Ok((pipeline, appsrc))
        },
    )
}

/// Like [`synthetic_source`] but fails when asked for MJPEG — emulates a
/// raw-only camera so the feeder's codec policy must fall back to raw.
fn raw_only_source() -> SourceBuilder {
    let inner = synthetic_source();
    Arc::new(move |cfg: &PipelineConfig, codec: SourceCodec| {
        if codec == SourceCodec::Mjpeg {
            anyhow::bail!("synthetic camera does not support MJPEG");
        }
        inner(cfg, codec)
    })
}

/// Emulates a camera that fails once while Live: the first graph comes from
/// `faulty`, every later one is healthy, so the feeder can recover.
fn failing_once_source(
    faulty: fn(&PipelineConfig) -> Result<(gst::Pipeline, gst_app::AppSink)>,
) -> SourceBuilder {
    let built = AtomicBool::new(false);
    Arc::new(move |cfg: &PipelineConfig, _codec: SourceCodec| {
        if built.swap(true, Ordering::SeqCst) {
            source_graph(cfg, -1, None)
        } else {
            faulty(cfg)
        }
    })
}

fn start_pipeline(cfg: PipelineConfig) -> (Pipeline, Arc<Mutex<Option<gst_app::AppSink>>>) {
    start_pipeline_with_source(cfg, synthetic_source())
}

fn start_pipeline_with_source(
    cfg: PipelineConfig,
    source: SourceBuilder,
) -> (Pipeline, Arc<Mutex<Option<gst_app::AppSink>>>) {
    let captured = Arc::new(Mutex::new(None));
    let p = Pipeline::start_with_builders(
        cfg,
        MODEL_MULTICLASS_ONNX,
        MODEL_RVM_ONNX,
        source,
        synthetic_sink(captured.clone()),
    )
    .expect("start_with_builders");
    (p, captured)
}

fn pull_frames_for(sink: &gst_app::AppSink, duration: Duration) -> usize {
    let deadline = Instant::now() + duration;
    let mut count = 0;
    while Instant::now() < deadline {
        if sink
            .try_pull_sample(gst::ClockTime::from_mseconds(50))
            .is_some()
        {
            count += 1;
        }
    }
    count
}

#[test]
fn pipeline_emits_frames_with_videotestsrc() {
    let mut c = cfg();
    c.background = Background::None;
    let (p, captured) = start_pipeline(c);

    // The GUI-preview heartbeat lights the synthetic camera. The
    // activation debounce is 2 s; pull for slightly longer to ensure we
    // land in Live and see real source frames flowing through, not just
    // idle re-pushes.
    p.cmd_sender()
        .send(Command::SetGuiPreviewActive(true))
        .unwrap();
    let sink = captured
        .lock()
        .unwrap()
        .clone()
        .expect("sink builder must have been called");

    // Drain idle re-pushes during the activation debounce window.
    let _ = pull_frames_for(&sink, Duration::from_millis(2200));
    // The feeder's outer select! is paced by IDLE_TICK = 100ms, so even
    // with a 30 fps source we expect ~10 fps end-to-end. Measure over
    // 2 s and require at least 12 frames — a comfortable margin over
    // the idle-push rate (5 fps) so a bug that fails to actually
    // transition into Live would be caught.
    let live_count = pull_frames_for(&sink, Duration::from_millis(2000));
    assert!(
        live_count >= 12,
        "expected ≥12 Live frames in 2s, got {live_count}",
    );

    drop(p);
}

#[test]
fn lazy_state_walks_idle_to_live_to_idle() {
    let c = cfg();
    let (p, _captured) = start_pipeline(c);

    // Initial state: Idle.
    assert!(matches!(p.state(), PipelineState::Idle));

    // Assert demand → state should reach Live within activation
    // debounce + slack.
    p.cmd_sender()
        .send(Command::SetGuiPreviewActive(true))
        .unwrap();
    let live_deadline = Instant::now() + Duration::from_secs(4);
    while Instant::now() < live_deadline {
        if matches!(p.state(), PipelineState::Live { .. }) {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        matches!(p.state(), PipelineState::Live { .. }),
        "expected Live within 4s",
    );

    // Drop demand → must return to Idle within deactivation debounce +
    // slack.
    p.cmd_sender()
        .send(Command::SetGuiPreviewActive(false))
        .unwrap();
    let idle_deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < idle_deadline {
        if matches!(p.state(), PipelineState::Idle) {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(matches!(p.state(), PipelineState::Idle));

    drop(p);
}

/// Dropping a Live pipeline (what the GUI does on Stop and Quit) must
/// join the feeder promptly instead of hanging on the source graph.
#[test]
fn shutdown_after_live_is_clean() {
    let c = cfg();
    let (p, captured) = start_pipeline(c);
    p.cmd_sender()
        .send(Command::SetGuiPreviewActive(true))
        .unwrap();
    let sink = captured.lock().unwrap().clone().unwrap();
    // Drive into Live so feeder is actively running source + composite.
    let _ = pull_frames_for(&sink, Duration::from_millis(2200));
    assert!(matches!(p.state(), PipelineState::Live { .. }));

    // Drop must complete promptly (Pipeline::Drop has a 2s join timeout
    // on the feeder; we give the whole thing 5s headroom).
    let start = Instant::now();
    drop(p);
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "Pipeline::drop took {:?} — feeder failed to join",
        start.elapsed(),
    );
}

#[test]
fn set_background_no_frame_gap() {
    // While Live with a synthetic source, swap backgrounds and verify
    // the sink keeps emitting frames with no extended gap. (Replaces
    // the original "source pipeline pointer unchanged" check, which
    // would have coupled to internals.)
    let mut c = cfg();
    c.background = Background::None;
    let (p, captured) = start_pipeline(c);
    p.cmd_sender()
        .send(Command::SetGuiPreviewActive(true))
        .unwrap();
    let sink = captured.lock().unwrap().clone().unwrap();

    // Wait through activation debounce.
    let _ = pull_frames_for(&sink, Duration::from_millis(2200));

    // Now run a sequence of background swaps and count frames pulled.
    // With None we skip segmentation entirely; with Blur we run
    // segmentation per frame. Both modes must keep producing frames at
    // roughly the configured framerate. The 1200ms window leaves slack
    // for shared-CPU CI runners — the first Blur tick pays for cold
    // ORT model warmup and a cold instruction cache.
    for bg in [
        Background::None,
        Background::Blur { strength: 0.5 },
        Background::None,
    ] {
        p.cmd_sender().send(Command::SetBackground(bg)).unwrap();
        let n = pull_frames_for(&sink, Duration::from_millis(1200));
        // The point is that frames keep flowing across the swap — a real
        // stall yields 0–1. The exact count is at the mercy of a loaded
        // shared CI runner (RVM CPU inference + cold warmup), so keep the
        // bar low enough to not flake while still catching a true gap.
        assert!(
            n >= 3,
            "expected ≥3 frames during 1200ms in mode swap, got {n}",
        );
    }

    drop(p);
}

/// When the camera can't negotiate MJPEG, the feeder's codec policy must
/// fall back to raw and still reach Live. Uses a source builder that
/// errors on `SourceCodec::Mjpeg` and succeeds on `Raw`.
#[test]
fn source_codec_falls_back_to_raw_when_mjpeg_unavailable() {
    let c = cfg();
    let (p, captured) = start_pipeline_with_source(c, raw_only_source());
    p.cmd_sender()
        .send(Command::SetGuiPreviewActive(true))
        .unwrap();
    let sink = captured.lock().unwrap().clone().unwrap();

    // Through the activation debounce, then assert Live was reached via
    // the raw fallback and frames actually flow.
    let _ = pull_frames_for(&sink, Duration::from_millis(2200));
    assert!(
        matches!(p.state(), PipelineState::Live { .. }),
        "expected Live via raw fallback after MJPEG failed",
    );
    let live_count = pull_frames_for(&sink, Duration::from_millis(2000));
    assert!(
        live_count >= 12,
        "expected ≥12 Live frames in 2s via raw fallback, got {live_count}",
    );

    drop(p);
}

fn wait_for(p: &Pipeline, timeout: Duration, pred: impl Fn(&PipelineState) -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if pred(&p.state()) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

/// A source that fails while Live must drop the feeder to Idle (no spin on
/// a dead appsink), then rebuild once the 2 s backoff and the 2 s activation
/// debounce pass, with frames flowing again.
fn assert_recovers_from_source_failure(source: SourceBuilder, failure_within: Duration) {
    let (p, _captured) = start_pipeline_with_source(cfg(), source);
    p.cmd_sender()
        .send(Command::SetGuiPreviewActive(true))
        .unwrap();

    let live = |s: &PipelineState| matches!(s, PipelineState::Live { .. });
    assert!(wait_for(&p, Duration::from_secs(4), live), "expected Live");
    assert!(
        wait_for(&p, failure_within, |s| matches!(s, PipelineState::Idle)),
        "expected Idle after the source failed",
    );
    // The fps meter restarts per engagement, so a measured rate proves the
    // rebuilt graph delivers frames.
    assert!(
        wait_for(&p, Duration::from_secs(7), |s| matches!(
            s,
            PipelineState::Live { fps: Some(f), .. } if *f > 5.0
        )),
        "expected Live with frames flowing after the backoff",
    );

    drop(p);
}

#[test]
fn recovers_when_source_hits_eos_while_live() {
    // EOS after ~1 s of frames, as on an unplug or USB reset.
    assert_recovers_from_source_failure(
        failing_once_source(|cfg| source_graph(cfg, 30, None)),
        Duration::from_secs(3),
    );
}

#[test]
fn recovers_when_source_stalls_while_live() {
    // One frame, then nothing for 10 s with no error or EOS: only the
    // stall watchdog can catch this.
    assert_recovers_from_source_failure(
        failing_once_source(|cfg| source_graph(cfg, -1, Some(gst::Fraction::new(1, 10)))),
        Duration::from_secs(5),
    );
}

/// Preview frames are forwarded only while the GUI preview is active
/// (window visible AND toggle on). A hidden window must not cost a
/// full-frame clone per pushed frame.
#[test]
fn preview_frames_follow_gui_preview_signal() {
    let (tx, rx) = crossbeam_channel::bounded(2);
    let mut c = cfg();
    c.preview_tx = Some(tx);
    let (p, _captured) = start_pipeline(c);
    let drain = || std::iter::from_fn(|| rx.try_recv().ok()).count();

    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(drain(), 0, "no preview frames while the GUI is inactive");

    p.cmd_sender()
        .send(Command::SetGuiPreviewActive(true))
        .unwrap();
    std::thread::sleep(Duration::from_millis(500));
    assert!(drain() > 0, "preview frames flow while the GUI is active");

    p.cmd_sender()
        .send(Command::SetGuiPreviewActive(false))
        .unwrap();
    std::thread::sleep(Duration::from_millis(300));
    drain();
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(drain(), 0, "preview frames stop once the GUI is inactive");

    drop(p);
}

fn wait_for_failure(p: &Pipeline, timeout: Duration) -> Option<String> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Some(f) = p.failure() {
            return Some(f);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    None
}

/// A config the graphs can't run must fail `start`, not panic the feeder.
#[test]
fn start_rejects_unrunnable_config() {
    for bad in [
        PipelineConfig {
            framerate: 0,
            ..cfg()
        },
        PipelineConfig { height: 0, ..cfg() },
        PipelineConfig {
            width: 321,
            ..cfg()
        },
    ] {
        let captured = Arc::new(Mutex::new(None));
        let res = Pipeline::start_with_builders(
            bad,
            MODEL_MULTICLASS_ONNX,
            MODEL_RVM_ONNX,
            synthetic_source(),
            synthetic_sink(captured),
        );
        assert!(res.is_err());
    }
}

/// A sink-graph error (device gone, negotiation failure) must reach the
/// caller instead of only the log.
#[test]
fn sink_error_is_reported_as_failure() {
    let (p, captured) = start_pipeline(cfg());
    assert!(p.failure().is_none());
    let sink = captured.lock().unwrap().clone().unwrap();
    sink.post_message(
        gst::message::Error::builder(gst::CoreError::Failed, "synthetic sink failure")
            .src(&sink)
            .build(),
    )
    .unwrap();
    let failure = wait_for_failure(&p, Duration::from_secs(1)).expect("failure reported");
    assert!(failure.contains("synthetic sink failure"), "{failure}");
}

/// A feeder panic must be reported instead of leaving the GUI on Idle.
#[test]
fn feeder_panic_is_reported_as_failure() {
    let panicking: SourceBuilder = Arc::new(|_: &PipelineConfig, _: SourceCodec| {
        panic!("synthetic feeder panic");
    });
    let (p, _captured) = start_pipeline_with_source(cfg(), panicking);
    p.cmd_sender()
        .send(Command::SetGuiPreviewActive(true))
        .unwrap();
    let failure = wait_for_failure(&p, Duration::from_secs(4)).expect("failure reported");
    assert!(failure.contains("synthetic feeder panic"), "{failure}");
}
