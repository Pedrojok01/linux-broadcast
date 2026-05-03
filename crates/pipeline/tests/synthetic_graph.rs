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

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use lb_pipeline::pipeline::{SinkBuilder, SourceBuilder};
use lb_pipeline::{Background, Command, Pipeline, PipelineConfig, PipelineState};

// Empty model bytes — these tests only run with `Background::None`, so
// the feeder short-circuits before invoking the segmenter. The
// Segmenter::from_bytes call still happens at thread spawn (loads the
// ONNX session), so we need real bytes there.
//
// To avoid a 450 KB blob in the test crate, we embed the same model the
// app embeds. The path is workspace-relative.
const MODEL_BINARY_ONNX: &[u8] = include_bytes!("../../../models/selfie_segmenter.onnx");
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
        model: lb_pipeline::ModelKind::SelfieBinary,
        preview_tx: None,
        framing: false,
    }
}

/// `videotestsrc → videoconvert → capsfilter(RGBA, WxH) → appsink`.
fn synthetic_source() -> SourceBuilder {
    Arc::new(
        |cfg: &PipelineConfig| -> Result<(gst::Pipeline, gst_app::AppSink)> {
            let pipeline = gst::Pipeline::new();
            let src = gst::ElementFactory::make("videotestsrc")
                .property_from_str("pattern", "ball")
                .property("is-live", true)
                .build()?;
            let convert = gst::ElementFactory::make("videoconvert").build()?;
            let caps = gst::Caps::builder("video/x-raw")
                .field("format", "RGBA")
                .field("width", cfg.width as i32)
                .field("height", cfg.height as i32)
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
        },
    )
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

fn start_pipeline(cfg: PipelineConfig) -> (Pipeline, Arc<Mutex<Option<gst_app::AppSink>>>) {
    let captured = Arc::new(Mutex::new(None));
    let p = Pipeline::start_with_builders(
        cfg,
        MODEL_BINARY_ONNX,
        MODEL_MULTICLASS_ONNX,
        MODEL_RVM_ONNX,
        synthetic_source(),
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

    p.stop();
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

    p.stop();
}

/// Regression for the shutdown path the GUI's `App::shutdown_cleanup`
/// drives on Quit: stop() then drop, with no panic and no hang. Used to
/// trigger NVIDIA's EGL Wayland abort when followed by eframe's GL
/// teardown — that's now bypassed via `std::process::exit(0)` in
/// `App::on_exit`, but the underlying pipeline shutdown sequence must
/// itself stay clean. Repeated stop() calls must be idempotent because
/// `Pipeline::stop` runs once, then `drop` re-asserts the same state.
#[test]
fn shutdown_after_live_is_clean_and_idempotent() {
    let c = cfg();
    let (p, captured) = start_pipeline(c);
    p.cmd_sender()
        .send(Command::SetGuiPreviewActive(true))
        .unwrap();
    let sink = captured.lock().unwrap().clone().unwrap();
    // Drive into Live so feeder is actively running source + composite.
    let _ = pull_frames_for(&sink, Duration::from_millis(2200));
    assert!(matches!(p.state(), PipelineState::Live { .. }));

    // Repeated stops mirror App::shutdown_cleanup → Drop, which calls
    // stop() twice in effect. Must not panic, must not deadlock.
    p.stop();
    p.stop();

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

    // Now run a sequence of background swaps over ~600ms and count
    // frames pulled. With None we skip segmentation entirely; with Blur
    // we run segmentation per frame. Both modes must keep producing
    // frames at roughly the configured framerate (30fps → ≥10 frames in
    // 600ms is a generous lower bound).
    for bg in [
        Background::None,
        Background::Blur { strength: 0.5 },
        Background::None,
    ] {
        p.cmd_sender().send(Command::SetBackground(bg)).unwrap();
        let n = pull_frames_for(&sink, Duration::from_millis(600));
        assert!(
            n >= 5,
            "expected ≥5 frames during 600ms in mode swap, got {n}",
        );
    }

    p.stop();
}
