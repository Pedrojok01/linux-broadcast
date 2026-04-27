use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{Receiver, Sender};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use std::sync::{Arc, Mutex};

use crate::compositor::{Background, Compositor};
use crate::segmenter::Segmenter;
use crate::temporal::MaskSmoother;

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Source v4l2 device, e.g. `/dev/video0`.
    pub source_device: String,
    /// Sink v4l2loopback device, e.g. `/dev/video10`.
    pub sink_device: String,
    pub width: u32,
    pub height: u32,
    pub framerate: u32,
    pub background: Background,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            source_device: "/dev/video0".to_string(),
            sink_device: "/dev/video10".to_string(),
            width: 1280,
            height: 720,
            framerate: 30,
            background: Background::Blur,
        }
    }
}

/// Live commands from the GUI thread to the running pipeline.
#[derive(Debug, Clone)]
pub enum Command {
    SetBackground(Background),
    Stop,
}

pub struct Pipeline {
    pipeline: gst::Pipeline,
    cmd_tx: Sender<Command>,
}

impl Pipeline {
    /// Build and start the pipeline. Returns once `Playing` is reached.
    pub fn start(cfg: PipelineConfig, model_onnx: &'static [u8]) -> Result<Self> {
        gst::init().context("gst::init")?;

        let pipeline = gst::Pipeline::new();

        // --- Source branch: v4l2src → videoscale → videoconvert → appsink ---
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
        // Caps between videoconvert and appsink: RGBA at our target size,
        // framerate left open so the camera's native rate flows through.
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
        // Belt-and-braces: also set sync=false explicitly post-build, since
        // some gstreamer-rs builder properties don't always round-trip.
        appsink.set_sync(false);
        // Caps emitted by appsrc — same RGBA frames, but with an *explicit*
        // framerate so the downstream videoconvert can match input/output
        // fps (otherwise gst_video_converter_new asserts fps_n equality).
        let appsrc_caps = gst::Caps::builder("video/x-raw")
            .field("format", "RGBA")
            .field("width", cfg.width as i32)
            .field("height", cfg.height as i32)
            .field("framerate", gst::Fraction::new(cfg.framerate as i32, 1))
            .build();

        // --- Sink branch: appsrc → videoconvert → v4l2sink ---
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
            // async=false → don't block pipeline in PAUSED waiting for a
            // preroll buffer from appsrc. Without this, the pipeline
            // deadlocks: v4l2sink waits for preroll, appsrc can't push
            // because new_sample never fires while we're in PAUSED, and
            // new_sample is what's *supposed* to push the first buffer.
            .property("async", false)
            .build()?;

        pipeline.add_many([
            &src,
            &src_scale,
            &src_convert,
            &src_capsfilter,
            appsink.upcast_ref(),
            appsrc.upcast_ref(),
            &sink_convert,
            &sink_capsfilter,
            &v4l2sink,
        ])?;
        gst::Element::link_many([
            &src,
            &src_scale,
            &src_convert,
            &src_capsfilter,
            appsink.upcast_ref(),
        ])?;
        gst::Element::link_many([appsrc.upcast_ref(), &sink_convert, &sink_capsfilter, &v4l2sink])?;

        // Shared state between the appsink callback, command receiver, and main thread.
        let segmenter = Arc::new(Mutex::new(
            Segmenter::from_bytes(model_onnx).context("load segmentation model")?,
        ));
        let compositor = Arc::new(Mutex::new(Compositor::new()));
        let smoother = Arc::new(Mutex::new(MaskSmoother::new(0.7)));
        let background = Arc::new(Mutex::new(cfg.background.clone()));

        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<Command>();

        // Command-handling: drain the channel inside the appsink callback so we
        // don't need a second thread.
        let bg_for_cmd = Arc::clone(&background);
        let cmd_drain = move |rx: &Receiver<Command>| {
            while let Ok(cmd) = rx.try_recv() {
                match cmd {
                    Command::SetBackground(bg) => {
                        if let Ok(mut slot) = bg_for_cmd.lock() {
                            *slot = bg;
                        }
                    }
                    Command::Stop => {
                        // Stop is observed by the main run-loop checking the bus.
                    }
                }
            }
        };

        let appsrc_clone = appsrc.clone();
        let segmenter_cb = Arc::clone(&segmenter);
        let compositor_cb = Arc::clone(&compositor);
        let smoother_cb = Arc::clone(&smoother);
        let bg_cb = Arc::clone(&background);
        let frame_w = cfg.width;
        let frame_h = cfg.height;
        let fps = cfg.framerate;
        let mut frame_idx: u64 = 0;
        let cmd_rx_for_cb = cmd_rx.clone();

        appsink.set_callbacks(
            gst_app::AppSinkCallbacks::builder()
                .new_preroll(|sink| {
                    log::info!("new_preroll fired");
                    let _ = sink.pull_preroll();
                    Ok(gst::FlowSuccess::Ok)
                })
                .new_sample(move |sink| {
                    cmd_drain(&cmd_rx_for_cb);

                    let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                    let buffer = sample.buffer().ok_or(gst::FlowError::Error)?;
                    let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
                    let in_bytes = map.as_slice();

                    // Copy to a writable scratch we can composite into.
                    let mut frame_rgba = in_bytes.to_vec();

                    // Segmentation.
                    let mask_res = {
                        let mut seg = segmenter_cb.lock().unwrap();
                        seg.segment(&frame_rgba, frame_w as usize, frame_h as usize)
                    };
                    let mut mask = match mask_res {
                        Ok(m) => m,
                        Err(e) => {
                            log::error!("segment: {e:#}");
                            return Err(gst::FlowError::Error);
                        }
                    };

                    // Temporal smoothing.
                    smoother_cb.lock().unwrap().smooth(&mut mask);

                    // Composite.
                    let bg = bg_cb.lock().unwrap().clone();
                    if let Err(e) = compositor_cb
                        .lock()
                        .unwrap()
                        .composite(&mut frame_rgba, frame_w, frame_h, &mask, &bg)
                    {
                        log::error!("composite: {e:#}");
                        return Err(gst::FlowError::Error);
                    }

                    // Push out.
                    let mut out = gst::Buffer::with_size(frame_rgba.len())
                        .map_err(|_| gst::FlowError::Error)?;
                    {
                        let out_mut = out.get_mut().ok_or(gst::FlowError::Error)?;
                        let pts = gst::ClockTime::from_nseconds(
                            frame_idx * 1_000_000_000 / fps as u64,
                        );
                        out_mut.set_pts(pts);
                        out_mut.set_duration(gst::ClockTime::from_nseconds(
                            1_000_000_000 / fps as u64,
                        ));
                        let mut wmap = out_mut
                            .map_writable()
                            .map_err(|_| gst::FlowError::Error)?;
                        wmap.copy_from_slice(&frame_rgba);
                    }
                    frame_idx += 1;
                    if frame_idx == 1 || frame_idx.is_multiple_of(fps as u64) {
                        log::info!(
                            "pushed frame #{} ({}x{} RGBA)",
                            frame_idx, frame_w, frame_h
                        );
                    }
                    appsrc_clone.push_buffer(out).map_err(|_| gst::FlowError::Error)?;

                    Ok(gst::FlowSuccess::Ok)
                })
                .build(),
        );

        // Install a sync handler so element errors / warnings are visible
        // immediately, not just when the main loop calls `iter_timed`.
        if let Some(bus) = pipeline.bus() {
            bus.set_sync_handler(|_, msg| {
                match msg.view() {
                    gst::MessageView::Error(err) => log::error!(
                        "GST {} :: {} (debug: {})",
                        err.src().map(|s| s.path_string().to_string()).unwrap_or_default(),
                        err.error(),
                        err.debug().unwrap_or_default(),
                    ),
                    gst::MessageView::Warning(w) => log::warn!(
                        "GST {} :: {}",
                        w.src().map(|s| s.path_string().to_string()).unwrap_or_default(),
                        w.error(),
                    ),
                    _ => {}
                }
                gst::BusSyncReply::Pass
            });
        }

        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| {
                let bus_errs = drain_bus_errors(&pipeline);
                if bus_errs.is_empty() {
                    anyhow!("set pipeline to Playing: {e}")
                } else {
                    anyhow!("set pipeline to Playing: {e}\n{}", bus_errs.join("\n"))
                }
            })?;

        // Sanity log of negotiated caps.
        if let Some(pad) = appsink.static_pad("sink") {
            if let Some(caps) = pad.current_caps() {
                if let Some(info) = gst_video::VideoInfo::from_caps(&caps).ok() {
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
        }

        Ok(Self { pipeline, cmd_tx })
    }

    pub fn cmd_sender(&self) -> Sender<Command> {
        self.cmd_tx.clone()
    }

    /// Block until EOS or error on the bus.
    pub fn run_until_done(&self) -> Result<()> {
        let bus = self
            .pipeline
            .bus()
            .ok_or_else(|| anyhow!("pipeline has no bus"))?;
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
        let _ = self.pipeline.set_state(gst::State::Null);
        let _ = self.cmd_tx.send(Command::Stop);
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

fn drain_bus_errors(pipeline: &gst::Pipeline) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(bus) = pipeline.bus() {
        while let Some(msg) = bus.pop() {
            if let gst::MessageView::Error(err) = msg.view() {
                out.push(format!(
                    "  ↳ {} :: {} (debug: {})",
                    err.src().map(|s| s.path_string().to_string()).unwrap_or_default(),
                    err.error(),
                    err.debug().unwrap_or_default(),
                ));
            }
        }
    }
    out
}
