//! `eframe`/`egui` GUI for the app.
//!
//! Layout: a 320 px sidebar (camera picker, model picker, scene mode +
//! blur slider, library grid, settings, footer) next to a preview pane
//! that renders the latest `PreviewFrame` produced by the pipeline. The
//! preview channel is a 2-deep `crossbeam` queue with drop-old semantics
//! so the segmenter never blocks on a slow GUI.
//!
//! The egui app loop is also where high-level lifecycle decisions land:
//!
//! - **Synthetic-consumer heartbeat.** While the window is visible AND
//!   the *Show preview* toggle is on, every `update()` tick pushes a
//!   `Command::SetGuiPreviewActive(true)` (edge-triggered) so the lazy
//!   feeder treats the GUI as a consumer and keeps the camera lit. The
//!   signal is explicitly cleared the moment the window goes to the
//!   tray, so a tray-only instance lets the camera drop to Idle.
//! - **Close-button intercept.** `ViewportCommand::CancelClose` +
//!   `Visible(false)` turns the window's X into "hide to tray". Only the
//!   tray's `Quit` menu sets `quit_requested` and lets the close go
//!   through. This is the single source of "the user actually wants to
//!   exit".
//! - **Headless boot guard.** `App::new` polls for the sink device
//!   (default `/dev/video10`) for up to `HEADLESS_DEVICE_WAIT` before
//!   starting the pipeline, so an XDG autostart entry that fires before
//!   `systemd-modules-load.service` finishes still recovers cleanly.
//! - **Live config writeback.** Every persisted setting (mode, blur
//!   strength, model, image, source/sink, `start_on_login`,
//!   `show_preview`, `auto_frame`) is debounced through `Config::save`
//!   so `~/.config/linux-broadcast/config.toml` stays in sync with the
//!   GUI without per-keystroke writes.
//!
//! Footer surface mirrors `PipelineState`: `● Idle`, `● Standby (no
//! consumer)`, or `● LIVE → name(pid)` while a real consumer is reading.

use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use eframe::egui::{self, Color32, Margin, Rounding, Stroke, ViewportCommand};
use lb_pipeline::{Background, Command, Pipeline, PipelineConfig, PreviewFrame};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use lb_pipeline::PipelineState;

use crate::autostart;
use crate::backgrounds::{self, LibraryEntry};
use crate::cameras::{CameraEntry, enumerate};
use crate::config::{Config, Mode, Model};
use crate::theme::{self, color, control, radius, space};
use crate::tray::{Tray, TrayEvent};
use crate::{MODEL_BINARY_ONNX, MODEL_MULTICLASS_ONNX, MODEL_RVM_ONNX};

const INITIAL_WINDOW_SIZE: [f32; 2] = [1280.0, 850.0];
const MIN_WINDOW_SIZE: [f32; 2] = [820.0, 560.0];
/// Bounded preview channel capacity: drop old frames if the GUI lags so
/// the segmentation thread never blocks.
const PREVIEW_CHANNEL_CAP: usize = 2;
/// Repaint roughly every 33 ms (~30 Hz) while running. Frame arrival
/// already drives texture updates; this just guarantees overlay redraws.
const PREVIEW_REPAINT_MS: u64 = 33;
const THUMBNAIL_PX: u32 = 160;

/// Maximum time `run_headless` is willing to wait for the sink device to
/// appear before erroring out. Sized to absorb the worst-case race between
/// XDG autostart firing and `systemd-modules-load.service` finishing on a
/// cold boot, while still failing fast when the module genuinely isn't
/// loaded.
const HEADLESS_DEVICE_WAIT: Duration = Duration::from_secs(10);

fn wait_for_device(path: &str, timeout: Duration) -> bool {
    let p = Path::new(path);
    let start = Instant::now();
    while !p.exists() {
        if start.elapsed() >= timeout {
            return false;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    true
}

/// Read the compositor-reported minimized state. `unwrap_or(false)` is
/// intentional: pre-first-frame and on platforms that don't supply the
/// field, treat as "not minimized" so a fresh window paints rather than
/// silently going blank. egui has no `viewport.visible` analogue —
/// `Visible(false)` is a no-op on Wayland and on other platforms eframe
/// pauses redraws anyway, so minimized is the only meaningful "is the
/// surface on the user's screen" signal we get from the toolkit.
fn is_minimized(ctx: &egui::Context) -> bool {
    ctx.input(|i| i.viewport().minimized).unwrap_or(false)
}

/// Single entry point. `headless` collapses what used to be `run_gui` /
/// `run_headless` into one eframe loop:
/// - GUI mode: window visible.
/// - Headless mode: window starts hidden in the tray.
///
/// In both modes the pipeline auto-starts at launch so consumers
/// (Meet, browsers, OBS) immediately see `/dev/video10` as a CAPTURE
/// device. The lazy state machine still keeps the physical camera
/// (`/dev/video0`, LED) released until a real consumer reads.
///
/// The window's close button always hides (minimises to tray); only the
/// tray's Quit menu actually exits. This lets a single instance own
/// `/dev/video10` for the whole session, addressing the "I autostarted
/// it on login and now have no way to stop it" pain point.
pub fn run(headless: bool) -> Result<()> {
    let viewport = egui::ViewportBuilder::default()
        .with_inner_size(INITIAL_WINDOW_SIZE)
        .with_min_inner_size(MIN_WINDOW_SIZE)
        .with_title("LinuxBroadcast")
        // app_id is required on Wayland for the compositor to match the
        // window to its desktop entry / icon. Must mirror the .desktop
        // file we ship at packaging time.
        .with_app_id("LinuxBroadcast")
        .with_icon(crate::icon::build())
        // In headless mode, start hidden so the autostart never flashes
        // a window. The tray's Show menu (or the user picking
        // LinuxBroadcast from the launcher) brings it up later.
        .with_visible(!headless);
    let opts = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native(
        "linux-broadcast",
        opts,
        Box::new(move |cc| {
            theme::apply(&cc.egui_ctx);
            Ok(Box::new(App::new(cc, headless)))
        }),
    )
    .map_err(|e| anyhow!("eframe: {e}"))
}

fn pipeline_config_from(cfg: &Config, preview_tx: Option<Sender<PreviewFrame>>) -> PipelineConfig {
    PipelineConfig {
        source_device: cfg.source_device.clone(),
        sink_device: cfg.sink_device.clone(),
        width: cfg.width,
        height: cfg.height,
        framerate: cfg.framerate,
        background: build_background(cfg),
        model: cfg.model.into_kind(),
        preview_tx,
        framing: cfg.auto_frame,
    }
}

fn build_background(cfg: &Config) -> Background {
    match cfg.mode {
        Mode::None => Background::None,
        Mode::Blur => Background::Blur {
            strength: cfg.blur_strength,
        },
        Mode::Replace => match cfg.background_image_path() {
            Some(path) => match load_background_rgba(path) {
                Ok(bg) => bg,
                Err(e) => {
                    log::warn!("background image load failed ({e:#}); falling back to blur");
                    Background::Blur {
                        strength: cfg.blur_strength,
                    }
                }
            },
            None => Background::Blur {
                strength: cfg.blur_strength,
            },
        },
    }
}

fn load_background_rgba(path: &Path) -> Result<Background> {
    let img = image::open(path)?.to_rgba8();
    let (w, h) = (img.width(), img.height());
    Ok(Background::Image {
        rgba: img.into_raw(),
        width: w,
        height: h,
    })
}

struct App {
    cfg: Config,
    cameras: Vec<CameraEntry>,
    library: Vec<LibraryEntry>,
    thumbnails: HashMap<PathBuf, egui::TextureHandle>,
    pipeline: Option<Pipeline>,
    cmd_tx: Option<Sender<Command>>,
    preview_rx: Option<Receiver<PreviewFrame>>,
    preview_tex: Option<egui::TextureHandle>,
    last_preview_size: Option<[usize; 2]>,
    /// Cached snapshot of the pipeline's lazy-mode state (Idle / Live
    /// with consumer list). Refreshed every `update()` call from
    /// `Pipeline::state()`. Used by the footer.
    pipeline_state: PipelineState,
    error: Option<String>,
    /// System tray handle. Held for the whole process lifetime. `None`
    /// only if the OS has no working tray host (logged at install time);
    /// the rest of the app continues to work — Quit just becomes
    /// "close the window with the Quit menu in the GUI".
    tray: Option<Tray>,
    /// Set to true by the tray's Quit handler. Without this, our
    /// close-requested intercept would refuse the close and hide instead.
    quit_requested: bool,
    /// Headless launches start with the window hidden. If the tray fails
    /// to install, the user has no way to interact, so we promote the
    /// window to visible on the first frame. Consumed (cleared) once.
    force_unhide_on_first_frame: bool,
    /// One-shot belt-and-braces for headless cold-start on Wayland:
    /// `with_visible(false)` set on the ViewportBuilder is best-effort
    /// (xdg-shell has no formal "start hidden" verb), so we explicitly
    /// send `Minimized(true)` from the first `update()` tick to make
    /// sure the surface is actually off-screen. Mutually exclusive with
    /// `force_unhide_on_first_frame` (only one can be true).
    headless_minimize_pending: bool,
    /// Last value sent to the pipeline via `Command::SetGuiPreviewActive`.
    /// `None` before the first send. Used to edge-trigger so we don't
    /// flood the command channel every frame.
    last_gui_preview_active: Option<bool>,
}

impl App {
    fn new(_cc: &eframe::CreationContext<'_>, headless: bool) -> Self {
        let cfg = Config::load();
        // Bring the on-disk autostart entry in line with the saved
        // preference. Cheap to call, no-op when in sync.
        if let Ok(exe) = std::env::current_exe()
            && let Err(e) = autostart::reconcile(cfg.start_on_login, &exe)
        {
            log::warn!("autostart reconcile: {e:#}");
        }
        let cameras = enumerate(&cfg.sink_device);
        let library = backgrounds::list();

        let tray = match Tray::install() {
            Ok(t) => Some(t),
            Err(e) => {
                log::warn!("tray icon install failed: {e:#} — Quit only via menu inside the GUI");
                None
            }
        };

        // Headless + tray-init failure would otherwise leave the user
        // with no way to interact: window starts hidden, no tray menu,
        // and InstanceLock blocks any second launch. Promote the window
        // to visible on the first frame and surface a hint in the
        // header.
        let tray_missing = tray.is_none();
        let force_unhide_on_first_frame = headless && tray_missing;
        let headless_minimize_pending = headless && !force_unhide_on_first_frame;
        let initial_error = if force_unhide_on_first_frame {
            Some(
                "Tray icon unavailable — close button hides the window; use Quit to exit."
                    .to_string(),
            )
        } else {
            None
        };

        let mut app = Self {
            cfg,
            cameras,
            library,
            thumbnails: HashMap::new(),
            pipeline: None,
            cmd_tx: None,
            preview_rx: None,
            preview_tex: None,
            last_preview_size: None,
            pipeline_state: PipelineState::default(),
            error: initial_error,
            tray,
            quit_requested: false,
            force_unhide_on_first_frame,
            headless_minimize_pending,
            last_gui_preview_active: None,
        };

        // Always start the pipeline at launch — both GUI and headless.
        // The sink graph is what advertises `/dev/video10` as a CAPTURE
        // device to consumers (Meet, browsers, OBS); without it,
        // `exclusive_caps=1` hides the loopback from camera lists. The
        // physical camera (`/dev/video0`, LED) stays released until a
        // real consumer reads — that's `lazy.rs`'s job, not this one.
        // On cold boot we may be racing `systemd-modules-load.service`,
        // so wait briefly for the v4l2 sink device to appear first.
        let sink = app.cfg.sink_device.clone();
        if wait_for_device(&sink, HEADLESS_DEVICE_WAIT) {
            app.start_pipeline();
        } else {
            let msg = format!(
                "{} did not appear within {:?} — is v4l2loopback loaded?",
                sink, HEADLESS_DEVICE_WAIT,
            );
            log::error!("{msg}");
            app.error = Some(msg);
        }

        app
    }

    fn save_settings(&self) {
        if let Err(e) = self.cfg.save() {
            log::warn!("config save failed: {e:#}");
        }
    }

    /// Quit-time cleanup: release the camera + virtual cam, then persist
    /// settings. Safe to call when the pipeline was never started — the
    /// helpers below short-circuit on `None`. Order matters: stop first
    /// so any final config changes triggered by stop are still saved.
    fn shutdown_cleanup(&mut self) {
        self.stop_pipeline();
        self.save_settings();
    }

    fn refresh_library(&mut self) {
        self.library = backgrounds::list();
        self.thumbnails
            .retain(|p, _| self.library.iter().any(|e| &e.path == p));
    }

    fn running(&self) -> bool {
        self.pipeline.is_some()
    }

    fn start_pipeline(&mut self) {
        if self.running() {
            return;
        }

        let (tx, rx) = crossbeam_channel::bounded::<PreviewFrame>(PREVIEW_CHANNEL_CAP);
        let pcfg = pipeline_config_from(&self.cfg, Some(tx));
        match Pipeline::start(
            pcfg,
            MODEL_BINARY_ONNX,
            MODEL_MULTICLASS_ONNX,
            MODEL_RVM_ONNX,
        ) {
            Ok(p) => {
                let cmd_tx = p.cmd_sender();
                // Push the saved toggle states to the freshly started
                // pipeline so they survive Stop+Start cycles (model
                // swaps, camera swaps).
                if !self.cfg.show_preview {
                    let _ = cmd_tx.send(Command::SetPreviewEnabled(false));
                }
                self.cmd_tx = Some(cmd_tx);
                self.pipeline = Some(p);
                self.preview_rx = Some(rx);
                self.preview_tex = None;
                self.last_preview_size = None;
                self.pipeline_state = PipelineState::default();
                self.error = None;
            }
            Err(e) => {
                let msg = format!("Failed to start: {e:#}");
                log::error!("{msg}");
                self.error = Some(msg);
            }
        }
    }

    fn stop_pipeline(&mut self) {
        if let Some(p) = self.pipeline.take() {
            p.stop();
            drop(p);
        }
        self.cmd_tx = None;
        self.preview_rx = None;
        self.preview_tex = None;
        self.last_preview_size = None;
        self.pipeline_state = PipelineState::default();
        // The next pipeline starts at gui_preview_active = false; clear
        // the cache so the next heartbeat re-sends.
        self.last_gui_preview_active = None;
    }

    fn restart_pipeline(&mut self) {
        if self.running() {
            self.stop_pipeline();
            self.start_pipeline();
        }
    }

    fn update_background_live(&mut self) {
        if let Some(tx) = &self.cmd_tx {
            let _ = tx.send(Command::SetBackground(build_background(&self.cfg)));
        }
    }

    fn drain_preview(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.preview_rx else { return };
        let mut latest: Option<PreviewFrame> = None;
        loop {
            match rx.try_recv() {
                Ok(f) => latest = Some(f),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.preview_rx = None;
                    break;
                }
            }
        }
        // Belt-and-braces: even if a frame slipped through the channel
        // before the pipeline received SetPreviewEnabled(false), don't
        // paint it.
        if !self.cfg.show_preview {
            return;
        }
        if let Some(frame) = latest {
            let size = [frame.width as usize, frame.height as usize];
            let color = egui::ColorImage::from_rgba_unmultiplied(size, &frame.rgba);
            match &mut self.preview_tex {
                Some(tex) if self.last_preview_size == Some(size) => {
                    tex.set(color, egui::TextureOptions::LINEAR);
                }
                slot => {
                    *slot = Some(ctx.load_texture("preview", color, egui::TextureOptions::LINEAR));
                    self.last_preview_size = Some(size);
                }
            }
        }
    }

    fn thumbnail_for(&mut self, ctx: &egui::Context, path: &Path) -> Option<egui::TextureHandle> {
        if let Some(t) = self.thumbnails.get(path) {
            return Some(t.clone());
        }
        let img = image::open(path)
            .ok()?
            .thumbnail(THUMBNAIL_PX, THUMBNAIL_PX)
            .to_rgba8();
        let size = [img.width() as usize, img.height() as usize];
        let color = egui::ColorImage::from_rgba_unmultiplied(size, &img.into_raw());
        let tex = ctx.load_texture(
            format!("thumb-{}", path.display()),
            color,
            egui::TextureOptions::LINEAR,
        );
        self.thumbnails.insert(path.to_path_buf(), tex.clone());
        Some(tex)
    }
}

// ---------------------------------------------------------------------------
//  egui::App impl — top-level layout
// ---------------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 0a. One-shot: headless cold start. `with_visible(false)` on the
        //     ViewportBuilder is best-effort on Wayland; explicitly send
        //     `Minimized(true)` from the first frame so the compositor
        //     reliably puts the surface off-screen. We also treat this
        //     frame as "hidden" for heartbeat purposes (see below) so a
        //     spurious activation doesn't fire while the command is in
        //     flight.
        let starting_headless_hidden = self.headless_minimize_pending;
        if self.headless_minimize_pending {
            self.headless_minimize_pending = false;
            ctx.send_viewport_cmd(ViewportCommand::Minimized(true));
        }

        // 0b. One-shot: if we started headless but the tray failed to
        //     install, promote the window to visible so the user can
        //     actually interact with the app. Mutually exclusive with 0a.
        if self.force_unhide_on_first_frame {
            self.force_unhide_on_first_frame = false;
            ctx.send_viewport_cmd(ViewportCommand::Visible(true));
            ctx.send_viewport_cmd(ViewportCommand::Focus);
        }

        // 1. Tray events first — they may toggle visibility or quit, and
        //    we want both effects applied before we lay out a frame.
        self.handle_tray_events(ctx);

        // 2. Close-button intercept: turn the close request into a hide
        //    unless `quit_requested` is set (i.e. the tray's Quit just
        //    fired).
        if ctx.input(|i| i.viewport().close_requested()) && !self.quit_requested {
            ctx.send_viewport_cmd(ViewportCommand::CancelClose);
            self.set_visible(ctx, false);
        }

        // 3. Read the *compositor-reported* minimized state. Egui has
        //    no `viewport.visible` analogue — `Visible(false)` on Wayland
        //    is a no-op and on other platforms eframe pauses redraws
        //    anyway, so minimized is the only meaningful signal.
        let minimized = is_minimized(ctx) || starting_headless_hidden;

        // GUI preview heartbeat: the pipeline treats this as a synthetic
        // consumer in its lazy state machine (so opening the preview
        // pane lights `/dev/video0` even with no real client attached,
        // and hiding to tray releases it after the deactivation
        // debounce). True only when the window is visible AND the user
        // has the preview toggle on. Edge-triggered to avoid flooding
        // the command channel every frame.
        let preview_active = !minimized && self.cfg.show_preview;
        self.send_gui_preview_active(preview_active);

        // 4. While minimized, skip layout entirely. eframe's repaint
        //    request keeps the loop ticking so tray events still land.
        if minimized {
            ctx.request_repaint_after(Duration::from_millis(150));
            return;
        }

        // Refresh the cached pipeline state for the footer; this also
        // doubles as cheap evidence that the pipeline is still alive.
        // Done after the early-return so we don't pay it on minimized
        // ticks.
        if let Some(p) = &self.pipeline {
            self.pipeline_state = p.state();
        }

        self.drain_preview(ctx);

        // Whole window fill (otherwise the area outside panels uses defaults).
        ctx.style_mut(|s| s.visuals.panel_fill = color::PANEL);

        self.header(ctx);
        self.footer(ctx);
        self.sidebar(ctx);
        self.preview_pane(ctx);

        if self.running() {
            ctx.request_repaint_after(Duration::from_millis(PREVIEW_REPAINT_MS));
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.shutdown_cleanup();

        // Bypass eframe's EGL/glutin teardown when the user explicitly
        // asked to quit. NVIDIA's proprietary EGL Wayland driver
        // (libnvidia-egl-wayland → libEGL_nvidia) aborts inside
        // `wl_display_dispatch_queue` during context destruction,
        // dumping core every time. Our resources are already released
        // by `shutdown_cleanup` (GStreamer pipelines → Null, settings
        // saved); the flock and tray thread go away with the process.
        // Only short-circuit on intentional Quit so accidental exit
        // paths still go through normal teardown.
        if self.quit_requested {
            std::process::exit(0);
        }
    }
}

impl App {
    fn handle_tray_events(&mut self, ctx: &egui::Context) {
        let Some(tray) = &self.tray else { return };
        // Collect first so we don't borrow `self.tray` across the
        // mutating calls below.
        let events: Vec<TrayEvent> = tray.drain().collect();
        for evt in events {
            match evt {
                TrayEvent::Show => self.set_visible(ctx, true),
                TrayEvent::Hide => self.set_visible(ctx, false),
                TrayEvent::Quit => {
                    self.quit_requested = true;
                    ctx.send_viewport_cmd(ViewportCommand::Close);
                }
            }
        }
    }

    /// Idempotent visibility request. Doesn't track local state — every
    /// call just emits the appropriate viewport commands and lets the
    /// compositor be the source of truth (read back via
    /// `is_minimized(ctx)` in `update()`). Sending `Visible(true)` to
    /// an already-visible window is a harmless re-assert; same for the
    /// other direction.
    fn set_visible(&mut self, ctx: &egui::Context, visible: bool) {
        if visible {
            // Un-minimize first so the compositor remaps the surface,
            // then ask for visibility (X11/Win/macOS), then take focus.
            // Order matters on KDE Plasma: focus on a still-minimized
            // toplevel is silently dropped.
            ctx.send_viewport_cmd(ViewportCommand::Minimized(false));
            ctx.send_viewport_cmd(ViewportCommand::Visible(true));
            ctx.send_viewport_cmd(ViewportCommand::Focus);
        } else {
            // `Visible(false)` is a no-op on Wayland (xdg-shell has no
            // "hide toplevel" verb — only destroy or minimize), so we
            // pair it with `Minimized(true)`, which xdg-shell *does*
            // honor:
            // - X11 / Windows / macOS: Visible(false) takes effect, the
            //   window vanishes from the taskbar entirely; the Minimized
            //   request lands harmlessly on an unmapped window.
            // - Wayland: Visible(false) is dropped, but Minimized takes
            //   the surface off-screen. The taskbar entry stays — that
            //   compromise is unavoidable without compositor-specific
            //   protocol extensions, and it's the same UX every other
            //   "minimize to tray" Wayland app ships with today.
            ctx.send_viewport_cmd(ViewportCommand::Minimized(true));
            ctx.send_viewport_cmd(ViewportCommand::Visible(false));
        }
    }

    /// Edge-triggered heartbeat: only sends `Command::SetGuiPreviewActive`
    /// when `active` differs from the last value we sent. Quietly drops
    /// the send on a slow / disconnected channel — the pipeline will
    /// observe demand via consumer detection or the next heartbeat.
    fn send_gui_preview_active(&mut self, active: bool) {
        if self.last_gui_preview_active == Some(active) {
            return;
        }
        if let Some(tx) = &self.cmd_tx {
            let _ = tx.send(Command::SetGuiPreviewActive(active));
            self.last_gui_preview_active = Some(active);
        }
    }
}

// ---------------------------------------------------------------------------
//  Layout sections
// ---------------------------------------------------------------------------

impl App {
    fn header(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("header")
            .exact_height(56.0)
            .frame(
                egui::Frame::none()
                    .fill(color::PANEL)
                    .stroke(Stroke::new(1.0, color::STROKE)),
            )
            .show(ctx, |ui| {
                ui.add_space(0.0);
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.add_space(space::LG);
                    // Brand mark — frame-within-a-frame logo
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(28.0, 28.0), egui::Sense::hover());
                    let p = ui.painter();
                    // outer rounded square
                    p.rect(
                        rect.shrink(2.0),
                        Rounding::same(7.0),
                        color::PANEL_INSET,
                        Stroke::new(1.25, color::STROKE_STRONG),
                    );
                    // inner square
                    p.rect_stroke(
                        egui::Rect::from_center_size(rect.center(), egui::vec2(14.0, 14.0)),
                        Rounding::same(3.5),
                        Stroke::new(2.0, color::TEXT),
                    );
                    // accent dot
                    p.circle_filled(rect.right_top() + egui::vec2(-7.0, 7.0), 2.4, color::ACCENT);

                    ui.add_space(space::MD);
                    ui.label(
                        egui::RichText::new("LinuxBroadcast")
                            .strong()
                            .size(14.0)
                            .color(color::TEXT),
                    );
                    ui.label(
                        egui::RichText::new("· virtual webcam")
                            .size(13.0)
                            .color(color::TEXT_WEAK),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(space::LG);
                        // No tray host (or it failed to install) → keep a
                        // reachable Quit button in the header so the user
                        // is never stranded.
                        if self.tray.is_none() && ghost_button(ui, "Quit").clicked() {
                            self.quit_requested = true;
                            ctx.send_viewport_cmd(ViewportCommand::Close);
                        }
                        if let Some(err) = &self.error {
                            let err = err.clone();
                            ui.label(egui::RichText::new(err).small().color(color::DANGER));
                        }
                    });
                });
            });
    }

    fn footer(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("footer")
            .exact_height(32.0)
            .frame(
                egui::Frame::none()
                    .fill(color::PANEL)
                    .stroke(Stroke::new(1.0, color::STROKE)),
            )
            .show(ctx, |ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.add_space(space::LG);
                    let (dot_color, label, label_color) =
                        footer_status(self.running(), &self.pipeline_state);
                    ui.label(egui::RichText::new("●").color(dot_color).small());
                    ui.add_space(space::XS);
                    ui.label(egui::RichText::new(label).small().color(label_color));
                    ui.add_space(space::MD);
                    sep(ui);
                    ui.add_space(space::MD);
                    ui.label(
                        egui::RichText::new(format!("in  {}", self.cfg.source_device))
                            .monospace()
                            .small()
                            .color(color::TEXT_MUTED),
                    );
                    ui.add_space(space::MD);
                    ui.label(
                        egui::RichText::new(format!("→ {}", self.cfg.sink_device))
                            .monospace()
                            .small()
                            .color(color::TEXT_MUTED),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(space::LG);
                        ui.label(
                            egui::RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                                .monospace()
                                .small()
                                .color(color::TEXT_MUTED),
                        );
                    });
                });
            });
    }

    fn sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("sidebar")
            .exact_width(320.0)
            .resizable(false)
            .frame(
                egui::Frame::none()
                    .fill(color::PANEL)
                    .stroke(Stroke::new(1.0, color::STROKE))
                    .inner_margin(Margin::symmetric(space::LG, space::PANEL_PAD_Y)),
            )
            .show(ctx, |ui| {
                // Reserve a fixed bottom strip for the primary action so it
                // can never be pushed off-screen by tall sections (Library,
                // Settings, …). The remaining top region holds the sections
                // inside a ScrollArea so short windows stay usable.
                let panel_rect = ui.max_rect();
                let action_h = control::PRIMARY_HEIGHT;
                let action_rect = egui::Rect::from_min_max(
                    egui::pos2(panel_rect.min.x, panel_rect.max.y - action_h),
                    panel_rect.max,
                );
                let scroll_rect = egui::Rect::from_min_max(
                    panel_rect.min,
                    egui::pos2(panel_rect.max.x, action_rect.min.y - space::MD),
                );

                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(scroll_rect), |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false; 2])
                        .show(ui, |ui| {
                            ui.spacing_mut().item_spacing.y = space::SECTION_GAP;
                            ui.style_mut().spacing.item_spacing.y = space::SECTION_GAP;
                            self.sidebar_camera(ui);
                            self.sidebar_model(ui);
                            self.sidebar_scene(ui);
                            self.sidebar_settings(ui);
                        });
                });

                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(action_rect), |ui| {
                    self.primary_action(ui);
                });
            });
    }

    fn preview_pane(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(
                egui::Frame::none()
                    .fill(color::BG)
                    .inner_margin(Margin::symmetric(20.0, 18.0)),
            )
            .show(ctx, |ui| {
                ui.spacing_mut().item_spacing.y = 14.0;

                // Header row: "Preview" + status pills
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Preview")
                            .size(13.0)
                            .strong()
                            .color(color::TEXT_WEAK),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        pill(
                            ui,
                            &format!("→ {}", self.cfg.sink_device),
                            color::TEXT_WEAK,
                            None,
                        );
                        pill(
                            ui,
                            &format!(
                                "{}×{} · {} fps",
                                self.cfg.width, self.cfg.height, self.cfg.framerate
                            ),
                            color::TEXT_WEAK,
                            None,
                        );
                        if self.running() {
                            pill(ui, "LIVE", color::TEXT, Some(color::DANGER));
                        }
                    });
                });

                // Preview surface
                let avail = ui.available_size_before_wrap();
                let (rect, _) = ui.allocate_exact_size(avail, egui::Sense::hover());
                let p = ui.painter();
                p.rect_filled(
                    rect,
                    Rounding::same(radius::LG),
                    color::PANEL_INSET, // letterbox
                );
                p.rect_stroke(
                    rect,
                    Rounding::same(radius::LG),
                    Stroke::new(1.0, color::STROKE),
                );

                if !self.cfg.show_preview {
                    p.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "Preview hidden",
                        egui::FontId::proportional(14.0),
                        color::TEXT_MUTED,
                    );
                } else if let Some(tex) = &self.preview_tex {
                    let [tw, th] = self.last_preview_size.unwrap_or([1280, 720]);
                    let aspect = tw as f32 / th as f32;
                    let mut w = rect.width();
                    let mut h = w / aspect;
                    if h > rect.height() {
                        h = rect.height();
                        w = h * aspect;
                    }
                    let centered = egui::Rect::from_center_size(rect.center(), egui::vec2(w, h));
                    let mut mesh = egui::Mesh::with_texture(tex.id());
                    mesh.add_rect_with_uv(
                        centered,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        Color32::WHITE,
                    );
                    p.add(egui::Shape::mesh(mesh));
                } else {
                    let msg = if self.running() {
                        "Waiting for first frame…"
                    } else {
                        "Press Start to begin streaming."
                    };
                    p.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        msg,
                        egui::FontId::proportional(14.0),
                        color::TEXT_MUTED,
                    );
                }

                // Hovering controls along the preview's bottom edge.
                if self.running() {
                    let pad = 12.0;
                    let bottom_rect = egui::Rect::from_min_max(
                        rect.left_bottom() + egui::vec2(pad, -36.0),
                        rect.right_bottom() + egui::vec2(-pad, -pad),
                    );
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(bottom_rect), |ui| {
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                            floating_pill(
                                ui,
                                &format!("● Sending to {}", self.cfg.sink_device),
                                color::DANGER,
                            );
                        });
                    });
                }
            });
    }
}

// ---------------------------------------------------------------------------
//  Sidebar sections
// ---------------------------------------------------------------------------

impl App {
    fn sidebar_camera(&mut self, ui: &mut egui::Ui) {
        theme::section_caption(ui, "Camera");
        ui.add_space(space::SM);

        let current = self
            .cameras
            .iter()
            .find(|c| c.path == self.cfg.source_device)
            .cloned();

        // Custom-rendered "device select" row.
        let row_h = 44.0;
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), row_h),
            egui::Sense::click(),
        );
        let p = ui.painter();
        let stroke = if resp.hovered() {
            color::STROKE_STRONG
        } else {
            color::STROKE
        };
        p.rect(
            rect,
            Rounding::same(radius::MD),
            color::PANEL_INSET,
            Stroke::new(1.0, stroke),
        );
        // status dot
        let dot = rect.left_center() + egui::vec2(12.0, 0.0);
        p.circle_filled(dot, 3.5, color::SUCCESS);
        p.circle_stroke(
            dot,
            6.0,
            Stroke::new(1.0, color::SUCCESS.linear_multiply(0.25)),
        );
        // label
        let label = current
            .as_ref()
            .map(|c| c.label.clone())
            .unwrap_or_else(|| self.cfg.source_device.clone());
        let detail = format!(
            "{} · {}×{} · {} fps",
            self.cfg.source_device, self.cfg.width, self.cfg.height, self.cfg.framerate
        );
        p.text(
            rect.left_top() + egui::vec2(28.0, 8.0),
            egui::Align2::LEFT_TOP,
            truncate(&label, 28),
            egui::FontId::proportional(13.0),
            color::TEXT,
        );
        p.text(
            rect.left_top() + egui::vec2(28.0, 25.0),
            egui::Align2::LEFT_TOP,
            truncate(&detail, 36),
            egui::FontId::monospace(10.5),
            color::TEXT_MUTED,
        );
        // chevron
        let cx = rect.right_center() - egui::vec2(14.0, 0.0);
        for i in 0..2 {
            let dx = (i as f32 - 0.5) * 5.0;
            p.line_segment(
                [
                    cx + egui::vec2(dx, -2.5),
                    cx + egui::vec2(dx + 2.5 * (1.0 - 2.0 * i as f32), 1.5),
                ],
                Stroke::new(1.5, color::TEXT_MUTED),
            );
        }

        let popup_id = ui.make_persistent_id("camera_popup");
        if resp.clicked() {
            ui.memory_mut(|m| m.toggle_popup(popup_id));
        }
        let mut camera_changed = false;
        egui::popup_below_widget(
            ui,
            popup_id,
            &resp,
            egui::PopupCloseBehavior::CloseOnClick,
            |ui| {
                ui.set_min_width(rect.width());
                ui.set_max_height(280.0);
                for cam in &self.cameras {
                    let selected = cam.path == self.cfg.source_device;
                    let resp = ui.selectable_label(selected, &cam.label);
                    if resp.clicked() && !selected {
                        self.cfg.source_device = cam.path.clone();
                        camera_changed = true;
                    }
                }
            },
        );

        if camera_changed {
            self.save_settings();
            if self.running() {
                self.restart_pipeline();
            }
        }
    }

    fn sidebar_model(&mut self, ui: &mut egui::Ui) {
        theme::section_caption(ui, "Model");
        ui.add_space(space::SM);
        let current_label = self.cfg.model.label();
        let mut changed = false;
        egui::ComboBox::from_id_salt("model_combo")
            .width(ui.available_width())
            .selected_text(current_label)
            .show_ui(ui, |ui| {
                for &m in Model::ALL {
                    if ui
                        .selectable_label(self.cfg.model == m, m.label())
                        .clicked()
                        && self.cfg.model != m
                    {
                        self.cfg.model = m;
                        changed = true;
                    }
                }
            });
        if changed {
            self.save_settings();
            // The segmenter is built at pipeline start, so a model swap
            // requires a restart of the GStreamer graph.
            if self.running() {
                self.restart_pipeline();
            }
        }
    }

    fn sidebar_scene(&mut self, ui: &mut egui::Ui) {
        // Mode tabs (no caption, no separate import button — Library has its
        // own dashed Import tile, and the tabs themselves announce the
        // section's purpose).
        let mut mode_changed = false;
        let tab_gap: f32 = 6.0;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = tab_gap;
            let total_w = ui.available_width();
            let cell_w = (total_w - tab_gap * 2.0) / 3.0;
            for (label, mode) in [
                ("None", Mode::None),
                ("Blur", Mode::Blur),
                ("Replace", Mode::Replace),
            ] {
                let active = self.cfg.mode == mode;
                let (rect, resp) =
                    ui.allocate_exact_size(egui::vec2(cell_w, 30.0), egui::Sense::click());
                let p = ui.painter();
                let bg = if active {
                    color::PANEL_ALT
                } else if resp.hovered() {
                    color::PANEL_ALT.linear_multiply(0.5)
                } else {
                    color::PANEL_INSET
                };
                let stroke = if active && mode == Mode::Blur {
                    Stroke::new(1.0, color::ACCENT.linear_multiply(0.6))
                } else if active {
                    Stroke::new(1.0, color::STROKE_STRONG)
                } else {
                    Stroke::new(1.0, color::STROKE)
                };
                p.rect(rect, Rounding::same(radius::SM), bg, stroke);
                let text_color = if active && mode == Mode::Blur {
                    color::ACCENT
                } else if active {
                    color::TEXT
                } else {
                    color::TEXT_WEAK
                };
                p.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    label,
                    egui::FontId::proportional(12.0),
                    text_color,
                );
                if resp.clicked() && !active {
                    self.cfg.mode = mode;
                    mode_changed = true;
                }
            }
        });
        ui.add_space(space::SM);

        // Blur intensity slider — shown only in Blur mode (else it's confusing)
        if self.cfg.mode == Mode::Blur {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Blur intensity")
                        .small()
                        .color(color::TEXT_WEAK),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "{}%",
                            (self.cfg.blur_strength * 100.0).round() as i32
                        ))
                        .monospace()
                        .small()
                        .color(color::TEXT),
                    );
                });
            });
            // slim track + accent fill
            let (track_rect, resp) = ui.allocate_exact_size(
                egui::vec2(ui.available_width(), 14.0),
                egui::Sense::click_and_drag(),
            );
            let p = ui.painter();
            let track_y = track_rect.center().y;
            let track_l = track_rect.left() + 1.0;
            let track_r = track_rect.right() - 1.0;
            let bar = egui::Rect::from_min_max(
                egui::pos2(track_l, track_y - 2.0),
                egui::pos2(track_r, track_y + 2.0),
            );
            p.rect(
                bar,
                Rounding::same(2.0),
                color::PANEL_INSET,
                Stroke::new(1.0, color::STROKE),
            );
            // fill
            let fill_w = (track_r - track_l) * self.cfg.blur_strength;
            let fill = egui::Rect::from_min_max(
                egui::pos2(track_l, track_y - 2.0),
                egui::pos2(track_l + fill_w, track_y + 2.0),
            );
            p.rect_filled(fill, Rounding::same(2.0), color::ACCENT);
            // handle
            let handle_x = track_l + fill_w;
            p.circle_filled(egui::pos2(handle_x, track_y), 7.0, color::TEXT);
            p.circle_stroke(
                egui::pos2(handle_x, track_y),
                7.0,
                Stroke::new(2.0, color::ACCENT),
            );

            // drag interaction
            if let Some(pos) = resp.interact_pointer_pos() {
                let t = ((pos.x - track_l) / (track_r - track_l)).clamp(0.0, 1.0);
                if (t - self.cfg.blur_strength).abs() > 0.005 {
                    self.cfg.blur_strength = t;
                    mode_changed = true;
                }
            }
            ui.add_space(space::SM);
        }

        // Library grid (Replace mode only — owns the Import affordance).
        if self.cfg.mode == Mode::Replace {
            ui.add_space(space::XS);
            theme::section_caption(ui, "Library");
            ui.add_space(space::SM);

            let mut to_select: Option<PathBuf> = None;
            let mut to_remove: Option<PathBuf> = None;

            let library = self.library.clone();
            let active_path = self.cfg.background_path.clone();
            let cols = 3usize;
            let avail_w = ui.available_width();
            let gap = 8.0;
            let cell = ((avail_w - gap * (cols as f32 - 1.0)) / cols as f32).floor();
            let ctx = ui.ctx().clone();

            egui::ScrollArea::vertical()
                .max_height(220.0)
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(gap, gap);
                    let mut row: Vec<&LibraryEntry> = Vec::with_capacity(cols);
                    for entry in &library {
                        row.push(entry);
                        if row.len() == cols {
                            let r = std::mem::take(&mut row);
                            self.draw_library_row(
                                ui,
                                &ctx,
                                &r,
                                cell,
                                active_path.as_deref(),
                                &mut to_select,
                                &mut to_remove,
                            );
                        }
                    }
                    // Trailing partial row always gets a + Import tile —
                    // even when the library is empty, draw_library_row()
                    // will append it.
                    self.draw_library_row(
                        ui,
                        &ctx,
                        &row,
                        cell,
                        active_path.as_deref(),
                        &mut to_select,
                        &mut to_remove,
                    );
                });

            if let Some(p) = to_select {
                self.cfg.background_path = Some(p);
                self.cfg.mode = Mode::Replace;
                mode_changed = true;
            }
            if let Some(p) = to_remove {
                if let Err(e) = backgrounds::remove(&p) {
                    log::warn!("remove: {e:#}");
                }
                if self.cfg.background_path.as_deref() == Some(&p) {
                    self.cfg.background_path = None;
                    if self.cfg.mode == Mode::Replace {
                        self.cfg.mode = Mode::Blur;
                        mode_changed = true;
                    }
                }
                self.refresh_library();
            }
        }

        if mode_changed {
            self.save_settings();
            if self.running() {
                self.update_background_live();
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_library_row(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        row: &[&LibraryEntry],
        cell: f32,
        active_path: Option<&Path>,
        to_select: &mut Option<PathBuf>,
        to_remove: &mut Option<PathBuf>,
    ) {
        ui.horizontal(|ui| {
            for entry in row {
                let is_active = active_path == Some(&entry.path);
                let tex = self.thumbnail_for(ctx, &entry.path);
                let (rect, resp) =
                    ui.allocate_exact_size(egui::vec2(cell, cell), egui::Sense::click());
                let p = ui.painter();
                let stroke_color = if is_active {
                    color::ACCENT
                } else if resp.hovered() {
                    color::STROKE_STRONG
                } else {
                    color::STROKE
                };
                p.rect(
                    rect,
                    Rounding::same(control::THUMB_RADIUS),
                    color::PANEL_INSET,
                    Stroke::new(if is_active { 2.0 } else { 1.0 }, stroke_color),
                );
                if let Some(t) = &tex {
                    let mut mesh = egui::Mesh::with_texture(t.id());
                    let inner = rect.shrink(2.0);
                    mesh.add_rect_with_uv(
                        inner,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        Color32::WHITE,
                    );
                    p.add(egui::Shape::mesh(mesh));
                }
                // label badge
                let badge_h = 16.0;
                let badge = egui::Rect::from_min_max(
                    rect.left_bottom() + egui::vec2(4.0, -badge_h - 4.0),
                    rect.left_bottom() + egui::vec2(cell - 4.0, -4.0),
                );
                p.rect_filled(
                    badge,
                    Rounding::same(3.0),
                    Color32::from_rgba_premultiplied(0x06, 0x09, 0x0D, 0xC8),
                );
                p.text(
                    badge.left_center() + egui::vec2(6.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    truncate(&entry.label, 12),
                    egui::FontId::proportional(10.5),
                    color::TEXT,
                );
                // active check mark
                if is_active {
                    let center = rect.right_top() + egui::vec2(-10.0, 10.0);
                    p.circle_filled(center, 7.0, color::ACCENT);
                    p.text(
                        center,
                        egui::Align2::CENTER_CENTER,
                        "✓",
                        egui::FontId::proportional(11.0),
                        Color32::from_rgb(0x06, 0x18, 0x0F),
                    );
                }
                // delete on right-click / shift-click
                if resp.clicked() {
                    if ui.input(|i| i.modifiers.shift) {
                        *to_remove = Some(entry.path.clone());
                    } else {
                        *to_select = Some(entry.path.clone());
                    }
                }
                resp.context_menu(|ui| {
                    if ui.button("Remove from library").clicked() {
                        *to_remove = Some(entry.path.clone());
                        ui.close_menu();
                    }
                });
            }
            // Trailing import-tile (dashed) only on the last partially-filled row
            // — keeps the grid always feeling complete.
            if row.len() < 3 {
                let (rect, resp) =
                    ui.allocate_exact_size(egui::vec2(cell, cell), egui::Sense::click());
                let p = ui.painter();
                let stroke_color = if resp.hovered() {
                    color::STROKE_STRONG
                } else {
                    color::STROKE
                };
                // Dashed border emulated via 6 short strokes per side.
                p.rect_filled(
                    rect,
                    Rounding::same(control::THUMB_RADIUS),
                    color::PANEL_INSET,
                );
                draw_dashed_rect(
                    p,
                    rect,
                    control::THUMB_RADIUS,
                    Stroke::new(1.0, stroke_color),
                );
                p.text(
                    rect.center() - egui::vec2(0.0, 6.0),
                    egui::Align2::CENTER_CENTER,
                    "+",
                    egui::FontId::proportional(20.0),
                    color::TEXT_MUTED,
                );
                p.text(
                    rect.center() + egui::vec2(0.0, 12.0),
                    egui::Align2::CENTER_CENTER,
                    "Import",
                    egui::FontId::proportional(10.5),
                    color::TEXT_MUTED,
                );
                if resp.clicked() {
                    self.do_import();
                }
            }
        });
    }

    /// Small settings panel under the scene controls. Right now it only
    /// holds the Start-on-login toggle, but the section caption keeps it
    /// extensible for future per-user preferences (fps, …).
    fn sidebar_settings(&mut self, ui: &mut egui::Ui) {
        theme::section_caption(ui, "Settings");
        ui.add_space(space::SM);

        // Start-on-login toggle.
        if toggle_row(
            ui,
            self.cfg.start_on_login,
            "Start on login",
            "Run headless at login so apps see the cam",
        ) {
            self.cfg.start_on_login = !self.cfg.start_on_login;
            self.save_settings();
            // Match the on-disk autostart entry to the new preference.
            // Failure here is non-fatal — surface it to the user via the
            // header so they know the setting didn't take.
            let exe = std::env::current_exe().ok();
            let result = match (self.cfg.start_on_login, exe.as_deref()) {
                (true, Some(p)) => autostart::install(p),
                (true, None) => Err(anyhow!("could not resolve current binary path")),
                (false, _) => autostart::uninstall(),
            };
            if let Err(e) = result {
                log::warn!("autostart toggle: {e:#}");
                self.error = Some(format!("Autostart toggle failed: {e:#}"));
            }
        }

        ui.add_space(space::SM);

        // Show-preview toggle. Off → the preview pane shows a static
        // placeholder and the pipeline stops forwarding frames to the
        // GUI (saves a per-frame RGBA clone). The broadcast itself is
        // unaffected: consumers of /dev/video10 see the same picture.
        if toggle_row(
            ui,
            self.cfg.show_preview,
            "Show preview",
            "Render live frames in this window",
        ) {
            self.cfg.show_preview = !self.cfg.show_preview;
            self.save_settings();
            if let Some(tx) = &self.cmd_tx {
                let _ = tx.send(Command::SetPreviewEnabled(self.cfg.show_preview));
            }
            // Drop the cached texture so the placeholder renders cleanly
            // (otherwise the last-painted frame would linger until the
            // next pane resize).
            if !self.cfg.show_preview {
                self.preview_tex = None;
                self.last_preview_size = None;
            }
        }

        ui.add_space(space::SM);

        // Auto-frame toggle. On → the pipeline locks onto the silhouette
        // on the first detection and keeps it centred (snap-once, à la
        // Meet). Works in all three bg modes: Blur and None do a
        // post-composite crop on the whole frame (wall zooms slightly
        // with you), Image keeps the bg image static and slides the
        // foreground over it.
        if toggle_row(
            ui,
            self.cfg.auto_frame,
            "Auto-frame",
            "Lock and centre on you (toggle off+on to re-frame)",
        ) {
            self.cfg.auto_frame = !self.cfg.auto_frame;
            self.save_settings();
            if let Some(tx) = &self.cmd_tx {
                let _ = tx.send(Command::SetFraming(self.cfg.auto_frame));
            }
        }
    }

    fn primary_action(&mut self, ui: &mut egui::Ui) {
        let running = self.running();
        let label = if running {
            "Stop broadcasting"
        } else {
            "Start broadcasting"
        };
        let (fill, stroke_color, text_color) = if running {
            (color::DANGER_SOFT, color::DANGER, color::DANGER)
        } else {
            (color::ACCENT_SOFT, color::ACCENT, color::ACCENT)
        };
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), control::PRIMARY_HEIGHT),
            egui::Sense::click(),
        );
        let p = ui.painter();
        let bg = if resp.hovered() {
            fill.linear_multiply(1.4)
        } else {
            fill
        };
        p.rect(
            rect,
            Rounding::same(radius::MD),
            bg,
            Stroke::new(1.0, stroke_color.linear_multiply(0.5)),
        );
        // tally / start dot
        let dot = rect.left_center() + egui::vec2(16.0, 0.0);
        p.circle_filled(dot, 4.0, stroke_color);
        p.circle_stroke(
            dot,
            7.0,
            Stroke::new(1.0, stroke_color.linear_multiply(0.35)),
        );
        p.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(13.0),
            text_color,
        );
        if resp.clicked() {
            if running {
                self.stop_pipeline();
            } else {
                self.start_pipeline();
            }
        }
    }

    fn do_import(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp"])
            .pick_file()
        {
            match backgrounds::import(&p) {
                Ok(stored) => {
                    self.cfg.background_path = Some(stored);
                    self.cfg.mode = Mode::Replace;
                    self.refresh_library();
                    self.save_settings();
                    if self.running() {
                        self.update_background_live();
                    }
                }
                Err(e) => {
                    log::warn!("import: {e:#}");
                    self.error = Some(format!("Import failed: {e:#}"));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  Small drawing helpers
// ---------------------------------------------------------------------------

fn pill(ui: &mut egui::Ui, text: &str, fg: Color32, bg_override: Option<Color32>) {
    let font = egui::FontId::monospace(10.5);
    let galley = ui.fonts(|f| f.layout_no_wrap(text.to_string(), font.clone(), fg));
    let pad = egui::vec2(10.0, 6.0);
    let has_dot = bg_override == Some(color::DANGER);
    // 14 px = dot (6 dia) + 8 px gap before the glyphs.
    let extra_left = if has_dot { 14.0 } else { 0.0 };
    let size = galley.size() + pad * 2.0 + egui::vec2(extra_left, 0.0);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let p = ui.painter();
    let bg = bg_override.unwrap_or(color::PANEL_ALT);
    p.rect(
        rect,
        Rounding::same(999.0),
        bg,
        Stroke::new(1.0, color::STROKE),
    );
    if has_dot {
        let dot = rect.left_center() + egui::vec2(10.0, 0.0);
        p.circle_filled(dot, 3.0, Color32::WHITE);
        p.text(
            rect.left_center() + egui::vec2(10.0 + extra_left, 0.0),
            egui::Align2::LEFT_CENTER,
            text,
            font,
            fg,
        );
    } else {
        p.text(rect.center(), egui::Align2::CENTER_CENTER, text, font, fg);
    }
    ui.add_space(6.0);
}

fn floating_pill(ui: &mut egui::Ui, text: &str, accent: Color32) {
    let font = egui::FontId::monospace(10.5);
    let galley = ui.fonts(|f| f.layout_no_wrap(text.to_string(), font.clone(), color::TEXT));
    let pad = egui::vec2(10.0, 6.0);
    let (rect, _) = ui.allocate_exact_size(galley.size() + pad * 2.0, egui::Sense::hover());
    let p = ui.painter();
    p.rect(
        rect,
        Rounding::same(999.0),
        Color32::from_rgba_premultiplied(0x08, 0x0A, 0x0E, 0xC0),
        Stroke::new(1.0, color::STROKE_STRONG),
    );
    let dot = rect.left_center() + egui::vec2(10.0, 0.0);
    p.circle_filled(dot, 3.0, accent);
    p.text(
        rect.left_center() + egui::vec2(20.0, 0.0),
        egui::Align2::LEFT_CENTER,
        text.trim_start_matches("● "),
        font,
        color::TEXT,
    );
}

fn ghost_button(ui: &mut egui::Ui, text: &str) -> egui::Response {
    let font = egui::FontId::proportional(11.0);
    let galley = ui.fonts(|f| f.layout_no_wrap(text.to_string(), font.clone(), color::TEXT_WEAK));
    let pad = egui::vec2(10.0, 6.0);
    let (rect, resp) = ui.allocate_exact_size(galley.size() + pad * 2.0, egui::Sense::click());
    let p = ui.painter();
    let (fill, stroke_color, text_color) = if resp.hovered() {
        (color::PANEL_ALT, color::STROKE_STRONG, color::TEXT)
    } else {
        (Color32::TRANSPARENT, color::STROKE, color::TEXT_WEAK)
    };
    p.rect(
        rect,
        Rounding::same(radius::SM),
        fill,
        Stroke::new(1.0, stroke_color),
    );
    p.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        text,
        font,
        text_color,
    );
    resp
}

/// Settings-section toggle row: title + subtitle on the left, pill
/// switch on the right. Returns `true` on click so the caller can flip
/// the underlying value and run any side effects.
fn toggle_row(ui: &mut egui::Ui, active: bool, title: &str, subtitle: &str) -> bool {
    let row_h = 36.0;
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), row_h),
        egui::Sense::click(),
    );
    let p = ui.painter();
    let stroke_color = if active {
        color::ACCENT.linear_multiply(0.6)
    } else if resp.hovered() {
        color::STROKE_STRONG
    } else {
        color::STROKE
    };
    p.rect(
        rect,
        Rounding::same(radius::MD),
        color::PANEL_INSET,
        Stroke::new(1.0, stroke_color),
    );
    p.text(
        rect.left_center() + egui::vec2(12.0, -6.0),
        egui::Align2::LEFT_CENTER,
        title,
        egui::FontId::proportional(13.0),
        color::TEXT,
    );
    p.text(
        rect.left_center() + egui::vec2(12.0, 8.0),
        egui::Align2::LEFT_CENTER,
        subtitle,
        egui::FontId::proportional(10.5),
        color::TEXT_MUTED,
    );
    let switch_w = 32.0;
    let switch_h = 18.0;
    let switch_rect = egui::Rect::from_center_size(
        egui::pos2(rect.right() - 12.0 - switch_w / 2.0, rect.center().y),
        egui::vec2(switch_w, switch_h),
    );
    let track_color = if active {
        color::ACCENT
    } else {
        color::STROKE_STRONG
    };
    p.rect_filled(switch_rect, Rounding::same(switch_h / 2.0), track_color);
    let knob_x = if active {
        switch_rect.right() - switch_h / 2.0
    } else {
        switch_rect.left() + switch_h / 2.0
    };
    p.circle_filled(
        egui::pos2(knob_x, switch_rect.center().y),
        switch_h / 2.0 - 2.0,
        color::TEXT,
    );
    resp.clicked()
}

/// Compute the footer status indicator for the current pipeline state.
/// Returns `(dot_color, label, label_color)`.
///
/// Three layers of nuance:
/// - pipeline not running at all → grey "Idle".
/// - running but lazy state is Idle (no consumer) → blue "Standby" so
///   the user sees the difference between "the app is on but the camera
///   is off" and "the app is off".
/// - running and Live → green "LIVE → name(pid)" if there's a real
///   consumer, otherwise green "LIVE (preview)" (the GUI's preview
///   heartbeat is the demand source).
fn footer_status(running: bool, state: &PipelineState) -> (Color32, String, Color32) {
    if !running {
        return (color::TEXT_MUTED, "Idle".to_string(), color::TEXT_MUTED);
    }
    match state {
        PipelineState::Live { consumers } if !consumers.is_empty() => {
            // Show the first consumer; if there are several, append a count.
            let first = &consumers[0];
            let label = if consumers.len() == 1 {
                format!("LIVE → {} ({})", first.name, first.pid)
            } else {
                format!(
                    "LIVE → {} ({}) + {}",
                    first.name,
                    first.pid,
                    consumers.len() - 1
                )
            };
            (color::SUCCESS, label, color::SUCCESS)
        }
        PipelineState::Live { .. } => {
            // Live without external consumers: the GUI preview heartbeat
            // is the demand source.
            (color::SUCCESS, "LIVE (preview)".to_string(), color::SUCCESS)
        }
        PipelineState::Idle => (
            color::ACCENT,
            "Standby (no consumer)".to_string(),
            color::ACCENT,
        ),
    }
}

fn sep(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(1.0, 14.0), egui::Sense::hover());
    ui.painter().vline(
        rect.center().x,
        rect.y_range(),
        Stroke::new(1.0, color::STROKE_STRONG),
    );
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max - 1).collect();
        out.push('…');
        out
    }
}

fn draw_dashed_rect(p: &egui::Painter, rect: egui::Rect, _radius: f32, stroke: Stroke) {
    let dash = 4.0;
    let gap = 3.0;
    let edges = [
        (rect.left_top(), rect.right_top()),
        (rect.right_top(), rect.right_bottom()),
        (rect.right_bottom(), rect.left_bottom()),
        (rect.left_bottom(), rect.left_top()),
    ];
    for (a, b) in edges {
        let dir = (b - a).normalized();
        let len = (b - a).length();
        let mut t = 0.0;
        while t < len {
            let s = a + dir * t;
            let e = a + dir * (t + dash).min(len);
            p.line_segment([s, e], stroke);
            t += dash + gap;
        }
    }
}
