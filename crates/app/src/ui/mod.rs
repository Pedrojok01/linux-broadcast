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
//!   `show_preview`, `auto_frame`) is written through `Config::save`
//!   on change so `~/.config/linux-broadcast/config.toml` stays in
//!   sync with the GUI.
//!
//! Footer surface mirrors `PipelineState`: `● Idle`, `● Standby (no
//! consumer)`, or `● LIVE → name(pid)` while a real consumer is reading.

mod footer;
mod header;
mod preview_pane;
mod sidebar;
mod widgets;

use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use eframe::egui::{self, ViewportCommand};
use lb_pipeline::{Background, Command, Pipeline, PipelineConfig, PipelineState, PreviewFrame};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::activation::{ActivationEvent, EguiWaker};
use crate::autostart;
use crate::backgrounds::{self, LibraryEntry};
use crate::cameras::{CameraEntry, enumerate};
use crate::config::{Config, Mode};
use crate::theme::{self, color};
use crate::tray::{Tray, TrayEvent};
use crate::wayland_activation::{self, WaylandHandleSlot, WaylandHandles};
use crate::{MODEL_MULTICLASS_ONNX, MODEL_RVM_ONNX};

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

/// Single eframe entry point.
/// - GUI mode: window visible.
/// - Headless mode: window starts hidden in the tray.
///
/// The pipeline auto-starts at launch in both modes so consumers
/// (Meet, browsers, OBS) immediately see `/dev/video10` as a CAPTURE
/// device. The lazy state machine still keeps `/dev/video0` (LED)
/// released until a real consumer reads.
///
/// The window's close button always hides (minimises to tray); only the
/// tray's Quit menu actually exits. This lets a single instance own
/// `/dev/video10` for the whole session, addressing the "I autostarted
/// it on login and now have no way to stop it" pain point.
pub fn run(
    headless: bool,
    activation_rx: Receiver<ActivationEvent>,
    waker: EguiWaker,
    wayland_handles: WaylandHandleSlot,
) -> Result<()> {
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
            // Publish the egui context to the shared waker before any
            // activation or tray event can fire. `set` only succeeds
            // once; subsequent App::new invocations (would only happen
            // if eframe ever recreates the app, which it currently
            // doesn't) silently ignore — the slot is shared across the
            // process. Both the D-Bus handler and the tray menu handler
            // read it to nudge this loop out of idle.
            let _ = waker.set(cc.egui_ctx.clone());
            Ok(Box::new(App::new(
                cc,
                headless,
                activation_rx,
                waker,
                wayland_handles,
            )))
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

pub(super) struct App {
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
    /// Activation events from the D-Bus service in `activation.rs`. A
    /// second `linux-broadcast` launch fires `Activate`, which lands here
    /// and triggers the same `set_visible(true)` path as the tray's Show
    /// menu. Disconnected silently when D-Bus is unavailable — drained
    /// every frame either way, the absent sender just means no events.
    activation_rx: Receiver<ActivationEvent>,
    /// Shared cache of winit's raw `wl_display` / `wl_surface` pointers
    /// for the D-Bus thread's direct xdg-activation path. Populated
    /// from the first frame of `update()` once `eframe::Frame` is
    /// available. `None` on X11.
    wayland_handles: WaylandHandleSlot,
    /// Cleared after the first frame populates the Wayland slot — keeps
    /// `update()` from re-running the (cheap but pointless) capture
    /// every tick.
    wayland_capture_pending: bool,
}

impl App {
    fn new(
        _cc: &eframe::CreationContext<'_>,
        headless: bool,
        activation_rx: Receiver<ActivationEvent>,
        waker: EguiWaker,
        wayland_handles: WaylandHandleSlot,
    ) -> Self {
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

        let tray = match Tray::install(waker) {
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
            activation_rx,
            wayland_handles,
            wayland_capture_pending: true,
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
        match Pipeline::start(pcfg, MODEL_MULTICLASS_ONNX, MODEL_RVM_ONNX) {
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

    /// Pre-layout chores: one-shot startup commands, tray events, the
    /// close-button intercept, the preview heartbeat. Returns whether
    /// the window is currently minimized — the caller uses that to skip
    /// the rest of the layout for this frame.
    fn handle_lifecycle(&mut self, ctx: &egui::Context, frame: &eframe::Frame) -> bool {
        // First frame: cache winit's raw Wayland handles so the D-Bus
        // activation thread can apply tokens without waiting for the
        // egui loop to wake. No-op on X11.
        if self.wayland_capture_pending {
            if let Some(h) = WaylandHandles::capture(frame, frame)
                && self.wayland_handles.set(h).is_ok()
            {
                log::info!("cached Wayland handles for direct xdg-activation");
            }
            self.wayland_capture_pending = false;
        }

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

        // 1b. D-Bus activation: a second `linux-broadcast` launch
        //     (.desktop click, terminal relaunch) fires `Activate`,
        //     which lands here and reuses the tray's Show path. Drained
        //     even when the channel's sender is gone — try_iter is a
        //     no-op in that case.
        self.handle_activation_events(ctx, frame);

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

        minimized
    }

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

    fn handle_activation_events(&mut self, ctx: &egui::Context, frame: &eframe::Frame) {
        // Coalesce: any number of Activate events in a single frame are
        // equivalent to one. Keep the latest non-empty token — a newer
        // token is more likely to still be valid for the compositor.
        let mut got = false;
        let mut latest_token: Option<String> = None;
        while let Ok(evt) = self.activation_rx.try_recv() {
            got = true;
            let ActivationEvent::Activate { token } = evt;
            if token.is_some() {
                latest_token = token;
            }
        }
        if !got {
            return;
        }

        // On Wayland, feed the token to the compositor through
        // xdg-activation-v1 *before* the toolkit-level focus commands.
        // Without this step, set_minimized(false) + Focus are dropped
        // for focus-stealing prevention — see wayland_activation.rs.
        // No-op on X11 / when no token came along / when the
        // compositor doesn't speak xdg-activation-v1.
        if let Some(token) = &latest_token {
            match wayland_activation::apply_token(frame, frame, token) {
                Ok(true) => log::info!("applied xdg-activation token ({} chars)", token.len()),
                Ok(false) => log::info!("xdg-activation skipped: not a Wayland session"),
                Err(e) => log::warn!("xdg-activation apply failed: {e:#}"),
            }
        } else {
            log::info!("activation requested without token — toolkit-level fallback only");
        }

        self.set_visible(ctx, true);

        // Belt-and-braces for compositors that still don't surface the
        // window (token expired, no xdg-activation support, etc.): ask
        // for the urgency / attention hint so the taskbar entry
        // highlights and the user can click it manually.
        ctx.send_viewport_cmd(ViewportCommand::RequestUserAttention(
            egui::UserAttentionType::Critical,
        ));
        ctx.request_repaint();
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

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let minimized = self.handle_lifecycle(ctx, frame);

        // While minimized, skip layout entirely. eframe's repaint
        // request keeps the loop ticking so tray events still land.
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
