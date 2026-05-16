//! System tray icon + menu bridge.
//!
//! The `tray-icon` crate's Linux backend (libayatana-appindicator) needs
//! a running GTK main loop on *some* thread, but eframe/winit don't host
//! one and aren't going to. We solve that by spawning a dedicated
//! `lb-tray-gtk` thread that calls `gtk::init()` + `gtk::main()` and
//! constructs the tray icon there (GTK objects are thread-affine — they
//! panic if poked from anywhere else). Menu events fire on that thread
//! and are forwarded to the egui app loop through a crossbeam channel
//! that `App::update()` drains every frame.
//!
//! Keep the public surface coarse: this module exposes `TrayEvent`
//! (`Show` / `Hide` / `Quit`) — never raw menu-item ids — so the rest of
//! the app reasons about user intent, not GTK plumbing.
//!
//! Install can fail on minimal sessions without an appindicator host
//! (some headless WMs, container-only desktops). When it does the error
//! is logged and the GUI keeps working without a tray entry — the close
//! button intercept in `ui.rs` still keeps the process alive on the
//! "headless autostart, no GUI visible" path because `--headless` skips
//! tray creation by design.

use anyhow::{Context, Result, anyhow};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use std::sync::OnceLock;

use tray_icon::{
    Icon, TrayIconBuilder,
    menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem},
};

use crate::activation::EguiWaker;
use crate::icon;

/// Coarse events emitted by the tray, consumed by the GUI on each
/// `egui::App::update()` tick. Granular menu-item identity stays inside
/// this module — the rest of the app only sees user intent.
#[derive(Debug, Clone, Copy)]
pub enum TrayEvent {
    /// Menu: Show. Also fired on a left-click of the icon (most desktops).
    Show,
    /// Menu: Hide.
    Hide,
    /// Menu: Quit. Triggers a real shutdown — only path that actually
    /// terminates the process now that the close button hides instead.
    Quit,
}

/// Handle returned to the app. The tray icon itself lives on the GTK
/// thread; this struct only owns the receiver-side of the event channel.
pub struct Tray {
    rx: Receiver<TrayEvent>,
}

impl Tray {
    /// Spawn the GTK thread, build the tray icon + menu **on that thread**
    /// (GTK objects are thread-affine and panic if touched from anywhere
    /// else), and wire menu events into a crossbeam channel that the egui
    /// loop polls.
    ///
    /// Must be called once per process. A second call returns an error
    /// because tray-icon's static event handlers can only be registered
    /// once. `waker` is the shared egui-context slot used to nudge the
    /// egui loop out of an idle (hidden+minimized on Wayland) state the
    /// instant a menu event fires — without it, picking *Quit* on a
    /// hidden window can sit in the channel for ~a minute waiting for
    /// the next natural redraw.
    pub fn install(waker: EguiWaker) -> Result<Self> {
        if INSTALLED.set(()).is_err() {
            return Err(anyhow!("tray already installed"));
        }

        let (event_tx, event_rx) = unbounded::<TrayEvent>();
        let (ready_tx, ready_rx) = bounded::<Result<()>>(1);

        std::thread::Builder::new()
            .name("lb-tray-gtk".into())
            .spawn(move || {
                run_gtk_thread(event_tx, waker, ready_tx);
            })
            .context("spawn tray gtk thread")?;

        ready_rx
            .recv()
            .context("tray gtk thread vanished before init")??;

        Ok(Self { rx: event_rx })
    }

    /// Drains all pending tray events without blocking. Called every
    /// frame from the egui update loop.
    pub fn drain(&self) -> impl Iterator<Item = TrayEvent> + '_ {
        self.rx.try_iter()
    }
}

static INSTALLED: OnceLock<()> = OnceLock::new();

/// Body of the dedicated `lb-tray-gtk` thread.
///
/// Everything tray-icon-related (menu, icon, builder) is constructed
/// here because GTK objects are thread-affine: GLib panics with "GTK may
/// only be used from the main thread" if a menu/widget is touched from
/// a thread that didn't call `gtk::init()`. Confusingly, "the main
/// thread" in GLib parlance is *the thread that called `gtk::init()`*,
/// not the OS-process main thread, which is why running GTK off-main
/// here is fine as long as we keep all of it on this one thread.
///
/// The thread blocks on `gtk::main()` for the life of the process so
/// the tray icon stays alive. `_tray` is held in this scope to keep the
/// `TrayIcon` from being dropped (which would remove the icon).
fn run_gtk_thread(event_tx: Sender<TrayEvent>, waker: EguiWaker, ready_tx: Sender<Result<()>>) {
    if let Err(e) = gtk::init() {
        let _ = ready_tx.send(Err(anyhow!("gtk::init failed: {e}")));
        return;
    }

    let menu = Menu::new();
    let show = MenuItem::new("Show LinuxBroadcast", true, None);
    let hide = MenuItem::new("Hide window", true, None);
    let quit = MenuItem::new("Quit", true, None);
    let show_id = show.id().clone();
    let hide_id = hide.id().clone();
    let quit_id = quit.id().clone();

    if let Err(e) = menu.append_items(&[&show, &hide, &PredefinedMenuItem::separator(), &quit]) {
        let _ = ready_tx.send(Err(anyhow!("tray menu append: {e}")));
        return;
    }

    let lb_icon = icon::build();
    let icon_image = match Icon::from_rgba(lb_icon.rgba, lb_icon.width, lb_icon.height) {
        Ok(i) => i,
        Err(e) => {
            let _ = ready_tx.send(Err(anyhow!("tray icon from rgba: {e}")));
            return;
        }
    };

    let _tray = match TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_icon(icon_image)
        .with_tooltip("LinuxBroadcast — virtual webcam")
        .with_title("LinuxBroadcast")
        .build()
    {
        Ok(t) => t,
        Err(e) => {
            let _ = ready_tx.send(Err(anyhow!("tray icon build: {e}")));
            return;
        }
    };

    // Bridge tray-icon's static MenuEvent handler into our channel.
    // The closure may fire on any thread, but `Sender` is Sync.
    // Compares by `MenuId` (not label) so localization or label edits
    // can't break the wiring. After pushing, request a repaint via the
    // shared egui waker so the loop drains the event immediately.
    //
    // For *Quit* on Wayland we also arm a watchdog: the toolkit
    // `ViewportCommand::Close` path ultimately needs winit's
    // `Window::request_redraw()` to fire `RedrawRequested`, which the
    // compositor refuses to do for a hidden xdg-toplevel. The egui
    // graceful path may sit waiting indefinitely. After 500 ms we just
    // `process::exit(0)` — pipeline shutdown is cosmetic (kernel
    // releases the v4l2 fds on exit) and config is auto-saved on every
    // change, so nothing is lost. If the graceful path wins first, it
    // exits the process and the watchdog dies with it.
    MenuEvent::set_event_handler(Some(move |e: MenuEvent| {
        let evt = if e.id == show_id {
            TrayEvent::Show
        } else if e.id == hide_id {
            TrayEvent::Hide
        } else if e.id == quit_id {
            TrayEvent::Quit
        } else {
            return;
        };
        let _ = event_tx.send(evt);
        if let Some(ctx) = waker.get() {
            ctx.request_repaint();
        }
        if matches!(evt, TrayEvent::Quit) {
            std::thread::Builder::new()
                .name("lb-quit-watchdog".into())
                .spawn(|| {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    log::info!("tray Quit: graceful path did not complete in 500ms, forcing exit");
                    // Bypass `std::process::exit` / `libc::exit` because
                    // gstreamer (and possibly glib) install atexit
                    // handlers that block on pipeline drain. With the
                    // egui loop stalled on hidden-Wayland-window
                    // frame callbacks the drain never completes and
                    // `exit()` hangs forever, leaving the process
                    // alive even though we asked it to quit. `_exit`
                    // is the POSIX immediate-termination syscall: no
                    // atexit, no libc cleanup, kernel reaps the
                    // process. File descriptors are released by the
                    // kernel; config is already auto-saved.
                    //
                    // SAFETY: `_exit` is signal-safe and takes no Rust
                    // state; calling it from any thread is well-defined.
                    #[allow(unsafe_code)]
                    unsafe {
                        libc::_exit(0);
                    }
                })
                .ok();
        }
    }));

    let _ = ready_tx.send(Ok(()));

    // Blocks until gtk::main_quit() — which we never call, so this
    // thread runs for the life of the process and `_tray` stays alive.
    gtk::main();
}
