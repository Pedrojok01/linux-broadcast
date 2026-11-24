//! System tray icon + menu, served over the StatusNotifierItem D-Bus
//! protocol by `ksni` (pure Rust, no GTK). KDE, GNOME with the
//! AppIndicator extension, XFCE, Cinnamon and waybar all host it.
//!
//! ksni runs the D-Bus service on its own thread. Menu picks and a left
//! click on the icon are forwarded to the egui loop through a crossbeam
//! channel that `App::logic()` drains every frame, and the shared egui
//! waker requests a repaint so they land at once even while the window
//! is hidden.
//!
//! Keep the public surface coarse: this module exposes `TrayEvent`
//! (`Show` / `Hide` / `Quit`) — never raw menu items — so the rest of the
//! app reasons about user intent, not D-Bus plumbing.
//!
//! Install fails when no tray host is running (some bare WMs, GNOME
//! without the AppIndicator extension). The error is logged and the GUI
//! keeps working without a tray entry. `--headless` installs the tray too;
//! if that fails, `App::new` un-hides the window on the first frame so the
//! user isn't left with no way to reach it.

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender, unbounded};
use ksni::blocking::{Handle, TrayMethods};
use std::time::Duration;

use crate::activation::EguiWaker;
use crate::icon;

/// How long the graceful Quit path gets before the watchdog forces exit.
const QUIT_GRACE: Duration = Duration::from_millis(500);

/// Coarse events emitted by the tray, consumed by the GUI on each
/// `App::logic()` tick.
#[derive(Debug, Clone, Copy)]
pub enum TrayEvent {
    /// Menu: Show, or a left click on the icon.
    Show,
    /// Menu: Hide.
    Hide,
    /// Menu: Quit. The only path that terminates the process: the window's
    /// close button hides instead.
    Quit,
}

/// Handle returned to the app: the event receiver, plus the service handle
/// that keeps the icon registered for the life of the process.
pub struct Tray {
    rx: Receiver<TrayEvent>,
    _service: Handle<LbTray>,
}

impl Tray {
    /// Register the tray icon with the session's StatusNotifierWatcher.
    /// `waker` is the shared egui-context slot used to wake the egui loop
    /// the instant a menu event fires.
    pub fn install(waker: EguiWaker) -> Result<Self> {
        let (tx, rx) = unbounded::<TrayEvent>();
        let tray = LbTray {
            tx,
            waker,
            icon: argb_icon(),
        };
        let service = tray.spawn().context("register tray icon")?;
        Ok(Self {
            rx,
            _service: service,
        })
    }

    /// Drains all pending tray events without blocking. Called every
    /// frame from the egui loop.
    pub fn drain(&self) -> impl Iterator<Item = TrayEvent> + '_ {
        self.rx.try_iter()
    }
}

struct LbTray {
    tx: Sender<TrayEvent>,
    waker: EguiWaker,
    icon: ksni::Icon,
}

impl LbTray {
    fn emit(&self, evt: TrayEvent) {
        let _ = self.tx.send(evt);
        if let Some(ctx) = self.waker.get() {
            ctx.request_repaint();
        }
        if matches!(evt, TrayEvent::Quit) {
            arm_quit_watchdog();
        }
    }
}

impl ksni::Tray for LbTray {
    fn id(&self) -> String {
        "linux-broadcast".into()
    }

    fn title(&self) -> String {
        "LinuxBroadcast".into()
    }

    fn icon_pixmap(&self) -> Vec<ksni::Icon> {
        vec![self.icon.clone()]
    }

    fn tool_tip(&self) -> ksni::ToolTip {
        ksni::ToolTip {
            title: "LinuxBroadcast — virtual webcam".into(),
            ..Default::default()
        }
    }

    fn activate(&mut self, _x: i32, _y: i32) {
        self.emit(TrayEvent::Show);
    }

    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::StandardItem;
        let item = |label: &str, evt: TrayEvent| {
            StandardItem {
                label: label.into(),
                activate: Box::new(move |this: &mut Self| this.emit(evt)),
                ..Default::default()
            }
            .into()
        };
        vec![
            item("Show LinuxBroadcast", TrayEvent::Show),
            item("Hide window", TrayEvent::Hide),
            ksni::MenuItem::Separator,
            item("Quit", TrayEvent::Quit),
        ]
    }
}

/// The app icon as StatusNotifierItem wants it: ARGB32, network byte order.
fn argb_icon() -> ksni::Icon {
    let lb = icon::build();
    let data = lb
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|&[r, g, b, a]| [a, r, g, b])
        .collect();
    ksni::Icon {
        width: lb.width as i32,
        height: lb.height as i32,
        data,
    }
}

/// Force the exit if the graceful Quit path stalls, as it can when the
/// window is hidden on Wayland and the compositor sends no frame callbacks.
fn arm_quit_watchdog() {
    std::thread::Builder::new()
        .name("lb-quit-watchdog".into())
        .spawn(|| {
            std::thread::sleep(QUIT_GRACE);
            log::info!("tray Quit: graceful path did not complete in time, forcing exit");
            // `_exit`, not `exit`: GStreamer/GLib atexit handlers block on a
            // pipeline drain that never finishes here. Settings are saved and
            // the kernel releases the v4l2 fds.
            // SAFETY: `_exit` is async-signal-safe and touches no Rust state.
            #[allow(unsafe_code)]
            unsafe {
                libc::_exit(0);
            }
        })
        .ok();
}
