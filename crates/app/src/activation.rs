//! D-Bus single-instance activation.
//!
//! When a second `linux-broadcast` process starts (user double-clicked
//! the .desktop entry while the first instance is hidden in the tray,
//! or relaunched from a terminal), we want it to raise the existing
//! window instead of exiting silently on the per-user flock.
//!
//! This module implements that with a freedesktop-standard mechanism:
//! the running instance claims `io.github.pedrojok01.LinuxBroadcast` on
//! the session bus and serves `org.freedesktop.Application`. A new
//! process first checks whether that name has an owner; if it does, it
//! calls `Activate` and exits. The owner pushes a single
//! `ActivationEvent` into a crossbeam channel that the egui loop drains
//! every frame, where it triggers the same code path as the tray's Show
//! menu.
//!
//! Failure modes are deliberately quiet: no session bus (sandboxed
//! launches, container test runs) and bus errors both fall through to
//! the legacy flock check in `main.rs`. The flock stays as the
//! belt-and-braces guarantee that two instances never own
//! `/dev/video10` at once.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result};
use crossbeam_channel::Sender;
use eframe::egui;
use zbus::interface;
use zbus::zvariant::{OwnedValue, Value};

use crate::wayland_activation::WaylandHandleSlot;

/// Shared handle to the running egui context. `main()` creates the
/// `OnceLock`; `App::new` populates it on first frame; this module's
/// D-Bus handler reads it to wake the egui loop the instant an
/// `Activate` call arrives. Without this, the loop can sit idle while
/// the window is hidden on Wayland — the channel would receive the
/// event but `App::update()` wouldn't run to drain it.
pub type EguiWaker = Arc<OnceLock<egui::Context>>;

/// The single intent emitted by the D-Bus interface. Modeled as an enum
/// so future actions (`open file`, `start broadcasting`, etc.) can be
/// added without changing the channel signature.
#[derive(Debug, Clone)]
pub enum ActivationEvent {
    /// `org.freedesktop.Application.Activate` fired — raise the window.
    /// The optional token is the freedesktop xdg-activation /
    /// startup-notification token the caller minted (or inherited from
    /// the launcher that spawned it). When present, the receiver should
    /// publish it to winit's activation path so the compositor allows
    /// the focus grab. Absent on terminal launches with no
    /// `XDG_ACTIVATION_TOKEN` / `DESKTOP_STARTUP_ID` in env.
    Activate { token: Option<String> },
}

/// Well-known bus name and object path. Reverse-DNS form so the AUR /
/// .deb package can later flip the .desktop file to `DBusActivatable=true`
/// (which requires the .desktop basename to match this name) without
/// renaming the D-Bus surface.
const BUS_NAME: &str = "io.github.pedrojok01.LinuxBroadcast";
const OBJECT_PATH: &str = "/io/github/pedrojok01/LinuxBroadcast";

/// Owner-side: implements `org.freedesktop.Application` by forwarding
/// `Activate` into the channel. `Open` and `ActivateAction` are accepted
/// (so peers calling them don't see a NoSuchMethod error) but ignored —
/// LinuxBroadcast has no file or action surface today.
struct AppService {
    tx: Sender<ActivationEvent>,
    waker: EguiWaker,
    wayland_handles: WaylandHandleSlot,
}

impl AppService {
    /// Apply the xdg-activation token *directly* from the D-Bus
    /// worker thread when we're on Wayland, bypassing the egui loop.
    ///
    /// Why: on Wayland a hidden xdg-toplevel doesn't get compositor
    /// frame callbacks, so `Window::request_redraw()` (the path
    /// `Context::request_repaint()` ultimately uses) doesn't fire
    /// `RedrawRequested`, and `App::update()` can take several
    /// seconds to wake. By that time the activation token has
    /// expired in the compositor (KWin/Mutter invalidate tokens
    /// after a couple of seconds). Going through libwayland on this
    /// thread gets the token to the compositor while it's still
    /// fresh.
    ///
    /// X11 / no-Wayland-handles cases just return — the toolkit
    /// fallback in `App::handle_activation_events` is enough there.
    fn apply_token_now(&self, token: &str) {
        let Some(handles) = self.wayland_handles.get() else {
            return;
        };
        match handles.apply_token(token) {
            Ok(_) => log::info!(
                "xdg-activation token applied directly from D-Bus thread ({} chars)",
                token.len()
            ),
            Err(e) => log::warn!("direct xdg-activation apply failed: {e:#}"),
        }
    }

    /// Push an event and immediately wake the egui loop so it drains
    /// the channel without waiting for the next 150 ms tick. The
    /// egui-loop pass still handles `RequestUserAttention`, the
    /// X11 fallback path, and the `set_visible` toolkit cleanup.
    fn dispatch(&self, evt: ActivationEvent) {
        // Receiver living for the process lifetime; send failure means
        // the egui side dropped, which is harmless here.
        let _ = self.tx.send(evt);
        if let Some(ctx) = self.waker.get() {
            ctx.request_repaint();
        }
    }
}

#[interface(name = "org.freedesktop.Application")]
impl AppService {
    fn activate(&self, platform_data: HashMap<String, OwnedValue>) {
        let token = extract_token(&platform_data);
        if let Some(t) = &token {
            // Apply to the compositor *before* dispatching to egui:
            // the token expires within seconds, so freshness matters
            // more than ordering with the toolkit-level cleanup.
            self.apply_token_now(t);
        }
        self.dispatch(ActivationEvent::Activate { token });
    }

    fn open(&self, _uris: Vec<String>, platform_data: HashMap<String, OwnedValue>) {
        let token = extract_token(&platform_data);
        if let Some(t) = &token {
            self.apply_token_now(t);
        }
        self.dispatch(ActivationEvent::Activate { token });
    }

    fn activate_action(
        &self,
        _action_name: String,
        _parameters: Vec<OwnedValue>,
        platform_data: HashMap<String, OwnedValue>,
    ) {
        let token = extract_token(&platform_data);
        if let Some(t) = &token {
            self.apply_token_now(t);
        }
        self.dispatch(ActivationEvent::Activate { token });
    }
}

/// Read the freedesktop activation token from a `platform_data` dict.
/// Spec key is `"activation-token"` (Wayland xdg-activation /
/// startup-notification ID for X11); some callers historically used
/// `"desktop-startup-id"` so we accept both.
fn extract_token(data: &HashMap<String, OwnedValue>) -> Option<String> {
    for key in ["activation-token", "desktop-startup-id"] {
        if let Some(v) = data.get(key)
            && let Ok(s) = <&str>::try_from(v)
        {
            return Some(s.to_string());
        }
    }
    None
}

/// RAII guard for the served D-Bus connection. Dropping releases the
/// well-known name and tears down the connection's internal worker
/// thread. Held in `main()` for the process lifetime.
pub struct ServiceHandle {
    _conn: zbus::blocking::Connection,
}

/// Claim the well-known name and start serving. Call exactly once, after
/// `try_activate_existing` returned `Ok(false)` and the flock was
/// acquired — otherwise we'd race a still-alive sibling for the name.
///
/// Returns `Err` only when the bus exists but rejected our claim (e.g. a
/// sibling raced past the activation check). Lack of a session bus is
/// reported via `Ok(None)` so the caller can keep going.
pub fn serve(
    tx: Sender<ActivationEvent>,
    waker: EguiWaker,
    wayland_handles: WaylandHandleSlot,
) -> Result<Option<ServiceHandle>> {
    let conn = match zbus::blocking::connection::Builder::session() {
        Ok(b) => b,
        Err(e) => {
            log::info!("no session bus available ({e:#}); D-Bus activation disabled");
            return Ok(None);
        }
    }
    .name(BUS_NAME)
    .with_context(|| format!("register bus name {BUS_NAME}"))?
    .serve_at(
        OBJECT_PATH,
        AppService {
            tx,
            waker,
            wayland_handles,
        },
    )
    .with_context(|| format!("serve at {OBJECT_PATH}"))?
    .build()
    .context("build zbus connection")?;

    log::info!("D-Bus activation service ready as {BUS_NAME}");
    Ok(Some(ServiceHandle { _conn: conn }))
}

/// Probe the session bus for an existing instance and ask it to raise
/// its window. Returns `Ok(true)` when the call succeeded (caller should
/// exit cleanly), `Ok(false)` when no peer is reachable for any reason
/// (no bus, name unowned, transient bus error — caller should keep
/// going). Never returns `Err`; D-Bus availability is best-effort.
pub fn try_activate_existing() -> bool {
    let conn = match zbus::blocking::Connection::session() {
        Ok(c) => c,
        Err(e) => {
            log::debug!("no session bus available for activation check: {e:#}");
            return false;
        }
    };

    let proxy = match zbus::blocking::Proxy::new(
        &conn,
        BUS_NAME,
        OBJECT_PATH,
        "org.freedesktop.Application",
    ) {
        Ok(p) => p,
        Err(e) => {
            log::debug!("D-Bus proxy build failed: {e:#}");
            return false;
        }
    };

    // Forward whatever activation token we inherited from our parent
    // (the launcher / shell / terminal). GNOME Shell, KWin, and most
    // launchers set `XDG_ACTIVATION_TOKEN` for Wayland and
    // `DESKTOP_STARTUP_ID` for X11 startup-notification when they spawn
    // a process. Without one of these, the compositor will refuse to
    // let the running instance steal focus on Wayland — that's the
    // "click the .desktop entry vs. relaunch from a terminal" gap.
    let mut platform_data: HashMap<String, OwnedValue> = HashMap::new();
    let token = std::env::var("XDG_ACTIVATION_TOKEN")
        .ok()
        .or_else(|| std::env::var("DESKTOP_STARTUP_ID").ok())
        .filter(|s| !s.is_empty());
    if let Some(t) = token {
        if let Ok(v) = OwnedValue::try_from(Value::from(t.as_str())) {
            platform_data.insert("activation-token".to_string(), v);
        }
        log::info!(
            "forwarding activation token to running instance ({} chars)",
            t.len()
        );
    } else {
        log::info!(
            "no XDG_ACTIVATION_TOKEN / DESKTOP_STARTUP_ID in env; activation may be a no-op on Wayland"
        );
    }
    match proxy.call_method("Activate", &(platform_data,)) {
        Ok(_) => {
            log::info!("activated existing instance via D-Bus");
            true
        }
        Err(e) => {
            // The common case here is "name has no owner" — there is no
            // running peer, so we should proceed to launch normally.
            log::debug!("no existing instance to activate: {e:#}");
            false
        }
    }
}
