//! Apply a Wayland `xdg-activation-v1` token to our own window.
//!
//! On Wayland, focus-stealing prevention means a client can't just call
//! `set_focused()` or `set_minimized(false)` and expect the compositor
//! to honor it — those requests are silently dropped unless the client
//! presents an activation token that was minted in response to a recent
//! user action. This is exactly the mechanism Slack, Telegram, and any
//! Qt/Chromium tray app rely on for their "click the launcher icon to
//! raise the hidden window" UX.
//!
//! The flow is:
//!
//! 1. The second `linux-broadcast` process inherits `$XDG_ACTIVATION_TOKEN`
//!    from the launcher (GNOME Shell / KWin mint it on the user click)
//!    or `$DESKTOP_STARTUP_ID` on X11.
//! 2. It forwards that token to the running instance through D-Bus
//!    `org.freedesktop.Application.Activate(platform_data)` —
//!    see `crate::activation`.
//! 3. The running instance hands the token + its window handles to
//!    [`apply_token`], which talks `xdg_activation_v1.activate(token,
//!    surface)` directly to the compositor. The compositor checks the
//!    token's serial against the original user action and grants the
//!    focus / unminimize.
//!
//! eframe 0.29 / egui 0.29 don't expose any way to do this through
//! `ViewportCommand`, so we go under the toolkit: we wrap winit's
//! existing `wl_display` and `wl_surface` raw pointers (via
//! [`raw_window_handle`]) into a guest [`wayland_client::Connection`]
//! that shares the same socket. This avoids opening a second wayland
//! connection (which would have a different `wl_display` — and you
//! cannot use a surface bound to display A through display B).
//!
//! Non-Wayland sessions (X11, headless) return `Ok(false)` from
//! [`apply_token`] without touching the display.

#![allow(unsafe_code)]

use std::os::raw::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};

use anyhow::{Result, anyhow};
use raw_window_handle::{
    DisplayHandle, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
    WindowHandle,
};
use wayland_backend::sys::client::{Backend, ObjectId};
use wayland_client::{
    Connection, Dispatch, Proxy, QueueHandle,
    protocol::{wl_registry, wl_surface::WlSurface},
};
use wayland_protocols::xdg::activation::v1::client::{
    xdg_activation_token_v1::XdgActivationTokenV1, xdg_activation_v1::XdgActivationV1,
};

/// Thread-safe slot that caches winit's raw Wayland handles so the
/// D-Bus activation thread can apply tokens without going through the
/// egui loop (which on Wayland doesn't tick while the window is
/// hidden — `request_redraw` requires a compositor frame callback the
/// compositor refuses to send to an unmapped surface).
///
/// Populated on the first egui frame via [`WaylandHandles::capture`].
/// Holds raw pointers from `raw_window_handle`; they remain valid for
/// the lifetime of the eframe window, which is the lifetime of the
/// process for our app.
#[derive(Clone, Copy)]
pub struct WaylandHandles {
    display: NonNull<c_void>,
    surface: NonNull<c_void>,
}

// SAFETY: `libwayland-client` is explicitly thread-safe at the
// `wl_display` level for sending requests, and `wl_surface` is a
// long-lived object on that display. The raw pointers we hold are
// references to objects winit owns; we never free them.
unsafe impl Send for WaylandHandles {}
unsafe impl Sync for WaylandHandles {}

/// Shared slot owned by `main()`, populated by `App` on its first frame
/// when running on Wayland, consumed by the zbus activation handler.
pub type WaylandHandleSlot = Arc<OnceLock<WaylandHandles>>;

impl WaylandHandles {
    /// Pull the raw Wayland pointers out of an eframe `Frame`. Returns
    /// `None` on X11 / non-Wayland sessions, or when the toolkit
    /// hasn't initialized the handles yet (first frame can race).
    pub fn capture<D: HasDisplayHandle, W: HasWindowHandle>(
        display: &D,
        window: &W,
    ) -> Option<Self> {
        let dh = display.display_handle().ok()?;
        let wh = window.window_handle().ok()?;
        match (dh.as_raw(), wh.as_raw()) {
            (RawDisplayHandle::Wayland(d), RawWindowHandle::Wayland(w)) => Some(Self {
                display: d.display,
                surface: w.surface,
            }),
            _ => None,
        }
    }

    /// Apply an xdg-activation token straight to the compositor. Safe
    /// to call from any thread. See [`apply_token`] for protocol-level
    /// notes — this is the same call, just with cached pointers
    /// instead of a `Frame` reference.
    pub fn apply_token(&self, token: &str) -> Result<bool> {
        apply_token_raw(self.display.as_ptr(), self.surface.as_ptr(), token).map(|()| true)
    }
}

/// Apply `token` to the window described by `display`/`window`.
///
/// Returns `Ok(true)` when the activation request was sent (the
/// compositor may still reject it if the token has expired — that's
/// fine, the worst case is a no-op), `Ok(false)` when we are not on a
/// Wayland session, and `Err` on protocol / pointer-wrapping failures.
///
/// Callers should treat errors as best-effort: log and continue. The
/// caller has already issued the toolkit-level focus / un-minimize
/// commands as a fallback.
pub fn apply_token<D: HasDisplayHandle, W: HasWindowHandle>(
    display: &D,
    window: &W,
    token: &str,
) -> Result<bool> {
    let dh: DisplayHandle<'_> = display.display_handle()?;
    let wh: WindowHandle<'_> = window.window_handle()?;
    let (display_ptr, surface_ptr) = match (dh.as_raw(), wh.as_raw()) {
        (RawDisplayHandle::Wayland(d), RawWindowHandle::Wayland(w)) => {
            (d.display.as_ptr(), w.surface.as_ptr())
        }
        _ => return Ok(false),
    };
    apply_token_raw(display_ptr, surface_ptr, token).map(|()| true)
}

/// Shared Wayland protocol code. Takes raw pointers so callers can
/// either go through the safe `apply_token` (with toolkit-validated
/// handles) or use `WaylandHandles::apply_token` from another thread.
fn apply_token_raw(display_ptr: *mut c_void, surface_ptr: *mut c_void, token: &str) -> Result<()> {
    // SAFETY: `display_ptr` was produced by `winit` via the
    // `raw_window_handle` impl on its event loop and is live for the
    // lifetime of the eframe window. The Backend is constructed in
    // "guest" mode, so dropping it does not call
    // `wl_display_disconnect` on winit's display.
    let backend = unsafe { Backend::from_foreign_display(display_ptr.cast()) };
    let conn = Connection::from_backend(backend);

    let mut queue = conn.new_event_queue::<State>();
    let qh = queue.handle();

    // Trigger registry enumeration on our own queue (winit's queue is
    // untouched). One blocking roundtrip is enough to receive every
    // `wl_registry.global` event that already happened plus the
    // bindings we request below.
    let _registry = conn.display().get_registry(&qh, ());
    let mut state = State::default();
    queue.roundtrip(&mut state)?;
    let activation = state
        .activation
        .ok_or_else(|| anyhow!("compositor does not advertise xdg_activation_v1"))?;

    // SAFETY: `surface_ptr` was produced by `winit` for the same
    // `wl_display` we just wrapped, so it is a valid `wl_proxy`
    // pointer on this connection. The `ObjectId` we build is used
    // only to construct a `WlSurface` *proxy reference* for the
    // duration of this call — dropping the proxy does not destroy
    // the underlying surface.
    let surface_id = unsafe { ObjectId::from_ptr(WlSurface::interface(), surface_ptr.cast()) }?;
    let surface = WlSurface::from_id(&conn, surface_id)?;

    activation.activate(token.to_string(), &surface);
    conn.flush()?;
    Ok(())
}

#[derive(Default)]
struct State {
    activation: Option<XdgActivationV1>,
}

impl Dispatch<wl_registry::WlRegistry, ()> for State {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
            && interface == "xdg_activation_v1"
        {
            // Cap at v1 — that's all the protocol revision we use.
            state.activation =
                Some(registry.bind::<XdgActivationV1, _, _>(name, version.min(1), qh, ()));
        }
    }
}

// xdg_activation_v1 has no events; the empty impl satisfies Dispatch.
impl Dispatch<XdgActivationV1, ()> for State {
    fn event(
        _: &mut Self,
        _: &XdgActivationV1,
        _: <XdgActivationV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

// We don't mint our own tokens (the launcher already did that for us
// and the second-instance process forwarded the string), but the
// generated `XdgActivationV1` proxy can in principle produce
// `xdg_activation_token_v1` children, so wayland-client wants the impl.
impl Dispatch<XdgActivationTokenV1, ()> for State {
    fn event(
        _: &mut Self,
        _: &XdgActivationTokenV1,
        _: <XdgActivationTokenV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
