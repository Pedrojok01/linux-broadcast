// LinuxBroadcast — virtual webcam with background replacement for Linux.
// Copyright (C) 2025-2026 Pedrojok01
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! `linux-broadcast` binary entry point.
//!
//! Responsibilities:
//! - Embed the two ONNX models at compile time (`include_bytes!`) so the
//!   shipped binary is self-contained — no separate model files, no first-run
//!   download.
//! - Acquire the per-user single-instance lock before doing anything else
//!   that could race a sibling instance (consumer poller, sink graph,
//!   autostart reconciliation). The lock is held for the lifetime of the
//!   process.
//! - Decide whether to run the icon-dump tool (`LB_DUMP_ICON=1`, used by
//!   `cargo deb` to regenerate the menu icon at packaging time) or hand off
//!   to the egui app loop.
//! - Detect headless mode (`--headless` from the autostart .desktop, or the
//!   legacy `LB_HEADLESS=1` env var) and forward it to `ui::run`. There is
//!   no separate headless code path — the same eframe loop just starts with
//!   the window hidden in the tray.

mod activation;
mod autostart;
mod backgrounds;
mod cameras;
mod config;
mod desktop_install;
mod icon;
mod lock;
mod theme;
mod tray;
mod ui;
mod wayland_activation;

use anyhow::Result;

use std::sync::{Arc, OnceLock};

use crate::activation::{ActivationEvent, EguiWaker};
use crate::lock::InstanceLock;
use crate::wayland_activation::WaylandHandleSlot;

/// Bundled MediaPipe Selfie Multiclass ONNX (6 classes: bg/hair/body/face/clothes/other,
/// ~16 MB). Converted via tf2onnx from the official MediaPipe TFLite.
pub(crate) const MODEL_MULTICLASS_ONNX: &[u8] =
    include_bytes!("../../../models/selfie_multiclass.onnx");

/// Bundled Robust Video Matting ONNX (MobileNetV3 backbone, fp32, ~15 MB).
/// Sourced from PeterL1n/RobustVideoMatting v1.0.0 release.
pub(crate) const MODEL_RVM_ONNX: &[u8] = include_bytes!("../../../models/rvm.onnx");

fn main() -> Result<()> {
    // Default filter: app logs at info, but silence zbus' connection
    // handshake chatter and the tracing crate's own spans (zbus uses
    // tracing internally and a tracing→log bridge picks them up).
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,zbus=warn,tracing=warn"),
    )
    .init();

    if std::env::var_os("LB_DUMP_ICON").is_some() {
        let icon = icon::build();
        let img: image::ImageBuffer<image::Rgba<u8>, _> =
            image::ImageBuffer::from_raw(icon.width, icon.height, icon.rgba.clone())
                .expect("icon buffer");
        img.save("/tmp/lb-icon.png")?;
        println!("wrote /tmp/lb-icon.png ({}×{})", icon.width, icon.height);
        return Ok(());
    }

    // First chance to short-circuit: if a sibling instance is already
    // serving the freedesktop activation interface, ask it to raise its
    // window and exit cleanly. Done before the flock so the user-visible
    // path on a second launch is "shortcut click → window reappears",
    // not "shortcut click → silent no-op".
    if activation::try_activate_existing() {
        return Ok(());
    }

    // Process-level single-instance lock. With lazy mode, the pipeline
    // can sit idle (no v4l2sink contention) for arbitrarily long, so we
    // need a real lock that covers the whole LB process — otherwise two
    // LB instances would both poll for consumers and race when one
    // arrives. Held until `main` returns. Also the fallback "another
    // instance is running" gate when D-Bus is unavailable (sandbox,
    // container, broken session bus).
    let _instance_lock = match InstanceLock::try_acquire()? {
        Some(l) => l,
        None => {
            log::info!("another LinuxBroadcast instance is already running; exiting");
            return Ok(());
        }
    };

    // Now that we own the flock, claim the D-Bus name so future second
    // launches activate us via `try_activate_existing` above. Held for
    // the process lifetime; dropping releases the name.
    //
    // Two shared slots, populated by the egui app on first frame:
    // - `waker`: lets the D-Bus and tray handlers nudge the egui loop
    //   immediately so it drains channels without waiting on its idle
    //   timer (Wayland-hidden windows can pause the loop for many
    //   seconds).
    // - `wayland_handles`: lets the D-Bus handler apply
    //   xdg-activation tokens directly from the worker thread,
    //   bypassing the egui loop entirely. The compositor refuses to
    //   send frame callbacks to a hidden xdg-toplevel, so
    //   `Window::request_redraw()` doesn't trigger `RedrawRequested`
    //   in that state — by the time the egui loop wakes up, the
    //   activation token has typically expired in the compositor.
    let waker: EguiWaker = Arc::new(OnceLock::new());
    let wayland_handles: WaylandHandleSlot = Arc::new(OnceLock::new());
    let (activation_tx, activation_rx) = crossbeam_channel::unbounded::<ActivationEvent>();
    let _activation_service = match activation::serve(
        activation_tx,
        Arc::clone(&waker),
        Arc::clone(&wayland_handles),
    ) {
        Ok(handle) => handle,
        Err(e) => {
            log::warn!("D-Bus activation service unavailable: {e:#}");
            None
        }
    };

    // Headless mode = same eframe loop, but starts with the window
    // hidden in the tray and auto-starts the pipeline. Same UX as the
    // old separate code path (autostart on login, no window flash), but
    // a single, testable code path. Triggered by `--headless` (used by
    // the autostart .desktop) or `LB_HEADLESS=1` (kept for back-compat).
    let argv: Vec<String> = std::env::args().collect();
    let headless =
        argv.iter().any(|a| a == "--headless") || std::env::var_os("LB_HEADLESS").is_some();

    if !headless && let Err(e) = desktop_install::ensure_desktop_entry() {
        log::warn!("desktop entry install: {e:#}");
    }

    ui::run(headless, activation_rx, waker, wayland_handles)
}
