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
//! - Embed the three ONNX models at compile time (`include_bytes!`) so the
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

use anyhow::Result;

use crate::lock::InstanceLock;

/// Bundled MediaPipe Selfie Segmentation ONNX (binary, ~450 KB).
/// Sourced from `onnx-community/mediapipe_selfie_segmentation` on Hugging Face.
pub(crate) const MODEL_BINARY_ONNX: &[u8] = include_bytes!("../../../models/selfie_segmenter.onnx");

/// Bundled MediaPipe Selfie Multiclass ONNX (6 classes: bg/hair/body/face/clothes/other,
/// ~16 MB). Converted via tf2onnx from the official MediaPipe TFLite.
pub(crate) const MODEL_MULTICLASS_ONNX: &[u8] =
    include_bytes!("../../../models/selfie_multiclass.onnx");

/// Bundled Robust Video Matting ONNX (MobileNetV3 backbone, fp32, ~15 MB).
/// Sourced from PeterL1n/RobustVideoMatting v1.0.0 release.
pub(crate) const MODEL_RVM_ONNX: &[u8] = include_bytes!("../../../models/rvm.onnx");

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    if std::env::var_os("LB_DUMP_ICON").is_some() {
        let icon = icon::build();
        let img: image::ImageBuffer<image::Rgba<u8>, _> =
            image::ImageBuffer::from_raw(icon.width, icon.height, icon.rgba.clone())
                .expect("icon buffer");
        img.save("/tmp/lb-icon.png")?;
        println!("wrote /tmp/lb-icon.png ({}×{})", icon.width, icon.height);
        return Ok(());
    }

    // Process-level single-instance lock. With lazy mode, the pipeline
    // can sit idle (no v4l2sink contention) for arbitrarily long, so we
    // need a real lock that covers the whole LB process — otherwise two
    // LB instances would both poll for consumers and race when one
    // arrives. Held until `main` returns.
    let _instance_lock = match InstanceLock::try_acquire()? {
        Some(l) => l,
        None => {
            log::info!("another LinuxBroadcast instance is already running; exiting");
            return Ok(());
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

    if !headless {
        if let Err(e) = desktop_install::ensure_desktop_entry() {
            log::warn!("desktop entry install: {e:#}");
        }
    }

    ui::run(headless)
}
