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

//! linux-broadcast video pipeline.
//!
//! Pulls frames from a v4l2 capture device, segments the foreground via
//! ONNX Runtime (Robust Video Matting by default, MediaPipe multiclass as
//! a low-CPU fallback), composites against a blurred or replaced
//! background, and writes the result to a `v4l2loopback` device that
//! conferencing apps consume.
//!
//! The pipeline runs in **lazy** mode by default: the physical camera is
//! only opened while a real consumer is reading the virtual cam (or the
//! GUI's preview pane is asserting demand). See
//! [`pipeline::PipelineState`] and [`consumer_watch`] for the moving
//! parts.

pub mod compositor;
pub mod consumer_watch;
pub mod framing;
mod idle_loader;
pub mod lazy;
pub mod pipeline;
pub mod segmenter;
pub mod temporal;

pub use compositor::{Background, Compositor};
pub use consumer_watch::Consumer;
pub use pipeline::{Command, Pipeline, PipelineConfig, PipelineState, PreviewFrame};
pub use segmenter::{Mask, ModelKind, Segmenter};

/// Native input resolution of the MediaPipe Selfie Multiclass model
/// (256×256). RVM does not use these constants — its mask is emitted at
/// frame resolution.
pub const MODEL_W: usize = 256;
pub const MODEL_H: usize = 256;
