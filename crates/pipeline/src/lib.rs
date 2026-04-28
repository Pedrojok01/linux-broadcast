//! linux-broadcast video pipeline.
//!
//! Pulls frames from a v4l2 capture device, segments the foreground via
//! ONNX Runtime (MediaPipe Selfie Segmentation, MediaPipe multiclass, or
//! Robust Video Matting), composites against a blurred or replaced
//! background, and writes the result to a `v4l2loopback` device that
//! conferencing apps consume.
//!
//! The pipeline runs in **lazy** mode by default: the physical camera is
//! only opened while a real consumer is reading the virtual cam (or the
//! GUI's preview pane / `force_on` toggle is asserting demand). See
//! [`pipeline::PipelineState`] and [`consumer_watch`] for the moving
//! parts.

pub mod compositor;
pub mod consumer_watch;
pub mod lazy;
pub mod pipeline;
pub mod segmenter;
pub mod temporal;

pub use compositor::{Background, Compositor};
pub use consumer_watch::Consumer;
pub use pipeline::{Command, Pipeline, PipelineConfig, PipelineState, PreviewFrame};
pub use segmenter::{Mask, ModelKind, Segmenter};

/// Native input resolution of the MediaPipe Selfie Segmentation (general) model.
/// The "landscape" variant is 256×144 but isn't on the `onnx-community` HF repo;
/// the general model is 256×256 and otherwise identical in op set / output.
pub const MODEL_W: usize = 256;
pub const MODEL_H: usize = 256;
