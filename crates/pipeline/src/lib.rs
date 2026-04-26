//! linux-broadcast video pipeline.
//!
//! Pulls frames from a v4l2 capture device, segments the foreground via
//! MediaPipe Selfie Segmentation (Landscape) running on `tract`, composites
//! against a blurred or replaced background, and writes the result to a
//! `v4l2loopback` device that conferencing apps consume.

pub mod compositor;
pub mod pipeline;
pub mod segmenter;
pub mod temporal;

pub use compositor::{Background, Compositor};
pub use pipeline::{Pipeline, PipelineConfig};
pub use segmenter::Segmenter;

/// Native input resolution of the MediaPipe Selfie Segmentation (general) model.
/// The "landscape" variant is 256×144 but isn't on the `onnx-community` HF repo;
/// the general model is 256×256 and otherwise identical in op set / output.
pub const MODEL_W: usize = 256;
pub const MODEL_H: usize = 256;
