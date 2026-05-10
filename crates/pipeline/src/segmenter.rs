//! Foreground segmentation. Wraps two interchangeable ONNX models
//! behind the [`Segmenter`] enum, each with its own pre/post:
//!
//! - **MediaPipe Selfie multiclass** — NHWC input, 6-class raw logits
//!   output. Foreground = `1 - softmax(logits)[bg]`. Fixed 256×256.
//! - **Robust Video Matting (RVM)** — recurrent. Six inputs (`src` + four
//!   recurrent state tensors `r1i..r4i` + `downsample_ratio` scalar) and
//!   six outputs (`fgr`, `pha`, `r1o..r4o`). We discard `fgr` and use
//!   `pha`. State lives on `RvmInner` across calls and is cleared by
//!   `reset()` on passthrough toggle or source-size change.
//!
//! Inference always runs at the model's native resolution
//! (`MODEL_W × MODEL_H` = 256×256 for multiclass, an internal
//! `RVM_DOWNSAMPLE_RATIO`-scaled tensor for RVM). Only the upsample +
//! composite step in `compositor.rs` touches frame-resolution pixels.
//! That's what keeps CPU usage low at 720p/1080p.
//!
//! All resizes use `fast_image_resize`; both `MpInner` and `RvmInner`
//! keep their input/output buffers in `self` so steady-state segmentation
//! does no heap allocation.

use anyhow::{Result, anyhow};
use fast_image_resize::{
    self as fr, FilterType, ResizeAlg, ResizeOptions,
    images::{Image, ImageRef},
};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::TensorRef;

use crate::{MODEL_H, MODEL_W};

/// Which segmentation model the pipeline should run with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelKind {
    /// MediaPipe Selfie Multiclass — 6 classes, 256×256 NHWC, raw logits.
    SelfieMulticlass,
    /// Robust Video Matting (MobileNetV3) — recurrent, frame-resolution
    /// alpha output, internal compute scaled by `downsample_ratio`.
    #[default]
    Rvm,
}

/// A segmentation mask with its native resolution. Variants of `Segmenter`
/// may return masks at different sizes — RVM at frame size, multiclass at
/// 256×256 — so callers must respect `width`/`height`.
pub struct Mask {
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

/// Public segmenter — internal implementation switches on `ModelKind`.
pub enum Segmenter {
    Multiclass(MpInner),
    // Boxed because `RvmInner` carries the four recurrent-state buffers
    // and is significantly larger than `MpInner`. Heap allocation cost
    // is paid once at construction; per-frame dispatch is unchanged.
    Rvm(Box<RvmInner>),
}

/// Session + reusable buffers for the multiclass MediaPipe model.
pub struct MpInner {
    session: Session,
    input_name: String,
    resizer: fr::Resizer,
    /// Reusable input buffer in float32. Layout depends on `kind`.
    input_buf: Vec<f32>,
    rgba_resized: Vec<u8>,
}

impl Segmenter {
    pub fn from_bytes(kind: ModelKind, onnx: &[u8]) -> Result<Self> {
        match kind {
            ModelKind::SelfieMulticlass => Ok(Segmenter::Multiclass(MpInner::new(onnx)?)),
            ModelKind::Rvm => Ok(Segmenter::Rvm(Box::new(RvmInner::new(onnx)?))),
        }
    }

    pub fn segment(&mut self, rgba: &[u8], width: usize, height: usize) -> Result<Mask> {
        match self {
            Segmenter::Multiclass(inner) => segment_multiclass(inner, rgba, width, height),
            Segmenter::Rvm(inner) => inner.segment(rgba, width, height),
        }
    }

    /// Drop any temporal state (used when toggling to passthrough or when
    /// the source frame size changes).
    pub fn reset(&mut self) {
        if let Segmenter::Rvm(inner) = self {
            inner.reset();
        }
    }
}

impl MpInner {
    fn new(onnx: &[u8]) -> Result<Self> {
        let session = build_session(onnx)?;
        let input_name = session
            .inputs()
            .first()
            .ok_or_else(|| anyhow!("model has no inputs"))?
            .name()
            .to_string();
        Ok(MpInner {
            session,
            input_name,
            resizer: fr::Resizer::new(),
            input_buf: vec![0.0_f32; MODEL_H * MODEL_W * 3],
            rgba_resized: vec![0_u8; MODEL_H * MODEL_W * 4],
        })
    }
}

fn build_session(onnx: &[u8]) -> Result<Session> {
    Session::builder()
        .map_err(|e| anyhow!("ort Session::builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("set optimization level: {e}"))?
        .with_intra_threads(num_threads())
        .map_err(|e| anyhow!("set intra threads: {e}"))?
        .commit_from_memory(onnx)
        .map_err(|e| anyhow!("commit ONNX model from memory: {e}"))
}

fn resize_opts() -> ResizeOptions {
    ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear))
}

/// Resize source RGBA into the model's native 256×256, in-place into
/// `inner.rgba_resized`.
fn resize_to_model(inner: &mut MpInner, rgba: &[u8], width: usize, height: usize) -> Result<()> {
    if rgba.len() != width * height * 4 {
        return Err(anyhow!(
            "rgba buffer size {} != {}*{}*4",
            rgba.len(),
            width,
            height
        ));
    }
    let src = ImageRef::new(width as u32, height as u32, rgba, fr::PixelType::U8x4)
        .map_err(|e| anyhow!("source image: {e}"))?;
    let mut dst = Image::from_slice_u8(
        MODEL_W as u32,
        MODEL_H as u32,
        &mut inner.rgba_resized,
        fr::PixelType::U8x4,
    )
    .map_err(|e| anyhow!("dst image: {e}"))?;
    inner
        .resizer
        .resize(&src, &mut dst, &resize_opts())
        .map_err(|e| anyhow!("resize: {e}"))
}

// -----------------------------------------------------------------------
//  Multiclass (MediaPipe Selfie Multiclass, 6 channels, NHWC 256×256)
// -----------------------------------------------------------------------

const MULTICLASS_BG: usize = 0;

fn segment_multiclass(
    inner: &mut MpInner,
    rgba: &[u8],
    width: usize,
    height: usize,
) -> Result<Mask> {
    resize_to_model(inner, rgba, width, height)?;

    let plane = MODEL_H * MODEL_W;
    for i in 0..plane {
        let o = i * 3;
        inner.input_buf[o] = inner.rgba_resized[i * 4] as f32 / 255.0;
        inner.input_buf[o + 1] = inner.rgba_resized[i * 4 + 1] as f32 / 255.0;
        inner.input_buf[o + 2] = inner.rgba_resized[i * 4 + 2] as f32 / 255.0;
    }

    let shape: [i64; 4] = [1, MODEL_H as i64, MODEL_W as i64, 3];
    let input_value = TensorRef::from_array_view((shape, &inner.input_buf[..]))
        .map_err(|e| anyhow!("ort Value: {e}"))?;
    let outputs = inner
        .session
        .run(ort::inputs![&inner.input_name => input_value])
        .map_err(|e| anyhow!("ort run: {e}"))?;
    let (_name, value) = outputs.iter().next().ok_or_else(|| anyhow!("no outputs"))?;
    let (out_shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow!("extract: {e}"))?;

    let n = MODEL_H * MODEL_W;
    let mut mask = vec![0.0_f32; n];
    match &out_shape[..] {
        [1, h, w, 6] if *h as usize == MODEL_H && *w as usize == MODEL_W => {
            for (out, logits) in mask.iter_mut().zip(data.chunks_exact(6)) {
                let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0_f32;
                let mut exp_bg = 0.0_f32;
                for (c, &v) in logits.iter().enumerate() {
                    let e = (v - max).exp();
                    if c == MULTICLASS_BG {
                        exp_bg = e;
                    }
                    sum += e;
                }
                *out = (1.0 - exp_bg / sum).clamp(0.0, 1.0);
            }
        }
        other => {
            return Err(anyhow!(
                "unexpected multiclass output shape {:?}, want [1,{},{},6]",
                other,
                MODEL_H,
                MODEL_W
            ));
        }
    }
    Ok(Mask {
        data: mask,
        width: MODEL_W as u32,
        height: MODEL_H as u32,
    })
}

// -----------------------------------------------------------------------
//  Robust Video Matting (RVM)
// -----------------------------------------------------------------------

/// Internal compute resolution = input × downsample_ratio. The official
/// recommendation is 0.375–0.4 at 720p for speed and 0.5 at 720p for
/// "higher quality" (sharper hair / shoulder edges). 0.5 → 640×360
/// internal compute on a 1280×720 frame; ~50 % more inference cost than
/// 0.4 in exchange for visibly tighter mattes.
const RVM_DOWNSAMPLE_RATIO: f32 = 0.50;

pub struct RvmInner {
    session: Session,
    /// Initial-frame recurrent state: zeros at shape `[1, C, 1, 1]` per
    /// state. Built once at construction, fed in on the first call after
    /// `reset()`. RVM's graph accepts any spatial size for `r*i` and
    /// broadcasts/upsamples internally.
    initial_states: [(Vec<i64>, Vec<f32>); 4],
    /// Recurrent state from the previous frame — reused across calls.
    /// Set to `None` until the first frame runs (or after `reset()`),
    /// in which case `initial_states` is fed instead. After the first
    /// call, the four `(Vec<i64>, Vec<f32>)` buffers are kept across
    /// frames and overwritten in place from each new output, so steady
    /// state allocates nothing.
    prev_states: Option<[(Vec<i64>, Vec<f32>); 4]>,
    /// Reusable f32 input buffer in NCHW layout.
    input_buf: Vec<f32>,
    last_dims: (u32, u32),
}

impl RvmInner {
    fn new(onnx: &[u8]) -> Result<Self> {
        Ok(RvmInner {
            session: build_session(onnx)?,
            initial_states: [
                (vec![1, 16, 1, 1], vec![0.0; 16]),
                (vec![1, 20, 1, 1], vec![0.0; 20]),
                (vec![1, 40, 1, 1], vec![0.0; 40]),
                (vec![1, 64, 1, 1], vec![0.0; 64]),
            ],
            prev_states: None,
            input_buf: Vec::new(),
            last_dims: (0, 0),
        })
    }

    fn reset(&mut self) {
        self.prev_states = None;
    }

    fn segment(&mut self, rgba: &[u8], width: usize, height: usize) -> Result<Mask> {
        if rgba.len() != width * height * 4 {
            return Err(anyhow!(
                "rgba buffer size {} != {}*{}*4",
                rgba.len(),
                width,
                height
            ));
        }

        // Reset recurrent state if the source dimensions change.
        if self.last_dims != (width as u32, height as u32) {
            self.prev_states = None;
            self.last_dims = (width as u32, height as u32);
        }

        // 1. Pack RGBA → NCHW f32 in [0, 1]. Plane order R, G, B.
        let plane = width * height;
        let needed = plane * 3;
        if self.input_buf.len() != needed {
            self.input_buf.resize(needed, 0.0);
        }
        for i in 0..plane {
            self.input_buf[i] = rgba[i * 4] as f32 / 255.0;
            self.input_buf[plane + i] = rgba[i * 4 + 1] as f32 / 255.0;
            self.input_buf[2 * plane + i] = rgba[i * 4 + 2] as f32 / 255.0;
        }

        // 2. Build inputs: src + 4 recurrent states + downsample_ratio scalar.
        // All input tensors borrow from buffers we own (`self.input_buf`,
        // `self.prev_states` / `self.initial_states`, `ratio`) — no
        // per-frame copies into the ORT runtime.
        let src_shape: [i64; 4] = [1, 3, height as i64, width as i64];
        let src_value = TensorRef::from_array_view((src_shape, &self.input_buf[..]))
            .map_err(|e| anyhow!("ort Value src: {e}"))?;

        let states = self.prev_states.as_ref().unwrap_or(&self.initial_states);

        let mut state_values = Vec::with_capacity(4);
        for (shape, data) in states {
            let v = TensorRef::from_array_view((&shape[..], &data[..]))
                .map_err(|e| anyhow!("ort Value state: {e}"))?;
            state_values.push(v);
        }
        let [r1i, r2i, r3i, r4i] = state_values
            .try_into()
            .map_err(|_| anyhow!("expected 4 state values"))?;

        let ratio = [RVM_DOWNSAMPLE_RATIO];
        let ratio_value = TensorRef::from_array_view(([1_i64], &ratio[..]))
            .map_err(|e| anyhow!("ort Value downsample_ratio: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![
                "src" => src_value,
                "r1i" => r1i,
                "r2i" => r2i,
                "r3i" => r3i,
                "r4i" => r4i,
                "downsample_ratio" => ratio_value,
            ])
            .map_err(|e| anyhow!("ort run rvm: {e}"))?;

        // 3. Pull `pha` (alpha matte) and the four `r*o` for next frame.
        let (pha_shape, pha_data) = outputs
            .get("pha")
            .ok_or_else(|| anyhow!("rvm output `pha` missing"))?
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("extract pha: {e}"))?;

        let mask = match &pha_shape[..] {
            [1, 1, h, w] if *h as usize == height && *w as usize == width => pha_data
                .iter()
                .map(|v| v.clamp(0.0, 1.0))
                .collect::<Vec<f32>>(),
            other => {
                return Err(anyhow!(
                    "unexpected rvm pha shape {:?}, want [1,1,{},{}]",
                    other,
                    height,
                    width
                ));
            }
        };

        // 4. Capture next-frame recurrent states. Reuses the existing
        // `prev_states` buffers in place so steady-state inference does
        // no per-frame allocation; on the very first frame (when
        // `prev_states` is None) we seed from the initial-state shapes
        // and let `resize` do a one-time grow.
        let mut next = self
            .prev_states
            .take()
            .unwrap_or_else(|| self.initial_states.clone());
        for (i, name) in ["r1o", "r2o", "r3o", "r4o"].iter().enumerate() {
            let (shape, data) = outputs
                .get(*name)
                .ok_or_else(|| anyhow!("rvm output `{name}` missing"))?
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow!("extract {name}: {e}"))?;
            next[i].0.clear();
            next[i].0.extend(shape.iter().copied());
            if next[i].1.len() != data.len() {
                next[i].1.resize(data.len(), 0.0);
            }
            next[i].1.copy_from_slice(data);
        }
        self.prev_states = Some(next);

        Ok(Mask {
            data: mask,
            width: width as u32,
            height: height as u32,
        })
    }
}

/// Hard cap on intra-op threads. ORT + CPU EP gets diminishing returns
/// past 4 threads on the workloads we run (256×256 conv stacks), and
/// burning more cores hurts when other apps share the host.
const MAX_INTRA_THREADS: usize = 4;
const FALLBACK_THREADS: usize = 2;

fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(FALLBACK_THREADS)
        .clamp(1, MAX_INTRA_THREADS)
}
