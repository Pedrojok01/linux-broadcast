use anyhow::{anyhow, Result};
use fast_image_resize::{
    self as fr,
    images::{Image, ImageRef},
    FilterType, ResizeAlg, ResizeOptions,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;

use crate::{MODEL_H, MODEL_W};

/// Which segmentation model the pipeline should run with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelKind {
    /// MediaPipe Selfie Segmentation, single-channel output.
    /// 256×256, NCHW input/output, output is already a probability.
    #[default]
    SelfieBinary,
    /// MediaPipe Selfie Multiclass (6 classes: background, hair, body-skin,
    /// face-skin, clothes, others). 256×256, NHWC input/output, raw logits.
    /// Foreground = `1 - softmax(logits)[bg]` per pixel.
    SelfieMulticlass,
}

/// Public segmenter — internal implementation switches on `ModelKind`.
pub enum Segmenter {
    Binary(Inner),
    Multiclass(Inner),
}

/// Shared session + reusable buffers. The actual layout (NCHW/NHWC) lives in
/// the `segment` arm.
pub struct Inner {
    session: Session,
    input_name: String,
    resizer: fr::Resizer,
    /// Reusable input buffer in float32. Layout depends on `kind`.
    input_buf: Vec<f32>,
    rgba_resized: Vec<u8>,
}

impl Segmenter {
    pub fn from_bytes(kind: ModelKind, onnx: &[u8]) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow!("ort Session::builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("set optimization level: {e}"))?
            .with_intra_threads(num_threads())
            .map_err(|e| anyhow!("set intra threads: {e}"))?
            .commit_from_memory(onnx)
            .map_err(|e| anyhow!("commit ONNX model from memory: {e}"))?;
        let input_name = session
            .inputs()
            .first()
            .ok_or_else(|| anyhow!("model has no inputs"))?
            .name()
            .to_string();

        // Both models are 256×256×3 input — one NCHW, one NHWC. Same buffer
        // size, just different element ordering.
        let inner = Inner {
            session,
            input_name,
            resizer: fr::Resizer::new(),
            input_buf: vec![0.0_f32; MODEL_H * MODEL_W * 3],
            rgba_resized: vec![0_u8; MODEL_H * MODEL_W * 4],
        };
        Ok(match kind {
            ModelKind::SelfieBinary => Segmenter::Binary(inner),
            ModelKind::SelfieMulticlass => Segmenter::Multiclass(inner),
        })
    }

    pub fn segment(
        &mut self,
        rgba: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>> {
        match self {
            Segmenter::Binary(inner) => segment_binary(inner, rgba, width, height),
            Segmenter::Multiclass(inner) => segment_multiclass(inner, rgba, width, height),
        }
    }
}

fn resize_opts() -> ResizeOptions {
    ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear))
}

/// Resize + drop-alpha shared step: produces 256×256 RGBA in `rgba_resized`.
fn resize_to_model(inner: &mut Inner, rgba: &[u8], width: usize, height: usize) -> Result<()> {
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
//  Binary (MediaPipe Selfie Segmentation, single channel, NCHW)
// -----------------------------------------------------------------------

fn segment_binary(
    inner: &mut Inner,
    rgba: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<f32>> {
    resize_to_model(inner, rgba, width, height)?;

    // NCHW preprocessing — R-plane, G-plane, B-plane.
    let plane = MODEL_H * MODEL_W;
    for i in 0..plane {
        inner.input_buf[i] = inner.rgba_resized[i * 4] as f32 / 255.0;
        inner.input_buf[plane + i] = inner.rgba_resized[i * 4 + 1] as f32 / 255.0;
        inner.input_buf[2 * plane + i] = inner.rgba_resized[i * 4 + 2] as f32 / 255.0;
    }

    let shape: [i64; 4] = [1, 3, MODEL_H as i64, MODEL_W as i64];
    let input_value = Value::from_array((shape, inner.input_buf.clone()))
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
        [1, 1, h, w] if *h as usize == MODEL_H && *w as usize == MODEL_W => {
            // Output is already a probability (sigmoid is part of the graph).
            for (i, &v) in data.iter().enumerate() {
                mask[i] = v.clamp(0.0, 1.0);
            }
        }
        other => {
            return Err(anyhow!(
                "unexpected binary output shape {:?}, want [1,1,{},{}]",
                other,
                MODEL_H,
                MODEL_W
            ));
        }
    }
    Ok(mask)
}

// -----------------------------------------------------------------------
//  Multiclass (MediaPipe Selfie Multiclass, 6 channels, NHWC)
// -----------------------------------------------------------------------

const MULTICLASS_BG: usize = 0;

fn segment_multiclass(
    inner: &mut Inner,
    rgba: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<f32>> {
    resize_to_model(inner, rgba, width, height)?;

    // NHWC preprocessing — interleaved R,G,B per pixel.
    let plane = MODEL_H * MODEL_W;
    for i in 0..plane {
        let o = i * 3;
        inner.input_buf[o]     = inner.rgba_resized[i * 4]     as f32 / 255.0;
        inner.input_buf[o + 1] = inner.rgba_resized[i * 4 + 1] as f32 / 255.0;
        inner.input_buf[o + 2] = inner.rgba_resized[i * 4 + 2] as f32 / 255.0;
    }

    let shape: [i64; 4] = [1, MODEL_H as i64, MODEL_W as i64, 3];
    let input_value = Value::from_array((shape, inner.input_buf.clone()))
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
            // NHWC, 6 raw logits per pixel. Foreground = 1 - softmax[bg].
            for i in 0..n {
                let o = i * 6;
                let logits = &data[o..o + 6];
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
                mask[i] = (1.0 - exp_bg / sum).clamp(0.0, 1.0);
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
    Ok(mask)
}

fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(4)
        .max(1)
}
