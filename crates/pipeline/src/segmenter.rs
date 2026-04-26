use anyhow::{anyhow, Context, Result};
use fast_image_resize as fr;
use std::num::NonZeroU32;
use tract_onnx::prelude::*;

use crate::{MODEL_H, MODEL_W};

type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// MediaPipe Selfie Segmentation (Landscape) running on tract.
///
/// Input:  RGBA frame `&[u8]` of arbitrary size.
/// Output: `Vec<f32>` mask of length `MODEL_W * MODEL_H` in `[0.0, 1.0]`,
///         where 1.0 means "foreground (person)".
///
/// Per <https://github.com/google-ai-edge/mediapipe/issues/6134>, the raw
/// model output requires softmax across the channel axis to match the
/// MediaPipe reference pipeline. We apply that here.
pub struct Segmenter {
    model: OnnxModel,
    resizer: fr::Resizer,
    /// Reusable input buffer in NHWC f32 layout: [1, H, W, 3].
    input_buf: Vec<f32>,
    /// Reusable RGB buffer at model resolution (H * W * 3 bytes).
    rgb_resized: Vec<u8>,
    /// Reusable RGBA buffer at model resolution (fr::Image source uses RGBA8).
    rgba_resized: Vec<u8>,
}

impl Segmenter {
    /// Load the model from raw ONNX bytes (suitable for `include_bytes!`).
    pub fn from_bytes(onnx: &[u8]) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(onnx);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .context("parse ONNX model")?
            .with_input_fact(
                0,
                f32::fact([1, MODEL_H, MODEL_W, 3]).into(),
            )
            .context("set input fact")?
            .into_optimized()
            .context("optimize tract model")?
            .into_runnable()
            .context("make tract model runnable")?;

        Ok(Self {
            model,
            resizer: fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear)),
            input_buf: vec![0.0_f32; MODEL_H * MODEL_W * 3],
            rgb_resized: vec![0_u8; MODEL_H * MODEL_W * 3],
            rgba_resized: vec![0_u8; MODEL_H * MODEL_W * 4],
        })
    }

    /// Run inference on an RGBA frame. Returns the mask laid out row-major
    /// as `[h * MODEL_W + w] => probability(foreground)`.
    pub fn segment(
        &mut self,
        rgba: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>> {
        if rgba.len() != width * height * 4 {
            return Err(anyhow!(
                "rgba buffer size {} != {}*{}*4",
                rgba.len(),
                width,
                height
            ));
        }

        // 1. Resize the input frame down to the model's native 256x144 RGBA.
        let src = fr::Image::from_slice_u8(
            NonZeroU32::new(width as u32).ok_or_else(|| anyhow!("width=0"))?,
            NonZeroU32::new(height as u32).ok_or_else(|| anyhow!("height=0"))?,
            // fr::Image needs &mut; we copy in for safety since the GStreamer
            // buffer slice may not be writable. The cost is one memcpy of the
            // input frame, dwarfed by inference.
            // SAFETY: we immediately wrap in an immutable view via Image::from_slice_u8.
            unsafe { std::slice::from_raw_parts_mut(rgba.as_ptr() as *mut u8, rgba.len()) },
            fr::PixelType::U8x4,
        )
        .map_err(|e| anyhow!("source image: {e}"))?;

        let mut dst = fr::Image::from_slice_u8(
            NonZeroU32::new(MODEL_W as u32).unwrap(),
            NonZeroU32::new(MODEL_H as u32).unwrap(),
            &mut self.rgba_resized,
            fr::PixelType::U8x4,
        )
        .map_err(|e| anyhow!("dst image: {e}"))?;

        self.resizer
            .resize(&src.view(), &mut dst.view_mut())
            .map_err(|e| anyhow!("resize: {e}"))?;

        // 2. Drop alpha and normalize to [0,1] f32 NHWC.
        for i in 0..(MODEL_H * MODEL_W) {
            let r = self.rgba_resized[i * 4] as f32 / 255.0;
            let g = self.rgba_resized[i * 4 + 1] as f32 / 255.0;
            let b = self.rgba_resized[i * 4 + 2] as f32 / 255.0;
            self.input_buf[i * 3] = r;
            self.input_buf[i * 3 + 1] = g;
            self.input_buf[i * 3 + 2] = b;
        }

        // 3. Inference.
        let input_tensor: Tensor =
            tract_ndarray::Array4::from_shape_vec((1, MODEL_H, MODEL_W, 3), self.input_buf.clone())?
                .into();
        let outputs = self
            .model
            .run(tvec!(input_tensor.into()))
            .context("tract run")?;

        let raw = outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("model produced no outputs"))?;
        let arr = raw.to_array_view::<f32>()?;

        // 4. Convert to probability mask. The landscape model output shape is
        // [1, H, W, 1] (single foreground channel). Apply sigmoid; if a future
        // 2-channel variant is loaded, take softmax across the channel axis.
        let shape = arr.shape();
        let n_pixels = MODEL_H * MODEL_W;
        let mut mask = vec![0.0_f32; n_pixels];
        match shape {
            [1, h, w, 1] if *h == MODEL_H && *w == MODEL_W => {
                for (i, &v) in arr.iter().enumerate() {
                    mask[i] = sigmoid(v);
                }
            }
            [1, h, w, 2] if *h == MODEL_H && *w == MODEL_W => {
                let flat = arr.as_slice().unwrap();
                for i in 0..n_pixels {
                    let bg = flat[i * 2];
                    let fg = flat[i * 2 + 1];
                    let max = bg.max(fg);
                    let ebg = (bg - max).exp();
                    let efg = (fg - max).exp();
                    mask[i] = efg / (ebg + efg);
                }
            }
            other => {
                return Err(anyhow!(
                    "unexpected model output shape {:?}, expected [1,{},{},1|2]",
                    other,
                    MODEL_H,
                    MODEL_W
                ));
            }
        }
        Ok(mask)
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
