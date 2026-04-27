use anyhow::{anyhow, Result};
use fast_image_resize::{
    self as fr,
    images::{Image, ImageRef},
    FilterType, ResizeAlg, ResizeOptions,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;

use crate::{MODEL_H, MODEL_W};

/// MediaPipe Selfie Segmentation running on `ort` (ONNX Runtime, CPU EP).
///
/// Input:  RGBA frame `&[u8]` of arbitrary size.
/// Output: `Vec<f32>` mask of length `MODEL_W * MODEL_H` in `[0.0, 1.0]`,
///         where 1.0 means "foreground (person)".
///
/// Per <https://github.com/google-ai-edge/mediapipe/issues/6134>, the raw
/// model output requires sigmoid (single-channel variant) or softmax across
/// the channel axis (2-channel variant) to match MediaPipe's reference
/// pipeline. We apply that here.
pub struct Segmenter {
    session: Session,
    input_name: String,
    resizer: fr::Resizer,
    /// Reusable input buffer in NCHW f32 layout: [1, 3, H, W].
    input_buf: Vec<f32>,
    /// Reusable RGBA buffer at model resolution.
    rgba_resized: Vec<u8>,
}

impl Segmenter {
    /// Load the model from raw ONNX bytes (suitable for `include_bytes!`).
    pub fn from_bytes(onnx: &[u8]) -> Result<Self> {
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

        Ok(Self {
            session,
            input_name,
            resizer: fr::Resizer::new(),
            input_buf: vec![0.0_f32; MODEL_H * MODEL_W * 3],
            rgba_resized: vec![0_u8; MODEL_H * MODEL_W * 4],
        })
    }

    fn resize_opts() -> ResizeOptions {
        ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear))
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

        // 1. Resize the input frame down to the model's native 256x256 RGBA.
        let src = ImageRef::new(width as u32, height as u32, rgba, fr::PixelType::U8x4)
            .map_err(|e| anyhow!("source image: {e}"))?;
        let mut dst = Image::from_slice_u8(
            MODEL_W as u32,
            MODEL_H as u32,
            &mut self.rgba_resized,
            fr::PixelType::U8x4,
        )
        .map_err(|e| anyhow!("dst image: {e}"))?;
        self.resizer
            .resize(&src, &mut dst, &Self::resize_opts())
            .map_err(|e| anyhow!("resize: {e}"))?;

        // 2. Drop alpha and normalize to [0,1] f32 in NCHW layout (model is
        //    [batch, 3, H, W]). R plane first, then G, then B.
        let plane = MODEL_H * MODEL_W;
        for i in 0..plane {
            let r = self.rgba_resized[i * 4] as f32 / 255.0;
            let g = self.rgba_resized[i * 4 + 1] as f32 / 255.0;
            let b = self.rgba_resized[i * 4 + 2] as f32 / 255.0;
            self.input_buf[i] = r;
            self.input_buf[plane + i] = g;
            self.input_buf[2 * plane + i] = b;
        }

        // 3. Inference. Use the (shape, Vec<T>) form to avoid pulling in ort's
        //    ndarray version.
        let shape: [i64; 4] = [1, 3, MODEL_H as i64, MODEL_W as i64];
        let input_value = Value::from_array((shape, self.input_buf.clone()))
            .map_err(|e| anyhow!("ort Value from array: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![&self.input_name => input_value])
            .map_err(|e| anyhow!("ort session.run: {e}"))?;

        // 4. Extract the (only) output. Apply sigmoid for [1,1,H,W] or softmax
        //    across channels for [1,2,H,W].
        let (out_name, out_value) = outputs
            .iter()
            .next()
            .ok_or_else(|| anyhow!("model produced no outputs"))?;
        let (shape, data) = out_value
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("extract f32 output tensor: {e}"))?;
        let n_pixels = MODEL_H * MODEL_W;
        let mut mask = vec![0.0_f32; n_pixels];
        match &shape[..] {
            [1, 1, h, w] if *h as usize == MODEL_H && *w as usize == MODEL_W => {
                // Output is already in [0, 1] (final sigmoid is part of the
                // ONNX graph for this variant). Just clamp for safety.
                for (i, &v) in data.iter().enumerate() {
                    mask[i] = v.clamp(0.0, 1.0);
                }
            }
            [1, 2, h, w] if *h as usize == MODEL_H && *w as usize == MODEL_W => {
                for i in 0..n_pixels {
                    let bg = data[i];
                    let fg = data[n_pixels + i];
                    let max = bg.max(fg);
                    let ebg = (bg - max).exp();
                    let efg = (fg - max).exp();
                    mask[i] = efg / (ebg + efg);
                }
            }
            other => {
                return Err(anyhow!(
                    "output {:?} shape {:?} unexpected; want [1,1|2,{},{}]",
                    out_name,
                    other,
                    MODEL_H,
                    MODEL_W
                ));
            }
        }
        Ok(mask)
    }
}

fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(4)
        .max(1)
}
