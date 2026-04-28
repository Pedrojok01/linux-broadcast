use anyhow::{anyhow, Result};
use fast_image_resize::{
    self as fr,
    images::{Image, ImageRef},
    FilterType, ResizeAlg, ResizeOptions,
};

use crate::Mask;

/// Minimum kernel radius (px) at `strength = 0.0`. Below this the blur is
/// imperceptible and the foreground edge stops being legible.
pub const BLUR_MIN_RADIUS: usize = 4;
/// Maximum kernel radius (px) at `strength = 1.0`. Past ~32 px the
/// background becomes unreadable for any text or facial cues.
pub const BLUR_MAX_RADIUS: usize = 32;
/// Radius (px) at and above which the second blur pass kicks in to push
/// the box-kernel output closer to a Gaussian.
const BLUR_TWO_PASS_THRESHOLD: usize = 8;

/// Background mode the compositor should produce behind the foreground mask.
#[derive(Debug, Clone)]
pub enum Background {
    /// Pass the input frame through unchanged. Useful as the "off" state and
    /// for diagnostics. The compositor short-circuits both the segmentation
    /// upsample and the per-pixel blend.
    None,
    /// Gaussian-blur the original frame and use that as the background.
    /// `strength` is in `[0.0, 1.0]` and maps to a kernel radius from
    /// `BLUR_MIN_RADIUS` (barely-visible) to `BLUR_MAX_RADIUS` (strong).
    Blur { strength: f32 },
    /// Composite over a static RGBA image, scaled to cover the frame.
    Image {
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    },
}

impl Background {
    /// Default blur intensity (mid-strength).
    pub const DEFAULT_BLUR_STRENGTH: f32 = 0.62;
}

pub struct Compositor {
    resizer: fr::Resizer,
    /// Mask upsampled to frame resolution, kept f32 across frames to avoid
    /// reallocating each callback.
    upsampled_mask: Vec<f32>,
    /// Pre-scaled background image at frame resolution, regenerated when the
    /// background or frame size changes.
    bg_scaled: Vec<u8>,
    bg_w: u32,
    bg_h: u32,
}

impl Compositor {
    pub fn new() -> Self {
        Self {
            resizer: fr::Resizer::new(),
            upsampled_mask: Vec::new(),
            bg_scaled: Vec::new(),
            bg_w: 0,
            bg_h: 0,
        }
    }

    fn resize_opts() -> ResizeOptions {
        ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear))
    }

    /// Composite an RGBA frame in place using the given mask and background.
    ///
    /// `mask.data` is foreground probabilities in `[0,1]`, sized
    /// `mask.width × mask.height`. The compositor handles upsampling to
    /// frame resolution if needed; if the mask already matches the frame
    /// (e.g. RVM at frame size) the upsample is a borrow.
    pub fn composite(
        &mut self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        mask: &Mask,
        background: &Background,
    ) -> Result<()> {
        if frame.len() != (width as usize) * (height as usize) * 4 {
            return Err(anyhow!("frame buffer size mismatch"));
        }
        let expected = (mask.width as usize) * (mask.height as usize);
        if mask.data.len() != expected {
            return Err(anyhow!(
                "mask data {} != {}*{}",
                mask.data.len(),
                mask.width,
                mask.height
            ));
        }

        // Short-circuit: passthrough mode skips the upsample + blend entirely.
        if matches!(background, Background::None) {
            return Ok(());
        }

        self.prepare_mask(mask, width, height)?;

        match background {
            Background::None => unreachable!(),
            Background::Blur { strength } => {
                composite_blur(frame, width, height, &self.upsampled_mask, *strength);
            }
            Background::Image {
                rgba: bg_rgba,
                width: bw,
                height: bh,
            } => {
                self.ensure_bg_scaled(bg_rgba, *bw, *bh, width, height)?;
                composite_image(frame, &self.bg_scaled, &self.upsampled_mask);
            }
        }
        Ok(())
    }

    /// Make sure `self.upsampled_mask` holds the mask at frame resolution.
    /// If the source already matches the frame size we just copy the slice;
    /// otherwise bilinear-upsample.
    fn prepare_mask(&mut self, mask: &Mask, width: u32, height: u32) -> Result<()> {
        let target = (width as usize) * (height as usize);
        if self.upsampled_mask.len() != target {
            self.upsampled_mask.resize(target, 0.0);
        }
        if mask.width == width && mask.height == height {
            self.upsampled_mask.copy_from_slice(&mask.data);
            return Ok(());
        }
        let src_w = mask.width as usize;
        let src_h = mask.height as usize;
        let src_w_f = src_w as f32;
        let src_h_f = src_h as f32;
        let dst_w = width as f32;
        let dst_h = height as f32;
        for y in 0..height {
            let sy = (y as f32 + 0.5) * src_h_f / dst_h - 0.5;
            let y0 = sy.floor().clamp(0.0, src_h_f - 1.0) as usize;
            let y1 = (y0 + 1).min(src_h - 1);
            let fy = (sy - y0 as f32).clamp(0.0, 1.0);
            let row_dst = (y as usize) * (width as usize);
            let row0 = y0 * src_w;
            let row1 = y1 * src_w;
            for x in 0..width {
                let sx = (x as f32 + 0.5) * src_w_f / dst_w - 0.5;
                let x0 = sx.floor().clamp(0.0, src_w_f - 1.0) as usize;
                let x1 = (x0 + 1).min(src_w - 1);
                let fx = (sx - x0 as f32).clamp(0.0, 1.0);
                let a = mask.data[row0 + x0];
                let b = mask.data[row0 + x1];
                let c = mask.data[row1 + x0];
                let d = mask.data[row1 + x1];
                let top = a * (1.0 - fx) + b * fx;
                let bot = c * (1.0 - fx) + d * fx;
                self.upsampled_mask[row_dst + x as usize] = top * (1.0 - fy) + bot * fy;
            }
        }
        Ok(())
    }

    fn ensure_bg_scaled(&mut self, bg: &[u8], bw: u32, bh: u32, fw: u32, fh: u32) -> Result<()> {
        if self.bg_w == fw && self.bg_h == fh && !self.bg_scaled.is_empty() {
            return Ok(());
        }
        self.bg_scaled.resize((fw as usize) * (fh as usize) * 4, 0);
        let src =
            ImageRef::new(bw, bh, bg, fr::PixelType::U8x4).map_err(|e| anyhow!("bg src: {e}"))?;
        let mut dst = Image::from_slice_u8(fw, fh, &mut self.bg_scaled, fr::PixelType::U8x4)
            .map_err(|e| anyhow!("bg dst: {e}"))?;
        self.resizer
            .resize(&src, &mut dst, &Self::resize_opts())
            .map_err(|e| anyhow!("bg resize: {e}"))?;
        self.bg_w = fw;
        self.bg_h = fh;
        Ok(())
    }
}

impl Default for Compositor {
    fn default() -> Self {
        Self::new()
    }
}

/// Two passes of a separable box kernel approximate a Gaussian with σ ≈
/// 1.5×radius. `strength` ∈ [0,1] maps to a radius from
/// `BLUR_MIN_RADIUS` to `BLUR_MAX_RADIUS` px.
fn composite_blur(frame: &mut [u8], width: u32, height: u32, mask: &[f32], strength: f32) {
    let w = width as usize;
    let h = height as usize;
    let s = strength.clamp(0.0, 1.0);
    let span = (BLUR_MAX_RADIUS - BLUR_MIN_RADIUS) as f32;
    let radius = (BLUR_MIN_RADIUS as f32 + s * span).round() as usize;
    // Two passes → quasi-Gaussian. Skip second pass for very small radii.
    let mut blurred = frame.to_vec();
    box_blur_rgba(&mut blurred, w, h, radius);
    if radius >= BLUR_TWO_PASS_THRESHOLD {
        box_blur_rgba(&mut blurred, w, h, radius);
    }

    // Plain alpha composite: out = fg*mask + blurred*(1-mask).
    blend(frame, &blurred, mask);
}

fn composite_image(frame: &mut [u8], bg_scaled: &[u8], mask: &[f32]) {
    debug_assert_eq!(bg_scaled.len(), frame.len());
    blend(frame, bg_scaled, mask);
}

/// Plain alpha composite of `frame` over `bg` using `mask`.
fn blend(frame: &mut [u8], bg: &[u8], mask: &[f32]) {
    for ((px, bg_px), &m) in frame
        .chunks_exact_mut(4)
        .zip(bg.chunks_exact(4))
        .zip(mask.iter())
    {
        let m = m.clamp(0.0, 1.0);
        let inv = 1.0 - m;
        for c in 0..3 {
            let fg = px[c] as f32;
            let bg_c = bg_px[c] as f32;
            px[c] = (fg * m + bg_c * inv) as u8;
        }
        px[3] = 255;
    }
}

/// Two-pass separable box blur on RGBA8. Radius is in pixels.
fn box_blur_rgba(buf: &mut [u8], w: usize, h: usize, r: usize) {
    if r == 0 {
        return;
    }
    let mut tmp = vec![0_u8; buf.len()];
    blur_horizontal(buf, &mut tmp, w, h, r);
    blur_vertical(&tmp, buf, w, h, r);
}

fn blur_horizontal(src: &[u8], dst: &mut [u8], w: usize, h: usize, r: usize) {
    let win = (2 * r + 1) as f32;
    for y in 0..h {
        let row = y * w * 4;
        for c in 0..3 {
            let mut sum = 0.0_f32;
            // Prime the window with edge replication.
            for k in 0..(2 * r + 1) {
                let xi = (k as isize - r as isize).clamp(0, w as isize - 1) as usize;
                sum += src[row + xi * 4 + c] as f32;
            }
            for x in 0..w {
                dst[row + x * 4 + c] = (sum / win) as u8;
                let x_out = (x as isize - r as isize).clamp(0, w as isize - 1) as usize;
                let x_in = (x as isize + r as isize + 1).clamp(0, w as isize - 1) as usize;
                sum += src[row + x_in * 4 + c] as f32 - src[row + x_out * 4 + c] as f32;
            }
        }
        // Alpha → 255.
        for x in 0..w {
            dst[row + x * 4 + 3] = 255;
        }
    }
}

fn blur_vertical(src: &[u8], dst: &mut [u8], w: usize, h: usize, r: usize) {
    let win = (2 * r + 1) as f32;
    for x in 0..w {
        for c in 0..3 {
            let mut sum = 0.0_f32;
            for k in 0..(2 * r + 1) {
                let yi = (k as isize - r as isize).clamp(0, h as isize - 1) as usize;
                sum += src[yi * w * 4 + x * 4 + c] as f32;
            }
            for y in 0..h {
                dst[y * w * 4 + x * 4 + c] = (sum / win) as u8;
                let y_out = (y as isize - r as isize).clamp(0, h as isize - 1) as usize;
                let y_in = (y as isize + r as isize + 1).clamp(0, h as isize - 1) as usize;
                sum += src[y_in * w * 4 + x * 4 + c] as f32 - src[y_out * w * 4 + x * 4 + c] as f32;
            }
        }
        for y in 0..h {
            dst[y * w * 4 + x * 4 + 3] = 255;
        }
    }
}
