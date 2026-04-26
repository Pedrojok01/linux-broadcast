use anyhow::{anyhow, Result};
use fast_image_resize as fr;
use std::num::NonZeroU32;

use crate::{MODEL_H, MODEL_W};

/// Background mode the compositor should produce behind the foreground mask.
#[derive(Debug, Clone)]
pub enum Background {
    /// Gaussian-blur the original frame and use that as the background.
    Blur,
    /// Composite over a static RGBA image, scaled to cover the frame.
    Image { rgba: Vec<u8>, width: u32, height: u32 },
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
            resizer: fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear)),
            upsampled_mask: Vec::new(),
            bg_scaled: Vec::new(),
            bg_w: 0,
            bg_h: 0,
        }
    }

    /// Composite an RGBA frame in place using the given mask and background.
    ///
    /// `mask` is `MODEL_W * MODEL_H` foreground probabilities in `[0,1]`.
    /// `frame` is mutated to hold the composited output.
    pub fn composite(
        &mut self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        mask: &[f32],
        background: &Background,
    ) -> Result<()> {
        if frame.len() != (width as usize) * (height as usize) * 4 {
            return Err(anyhow!("frame buffer size mismatch"));
        }
        if mask.len() != MODEL_W * MODEL_H {
            return Err(anyhow!("mask size {} != {}", mask.len(), MODEL_W * MODEL_H));
        }

        self.upsample_mask(mask, width, height)?;

        match background {
            Background::Blur => composite_blur(frame, width, height, &self.upsampled_mask),
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

    fn upsample_mask(&mut self, mask: &[f32], width: u32, height: u32) -> Result<()> {
        let target = (width as usize) * (height as usize);
        if self.upsampled_mask.len() != target {
            self.upsampled_mask.resize(target, 0.0);
        }
        // Bilinear sampling from the 256x144 mask into the frame-sized buffer.
        let src_w = MODEL_W as f32;
        let src_h = MODEL_H as f32;
        let dst_w = width as f32;
        let dst_h = height as f32;
        for y in 0..height {
            let sy = (y as f32 + 0.5) * src_h / dst_h - 0.5;
            let y0 = sy.floor().clamp(0.0, src_h - 1.0) as usize;
            let y1 = (y0 + 1).min(MODEL_H - 1);
            let fy = (sy - y0 as f32).clamp(0.0, 1.0);
            let row_dst = (y as usize) * (width as usize);
            let row0 = y0 * MODEL_W;
            let row1 = y1 * MODEL_W;
            for x in 0..width {
                let sx = (x as f32 + 0.5) * src_w / dst_w - 0.5;
                let x0 = sx.floor().clamp(0.0, src_w - 1.0) as usize;
                let x1 = (x0 + 1).min(MODEL_W - 1);
                let fx = (sx - x0 as f32).clamp(0.0, 1.0);
                let a = mask[row0 + x0];
                let b = mask[row0 + x1];
                let c = mask[row1 + x0];
                let d = mask[row1 + x1];
                let top = a * (1.0 - fx) + b * fx;
                let bot = c * (1.0 - fx) + d * fx;
                self.upsampled_mask[row_dst + x as usize] = top * (1.0 - fy) + bot * fy;
            }
        }
        Ok(())
    }

    fn ensure_bg_scaled(
        &mut self,
        bg: &[u8],
        bw: u32,
        bh: u32,
        fw: u32,
        fh: u32,
    ) -> Result<()> {
        if self.bg_w == fw && self.bg_h == fh && !self.bg_scaled.is_empty() {
            return Ok(());
        }
        self.bg_scaled.resize((fw as usize) * (fh as usize) * 4, 0);
        let src = fr::Image::from_slice_u8(
            NonZeroU32::new(bw).ok_or_else(|| anyhow!("bg width=0"))?,
            NonZeroU32::new(bh).ok_or_else(|| anyhow!("bg height=0"))?,
            unsafe { std::slice::from_raw_parts_mut(bg.as_ptr() as *mut u8, bg.len()) },
            fr::PixelType::U8x4,
        )
        .map_err(|e| anyhow!("bg src: {e}"))?;
        let mut dst = fr::Image::from_slice_u8(
            NonZeroU32::new(fw).unwrap(),
            NonZeroU32::new(fh).unwrap(),
            &mut self.bg_scaled,
            fr::PixelType::U8x4,
        )
        .map_err(|e| anyhow!("bg dst: {e}"))?;
        self.resizer
            .resize(&src.view(), &mut dst.view_mut())
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

/// Box-blur approximation good enough for a "Broadcast-like" backdrop blur.
/// Two passes of a separable 1D kernel, radius 12. Done on RGBA in place into
/// a scratch buffer that is then used as the background plane.
fn composite_blur(frame: &mut [u8], width: u32, height: u32, mask: &[f32]) {
    let w = width as usize;
    let h = height as usize;
    // Scratch: blurred copy of the frame.
    let mut blurred = frame.to_vec();
    box_blur_rgba(&mut blurred, w, h, 12);

    // Composite: out = fg*mask + blurred*(1-mask).
    for i in 0..(w * h) {
        let m = mask[i].clamp(0.0, 1.0);
        let inv = 1.0 - m;
        let o = i * 4;
        for c in 0..3 {
            let fg = frame[o + c] as f32;
            let bg = blurred[o + c] as f32;
            frame[o + c] = (fg * m + bg * inv) as u8;
        }
        frame[o + 3] = 255;
    }
}

fn composite_image(frame: &mut [u8], bg_scaled: &[u8], mask: &[f32]) {
    let n = frame.len() / 4;
    debug_assert_eq!(bg_scaled.len(), frame.len());
    for i in 0..n {
        let m = mask[i].clamp(0.0, 1.0);
        let inv = 1.0 - m;
        let o = i * 4;
        for c in 0..3 {
            let fg = frame[o + c] as f32;
            let bg = bg_scaled[o + c] as f32;
            frame[o + c] = (fg * m + bg * inv) as u8;
        }
        frame[o + 3] = 255;
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
                sum += src[y_in * w * 4 + x * 4 + c] as f32
                    - src[y_out * w * 4 + x * 4 + c] as f32;
            }
        }
        for y in 0..h {
            dst[y * w * 4 + x * 4 + 3] = 255;
        }
    }
}
