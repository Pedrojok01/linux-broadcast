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

/// Affine remap of the foreground sample point. The compositor reads
/// foreground RGBA + mask at
/// `src = src_anchor + (out - dst_anchor) / zoom`,
/// bilinearly interpolated when the result is fractional. The
/// background plane is sampled at unshifted output coordinates, so a
/// non-trivial framing makes the silhouette slide and/or grow over a
/// stationary background.
///
/// Used by auto-frame to recenter horizontally on the silhouette
/// centroid and apply a static [`crate::framing::FG_ZOOM`] anchored at
/// the head-top.
#[derive(Debug, Clone, Copy)]
pub struct Framing {
    /// Source pixel coordinates of the anchor point (where the
    /// silhouette is being "held" during the remap).
    pub src_anchor_x: f32,
    pub src_anchor_y: f32,
    /// Output pixel coordinates the anchor lands at.
    pub dst_anchor_x: f32,
    pub dst_anchor_y: f32,
    /// Foreground zoom factor, ≥ 1.0.
    pub zoom: f32,
}

impl Framing {
    /// True when the framing has no visible effect (zoom 1.0 and
    /// anchors coincide), so the compositor can take its in-place fast
    /// path.
    fn is_identity(&self) -> bool {
        (self.zoom - 1.0).abs() < 1e-4
            && (self.src_anchor_x - self.dst_anchor_x).abs() < 1e-3
            && (self.src_anchor_y - self.dst_anchor_y).abs() < 1e-3
    }
}

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
    /// Fingerprint of the source image last scaled into `bg_scaled`. The
    /// cache is keyed on this in addition to the frame size so picking a
    /// different library image (same frame size, different pixels) actually
    /// invalidates the rescaled buffer.
    bg_fingerprint: Option<u64>,
    /// Unframed copy of the input frame, kept across calls so the
    /// auto-frame stage can sample foreground RGBA from a remapped
    /// source position while writing to the in-place output. Allocated
    /// lazily; only used when a non-identity `Framing` is supplied.
    fg_scratch: Vec<u8>,
}

impl Compositor {
    pub fn new() -> Self {
        Self {
            resizer: fr::Resizer::new(),
            upsampled_mask: Vec::new(),
            bg_scaled: Vec::new(),
            bg_w: 0,
            bg_h: 0,
            bg_fingerprint: None,
            fg_scratch: Vec::new(),
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
    ///
    /// `framing` reparameterizes the per-output-pixel foreground sample
    /// point (translation + uniform zoom). The background plane is
    /// always sampled at unshifted output coordinates, so the blurred
    /// wall (or replacement image) stays put while the silhouette
    /// slides and/or grows over it. Any output pixels whose remapped
    /// source falls outside the source frame are pure background
    /// (`mask = 0`). Pass `None` (or an identity framing) to take the
    /// fast in-place path.
    pub fn composite(
        &mut self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        mask: &Mask,
        background: &Background,
        framing: Option<Framing>,
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

        // Short-circuit: passthrough mode skips the upsample + blend
        // entirely. Auto-frame is disabled in this mode (no background
        // plane to fill the strip vacated by the remapped silhouette),
        // so a non-trivial framing here is a programming error in the
        // feeder.
        if matches!(background, Background::None) {
            debug_assert!(
                framing.map_or(true, |f| f.is_identity()),
                "Background::None must not receive a non-identity framing",
            );
            return Ok(());
        }

        self.prepare_mask(mask, width, height)?;

        // Background prep first — these methods take `&mut self`, so we
        // can't be holding the `fg_scratch` borrow yet.
        if let Background::Image {
            rgba: bg_rgba,
            width: bw,
            height: bh,
        } = background
        {
            self.ensure_bg_scaled(bg_rgba, *bw, *bh, width, height)?;
        }

        // Identity framing collapses to the in-place fast path.
        let framing = framing.filter(|f| !f.is_identity());

        // When remapping, foreground reads come from positions we may
        // already have written in place — keep a clean copy.
        if framing.is_some() {
            let n = frame.len();
            if self.fg_scratch.len() != n {
                self.fg_scratch.resize(n, 0);
            }
            self.fg_scratch.copy_from_slice(frame);
        }
        let fg_src: Option<&[u8]> = framing.map(|_| self.fg_scratch.as_slice());

        match background {
            Background::None => unreachable!(),
            Background::Blur { strength } => {
                composite_blur(
                    frame,
                    width,
                    height,
                    &self.upsampled_mask,
                    *strength,
                    framing,
                    fg_src,
                );
            }
            Background::Image { .. } => {
                composite_image(
                    frame,
                    width,
                    height,
                    &self.bg_scaled,
                    &self.upsampled_mask,
                    framing,
                    fg_src,
                );
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
        let fp = bg_fingerprint(bg, bw, bh);
        if self.bg_w == fw
            && self.bg_h == fh
            && self.bg_fingerprint == Some(fp)
            && !self.bg_scaled.is_empty()
        {
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
        self.bg_fingerprint = Some(fp);
        Ok(())
    }
}

impl Default for Compositor {
    fn default() -> Self {
        Self::new()
    }
}

/// Cheap content fingerprint for an RGBA buffer. We hash the source
/// dimensions plus a sparse byte sample so two same-size library images
/// with different pixels disambiguate without re-hashing the whole buffer.
/// 1280×720 RGBA is ~3.7 MB — touching every byte each call would dwarf
/// the actual rescale.
fn bg_fingerprint(bg: &[u8], bw: u32, bh: u32) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bw.hash(&mut h);
    bh.hash(&mut h);
    bg.len().hash(&mut h);
    let n = bg.len();
    if n >= 64 {
        bg[..32].hash(&mut h);
        bg[n - 32..].hash(&mut h);
        bg[n / 2..(n / 2 + 32).min(n)].hash(&mut h);
    } else {
        bg.hash(&mut h);
    }
    h.finish()
}

/// Two passes of a separable box kernel approximate a Gaussian with σ ≈
/// 1.5×radius. `strength` ∈ [0,1] maps to a radius from
/// `BLUR_MIN_RADIUS` to `BLUR_MAX_RADIUS` px.
///
/// `fg_src` must be `Some` when `framing.is_some()` (a clean copy of
/// the unframed input), `None` otherwise. The blur is computed from
/// `frame` in either case so the background plane keeps its original
/// content — that's what makes the silhouette appear to slide and grow
/// *over* a stationary background.
fn composite_blur(
    frame: &mut [u8],
    width: u32,
    height: u32,
    mask: &[f32],
    strength: f32,
    framing: Option<Framing>,
    fg_src: Option<&[u8]>,
) {
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

    blend(frame, &blurred, mask, w, h, framing, fg_src);
}

fn composite_image(
    frame: &mut [u8],
    width: u32,
    height: u32,
    bg_scaled: &[u8],
    mask: &[f32],
    framing: Option<Framing>,
    fg_src: Option<&[u8]>,
) {
    debug_assert_eq!(bg_scaled.len(), frame.len());
    let w = width as usize;
    let h = height as usize;
    blend(frame, bg_scaled, mask, w, h, framing, fg_src);
}

/// Alpha composite of foreground over background using `mask`.
///
/// With `framing == None`, takes the in-place fast path that reads the
/// foreground from `frame` itself. With `framing == Some`, foreground
/// and mask are sampled at the remapped `(src_x, src_y)` (bilinear),
/// reading from `fg_src` — a separate buffer holding the unframed
/// input — so source and destination don't alias. Out-of-source
/// samples produce pure background (mask = 0).
fn blend(
    frame: &mut [u8],
    bg: &[u8],
    mask: &[f32],
    w: usize,
    h: usize,
    framing: Option<Framing>,
    fg_src: Option<&[u8]>,
) {
    let Some(framing) = framing else {
        // In-place fast path: foreground = current `frame`, output overwrites it.
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
        return;
    };

    let fg = fg_src.expect("fg_src required when framing is Some");
    let inv_zoom = 1.0 / framing.zoom.max(1e-4);
    let wf = w as f32;
    let hf = h as f32;
    for y in 0..h {
        let src_yf = framing.src_anchor_y + (y as f32 + 0.5 - framing.dst_anchor_y) * inv_zoom - 0.5;
        let row = y * w;
        for x in 0..w {
            let dst_pi = (row + x) * 4;
            let src_xf =
                framing.src_anchor_x + (x as f32 + 0.5 - framing.dst_anchor_x) * inv_zoom - 0.5;

            // Pure-bg case: remapped source falls outside the source
            // frame entirely (or far enough that no neighbor is in
            // range). The bilinear taps below clamp to edge, but we
            // want the silhouette to "end" at the source bounds rather
            // than smearing edge pixels across the vacated strip — so
            // bypass when fully outside.
            if src_xf <= -1.0 || src_xf >= wf || src_yf <= -1.0 || src_yf >= hf {
                frame[dst_pi] = bg[dst_pi];
                frame[dst_pi + 1] = bg[dst_pi + 1];
                frame[dst_pi + 2] = bg[dst_pi + 2];
                frame[dst_pi + 3] = 255;
                continue;
            }

            let (m, fg_rgb) = sample_fg_bilinear(fg, mask, w, h, src_xf, src_yf);
            let inv = 1.0 - m;
            frame[dst_pi] = (fg_rgb[0] * m + bg[dst_pi] as f32 * inv) as u8;
            frame[dst_pi + 1] = (fg_rgb[1] * m + bg[dst_pi + 1] as f32 * inv) as u8;
            frame[dst_pi + 2] = (fg_rgb[2] * m + bg[dst_pi + 2] as f32 * inv) as u8;
            frame[dst_pi + 3] = 255;
        }
    }
}

/// Bilinear sample of foreground RGB and mask α at fractional source
/// coords. Returns `(mask α in [0,1], rgb as f32)`. Edge taps are
/// clamped to the source bounds.
#[inline]
fn sample_fg_bilinear(
    fg: &[u8],
    mask: &[f32],
    w: usize,
    h: usize,
    sx: f32,
    sy: f32,
) -> (f32, [f32; 3]) {
    let x0 = sx.floor();
    let y0 = sy.floor();
    let fx = sx - x0;
    let fy = sy - y0;
    let xi0 = (x0 as isize).clamp(0, w as isize - 1) as usize;
    let xi1 = ((x0 as isize) + 1).clamp(0, w as isize - 1) as usize;
    let yi0 = (y0 as isize).clamp(0, h as isize - 1) as usize;
    let yi1 = ((y0 as isize) + 1).clamp(0, h as isize - 1) as usize;

    let i00 = yi0 * w + xi0;
    let i01 = yi0 * w + xi1;
    let i10 = yi1 * w + xi0;
    let i11 = yi1 * w + xi1;

    let m = {
        let m00 = mask[i00];
        let m01 = mask[i01];
        let m10 = mask[i10];
        let m11 = mask[i11];
        let top = m00 * (1.0 - fx) + m01 * fx;
        let bot = m10 * (1.0 - fx) + m11 * fx;
        (top * (1.0 - fy) + bot * fy).clamp(0.0, 1.0)
    };

    let mut rgb = [0.0f32; 3];
    for c in 0..3 {
        let p00 = fg[i00 * 4 + c] as f32;
        let p01 = fg[i01 * 4 + c] as f32;
        let p10 = fg[i10 * 4 + c] as f32;
        let p11 = fg[i11 * 4 + c] as f32;
        let top = p00 * (1.0 - fx) + p01 * fx;
        let bot = p10 * (1.0 - fx) + p11 * fx;
        rgb[c] = top * (1.0 - fy) + bot * fy;
    }
    (m, rgb)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mask;
    use proptest::prelude::*;

    fn solid_frame(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
        let mut buf = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            buf.extend_from_slice(&rgba);
        }
        buf
    }

    fn mask_const(w: u32, h: u32, v: f32) -> Mask {
        Mask {
            data: vec![v; (w as usize) * (h as usize)],
            width: w,
            height: h,
        }
    }

    fn red_image_bg(w: u32, h: u32) -> Background {
        Background::Image {
            rgba: solid_frame(w, h, [255, 0, 0, 255]),
            width: w,
            height: h,
        }
    }

    #[test]
    fn none_is_byte_identical_passthrough() {
        // Even with a non-trivial mask, Background::None must be a true
        // bytewise short-circuit.
        let (w, h) = (32, 32);
        let mut frame = solid_frame(w, h, [10, 20, 30, 255]);
        // Add a recognisable pattern so any modification is visible.
        for (i, px) in frame.chunks_exact_mut(4).enumerate() {
            px[0] = (i % 251) as u8;
            px[1] = ((i * 3) % 251) as u8;
            px[2] = ((i * 7) % 251) as u8;
        }
        let original = frame.clone();
        let mask = Mask {
            data: (0..(w * h))
                .map(|i| (i as f32) / ((w * h) as f32))
                .collect(),
            width: w,
            height: h,
        };
        let mut c = Compositor::new();
        c.composite(&mut frame, w, h, &mask, &Background::None, None)
            .unwrap();
        assert_eq!(
            frame, original,
            "Background::None must not modify the frame"
        );
    }

    /// Identity framing: `src_anchor == dst_anchor`, zoom = 1.0. Should
    /// be detected as identity and trigger the fast path.
    fn identity_framing(w: u32, h: u32) -> Framing {
        let cx = w as f32 * 0.5;
        let cy = h as f32 * 0.5;
        Framing {
            src_anchor_x: cx,
            src_anchor_y: cy,
            dst_anchor_x: cx,
            dst_anchor_y: cy,
            zoom: 1.0,
        }
    }

    #[test]
    fn mask_full_foreground_preserves_frame() {
        // mask = 1.0 everywhere → output equals foreground (input frame).
        let (w, h) = (16, 16);
        let mut frame = solid_frame(w, h, [50, 100, 150, 255]);
        let original = frame.clone();
        let mask = mask_const(w, h, 1.0);
        let mut c = Compositor::new();
        c.composite(&mut frame, w, h, &mask, &red_image_bg(w, h), None)
            .unwrap();
        assert_eq!(frame, original);
    }

    #[test]
    fn mask_full_background_replaces_frame() {
        // mask = 0.0 everywhere → output equals background.
        let (w, h) = (16, 16);
        let mut frame = solid_frame(w, h, [50, 100, 150, 255]);
        let mask = mask_const(w, h, 0.0);
        let mut c = Compositor::new();
        c.composite(&mut frame, w, h, &mask, &red_image_bg(w, h), None)
            .unwrap();
        for px in frame.chunks_exact(4) {
            assert_eq!(px, &[255, 0, 0, 255]);
        }
    }

    #[test]
    fn mask_half_blends_midpoint() {
        // mask = 0.5, grey input + red bg → per-channel midpoint within ±2.
        let (w, h) = (16, 16);
        let mut frame = solid_frame(w, h, [100, 100, 100, 255]);
        let mask = mask_const(w, h, 0.5);
        let mut c = Compositor::new();
        c.composite(&mut frame, w, h, &mask, &red_image_bg(w, h), None)
            .unwrap();
        // out_R ≈ 100*0.5 + 255*0.5 = 177; out_G/B ≈ 100*0.5 + 0*0.5 = 50.
        for px in frame.chunks_exact(4) {
            assert!((px[0] as i32 - 177).abs() <= 2, "R={}", px[0]);
            assert!((px[1] as i32 - 50).abs() <= 2, "G={}", px[1]);
            assert!((px[2] as i32 - 50).abs() <= 2, "B={}", px[2]);
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn mismatched_mask_size_upsamples() {
        // 64×64 mask + 256×256 frame: must produce a 256×256 output and
        // not panic. Use a half-mask so the upsample exercises the
        // bilinear path.
        let (fw, fh) = (256, 256);
        let mut frame = solid_frame(fw, fh, [40, 40, 40, 255]);
        let mask = mask_const(64, 64, 0.5);
        let mut c = Compositor::new();
        c.composite(&mut frame, fw, fh, &mask, &red_image_bg(fw, fh), None)
            .unwrap();
        assert_eq!(frame.len(), (fw * fh * 4) as usize);
        // First pixel ≈ midpoint blend of (40,40,40) over (255,0,0) at α=0.5.
        let px = &frame[..4];
        assert!((px[0] as i32 - 147).abs() <= 3);
        assert!((px[1] as i32 - 20).abs() <= 3);
        assert!((px[2] as i32 - 20).abs() <= 3);
    }

    #[test]
    fn identity_framing_matches_no_framing() {
        // Identity framing must collapse to the in-place fast path and
        // produce byte-identical output to the no-framing call.
        let (w, h) = (32, 32);
        let mut a = solid_frame(w, h, [80, 120, 200, 255]);
        let mask = mask_const(w, h, 0.5);
        let mut c1 = Compositor::new();
        c1.composite(&mut a, w, h, &mask, &red_image_bg(w, h), None)
            .unwrap();
        let mut b = solid_frame(w, h, [80, 120, 200, 255]);
        let mut c2 = Compositor::new();
        c2.composite(
            &mut b,
            w,
            h,
            &mask,
            &red_image_bg(w, h),
            Some(identity_framing(w, h)),
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn framing_translates_silhouette() {
        // Build a mask with a vertical fg "stripe" 4 px wide on the left
        // (x ∈ [4, 8) of a 32-wide frame). With src_anchor at x=6 and
        // dst_anchor at x=14 (shift = +8) the silhouette should land at
        // x ∈ [12, 16). Outside the silhouette is pure red bg; inside
        // is pure foreground (mask=1, fg=blue).
        let (w, h) = (32, 8);
        let mut frame = solid_frame(w, h, [0, 0, 255, 255]);
        let mut mask_data = vec![0.0f32; (w * h) as usize];
        for y in 0..(h as usize) {
            for x in 4..8usize {
                mask_data[y * w as usize + x] = 1.0;
            }
        }
        let mask = Mask {
            data: mask_data,
            width: w,
            height: h,
        };
        let framing = Framing {
            src_anchor_x: 6.0,
            src_anchor_y: h as f32 * 0.5,
            dst_anchor_x: 14.0,
            dst_anchor_y: h as f32 * 0.5,
            zoom: 1.0,
        };
        let mut c = Compositor::new();
        c.composite(&mut frame, w, h, &mask, &red_image_bg(w, h), Some(framing))
            .unwrap();
        for y in 0..(h as usize) {
            for x in 0..(w as usize) {
                let pi = (y * w as usize + x) * 4;
                let in_shifted_silhouette = (12..16).contains(&x);
                if in_shifted_silhouette {
                    assert_eq!(
                        &frame[pi..pi + 3],
                        &[0, 0, 255],
                        "expected blue fg at x={x}",
                    );
                } else {
                    assert_eq!(&frame[pi..pi + 3], &[255, 0, 0], "expected red bg at x={x}",);
                }
            }
        }
    }

    #[test]
    fn framing_zoom_enlarges_silhouette() {
        // Centered fg square at x ∈ [12, 20), y ∈ [12, 20) of a 32×32
        // frame (8×8 → mass-weighted center at (16, 16)). Zoom 2× around
        // the frame center should roughly double the silhouette extent
        // to x,y ∈ [8, 24) — verify the corners are foreground (within
        // bilinear slop) and the edges past x=24 are background.
        let (w, h) = (32, 32);
        let mut frame = solid_frame(w, h, [0, 0, 255, 255]);
        let mut mask_data = vec![0.0f32; (w * h) as usize];
        for y in 12..20usize {
            for x in 12..20usize {
                mask_data[y * w as usize + x] = 1.0;
            }
        }
        let mask = Mask {
            data: mask_data,
            width: w,
            height: h,
        };
        let framing = Framing {
            src_anchor_x: 16.0,
            src_anchor_y: 16.0,
            dst_anchor_x: 16.0,
            dst_anchor_y: 16.0,
            zoom: 2.0,
        };
        let mut c = Compositor::new();
        c.composite(&mut frame, w, h, &mask, &red_image_bg(w, h), Some(framing))
            .unwrap();
        // Center pixel: solidly inside the zoomed silhouette → blue.
        let pi = (16 * w as usize + 16) * 4;
        assert_eq!(&frame[pi..pi + 3], &[0, 0, 255]);
        // Pixel at (10, 16): src_x = 16 + (10.5 - 16) * 0.5 - 0.5 = 12.75
        // → inside source silhouette (mask=1) → blue.
        let pi = (16 * w as usize + 10) * 4;
        assert_eq!(&frame[pi..pi + 3], &[0, 0, 255]);
        // Pixel at (28, 16): src_x = 16 + (28.5 - 16) * 0.5 - 0.5 = 21.75
        // → outside source silhouette (mask=0) → red.
        let pi = (16 * w as usize + 28) * 4;
        assert_eq!(&frame[pi..pi + 3], &[255, 0, 0]);
    }

    proptest! {
        // For a fixed input frame and bg, increasing mask α must move
        // each output channel monotonically from background toward
        // foreground. Catches sign flips / mis-inverted alpha in `blend`.
        #[test]
        fn mask_monotonic_in_alpha(
            fg in 0u8..=255,
            bg in 0u8..=255,
            // Three increasing α values in [0,1].
            a0 in 0.0f32..=0.33,
            a1 in 0.34f32..=0.66,
            a2 in 0.67f32..=1.0,
        ) {
            let (w, h) = (8u32, 8u32);
            let frame_init = solid_frame(w, h, [fg, fg, fg, 255]);
            let bg_image = Background::Image {
                rgba: solid_frame(w, h, [bg, bg, bg, 255]),
                width: w,
                height: h,
            };
            let mut out = [frame_init.clone(), frame_init.clone(), frame_init.clone()];
            for (frame, &alpha) in out.iter_mut().zip(&[a0, a1, a2]) {
                let mut c = Compositor::new();
                let mask = mask_const(w, h, alpha);
                c.composite(frame, w, h, &mask, &bg_image, None).unwrap();
            }
            // Pick channel 0 of pixel 0 from each output. Monotone toward
            // fg as α grows. Allow ±1 for u8 rounding.
            let v: Vec<i32> = out.iter().map(|f| f[0] as i32).collect();
            let toward_fg = (fg as i32) - (bg as i32);
            // sign of (v[i+1]-v[i]) should match sign of toward_fg (or be 0).
            for w in v.windows(2) {
                let delta = w[1] - w[0];
                if toward_fg > 0 {
                    prop_assert!(delta >= -1, "expected non-decreasing, got {:?}", v);
                } else if toward_fg < 0 {
                    prop_assert!(delta <= 1, "expected non-increasing, got {:?}", v);
                }
            }
        }
    }
}
