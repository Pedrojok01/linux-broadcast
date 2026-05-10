//! Bilinear samplers + the auto-frame crop-and-rescale used by both
//! `Background::Blur` (post-composite) and `Background::None` (raw frame)
//! when framing is on.

use super::Framing;

/// Minimum α at which alpha decontamination is applied. Below this the
/// `1/α` division amplifies bilinear noise badly without changing the
/// visible output (the foreground contribution is already vanishingly
/// small). Picked so the noise stays under one channel quantization step.
pub(super) const DECONTAM_MIN_ALPHA: f32 = 0.05;

/// Refine the soft mask α before the asymmetric blend.
///
/// Linear shift+rescale: `α ≤ HALO_LO` → 0 (kills the halo tail beyond
/// the silhouette), otherwise rescale `(α − HALO_LO) / (1 − HALO_LO)`.
/// Crucially does NOT saturate near the silhouette interior — the soft
/// α gradient at the edge is preserved so the decontamination step has
/// real data to work with. Saturating early (as a smoothstep would)
/// locks in the camera-pixel/wall-blend as 100% foreground and
/// produces a visible white halo around the user.
#[inline]
pub(super) fn refine_mask(m: f32) -> f32 {
    const HALO_LO: f32 = 0.10;
    if m <= HALO_LO {
        0.0
    } else {
        ((m - HALO_LO) / (1.0 - HALO_LO)).min(1.0)
    }
}

/// Bilinear RGB sample (no mask) used for the plate `B_estimate` lookup
/// in `asymmetric_blend`. Edge taps are clamped to source bounds.
#[inline]
pub(super) fn sample_rgb_bilinear(buf: &[u8], w: usize, h: usize, sx: f32, sy: f32) -> [f32; 3] {
    let (xi0, xi1, fx) = clamp_idx(sx, w);
    let (yi0, yi1, fy) = clamp_idx(sy, h);

    let i00 = (yi0 * w + xi0) * 4;
    let i01 = (yi0 * w + xi1) * 4;
    let i10 = (yi1 * w + xi0) * 4;
    let i11 = (yi1 * w + xi1) * 4;

    let mut rgb = [0.0f32; 3];
    for c in 0..3 {
        let p00 = buf[i00 + c] as f32;
        let p01 = buf[i01 + c] as f32;
        let p10 = buf[i10 + c] as f32;
        let p11 = buf[i11 + c] as f32;
        let top = p00 * (1.0 - fx) + p01 * fx;
        let bot = p10 * (1.0 - fx) + p11 * fx;
        rgb[c] = top * (1.0 - fy) + bot * fy;
    }
    rgb
}

/// Bilinear sample of foreground RGB and mask α at fractional source
/// coords. Returns `(α in [0,1], rgb as f32)`. Edge taps are clamped to
/// the source bounds.
#[inline]
pub(super) fn sample_fg_bilinear(
    fg: &[u8],
    mask: &[f32],
    w: usize,
    h: usize,
    sx: f32,
    sy: f32,
) -> (f32, [f32; 3]) {
    let (xi0, xi1, fx) = clamp_idx(sx, w);
    let (yi0, yi1, fy) = clamp_idx(sy, h);

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

/// Crop a window of `scratch` and bilinearly rescale it into `frame`.
/// Both buffers are RGBA8 of dimensions `w × h`. Used by the
/// `Background::Blur` and `Background::None` framing paths to apply the
/// auto-frame transform after (Blur) or instead of (None) compositing.
/// Sampling indices are edge-clamped, but the caller's anchor math
/// (`lazy.rs` `min_dst_y` clamp + horizontal `min_src_x..max_src_x`
/// clamp) keeps the window strictly inside the source in normal
/// operation.
pub(super) fn crop_and_rescale_in_place(
    frame: &mut [u8],
    scratch: &[u8],
    w: usize,
    h: usize,
    f: Framing,
) {
    debug_assert_eq!(scratch.len(), frame.len());
    let inv_zoom = 1.0 / f.zoom.max(1e-4);
    for y in 0..h {
        let src_yf = f.src_anchor_y + (y as f32 + 0.5 - f.dst_anchor_y) * inv_zoom - 0.5;
        for x in 0..w {
            let src_xf = f.src_anchor_x + (x as f32 + 0.5 - f.dst_anchor_x) * inv_zoom - 0.5;
            let rgb = sample_rgb_bilinear(scratch, w, h, src_xf, src_yf);
            let pi = (y * w + x) * 4;
            frame[pi] = rgb[0] as u8;
            frame[pi + 1] = rgb[1] as u8;
            frame[pi + 2] = rgb[2] as u8;
            frame[pi + 3] = 255;
        }
    }
}

/// Resolve a fractional 1D coordinate into the (lo, hi, frac) triple a
/// bilinear sampler needs. Edge taps are clamped to `[0, max - 1]`.
#[inline]
fn clamp_idx(s: f32, max: usize) -> (usize, usize, f32) {
    let s0 = s.floor();
    let frac = (s - s0).clamp(0.0, 1.0);
    let max_i = max as isize - 1;
    let lo = (s0 as isize).clamp(0, max_i) as usize;
    let hi = ((s0 as isize) + 1).clamp(0, max_i) as usize;
    (lo, hi, frac)
}
