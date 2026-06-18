//! Bilinear samplers used by the `Background::Image` asymmetric framing
//! blend. The Blur/None auto-frame crop now goes through the SIMD resizer
//! (`Compositor::crop_and_rescale`); only the per-pixel decontaminating
//! Image remap still samples by hand.

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

#[cfg(test)]
mod tests {
    use super::refine_mask;

    #[test]
    fn clamps_low_alpha_tail_to_zero() {
        assert_eq!(refine_mask(0.0), 0.0);
        assert_eq!(refine_mask(0.05), 0.0);
        // The HALO_LO boundary itself maps to 0 (inclusive).
        assert_eq!(refine_mask(0.10), 0.0);
    }

    #[test]
    fn preserves_and_saturates_high_alpha() {
        assert_eq!(refine_mask(1.0), 1.0);
        // Just above the floor stays soft (not snapped to 0 or 1).
        let r = refine_mask(0.11);
        assert!(r > 0.0 && r < 0.05, "expected small positive α, got {r}");
    }

    #[test]
    fn rescales_midrange_linearly() {
        // (m − 0.10) / (1 − 0.10): 0.55 → 0.5, 0.10 → 0.0, 1.0 → 1.0.
        assert!(
            (refine_mask(0.55) - 0.5).abs() < 1e-6,
            "{}",
            refine_mask(0.55)
        );
    }

    #[test]
    fn is_monotonic_non_decreasing() {
        let mut prev = -1.0;
        for i in 0..=100 {
            let m = i as f32 / 100.0;
            let r = refine_mask(m);
            assert!(
                r >= prev,
                "refine_mask not monotonic at m={m}: {r} < {prev}"
            );
            prev = r;
        }
    }
}
