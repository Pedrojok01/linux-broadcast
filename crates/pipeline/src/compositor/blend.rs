//! Per-pixel blend kernels: the fast in-place path and the asymmetric
//! remap used when auto-frame is on with an Image background.

use super::Framing;
use super::sample::{DECONTAM_MIN_ALPHA, refine_mask, sample_fg_bilinear, sample_rgb_bilinear};

/// Alpha composite of foreground over background using `mask`. In-place
/// fast path: foreground = current `frame`, output overwrites it. Used
/// when no auto-frame transform is supplied.
#[inline]
pub(super) fn blend(frame: &mut [u8], bg: &[u8], mask: &[f32]) {
    debug_assert_eq!(bg.len(), frame.len());
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

/// Asymmetric blend used when auto-frame is on. The bg plane is sampled
/// at unshifted output coords (so the virtual room stays put under
/// motion), while foreground RGBA + mask α are bilinearly sampled at
/// the remapped source position.
///
/// Two extra steps clean up the silhouette boundary:
/// 1. **Mask refinement** ([`refine_mask`]) shifts the soft α tail
///    down by `HALO_LO` and rescales — α below the floor clamps to 0
///    (kills the halo seam beyond the silhouette), the rest stays soft
///    so the edge anti-aliases naturally.
/// 2. **Alpha decontamination**: the observed camera pixel is a blend
///    `C = α·F + (1-α)·B_real` where `B_real` is the wall actually
///    behind the user. Without correcting for this the soft edges
///    drag the user's *real* wall colour into the virtual bg as a
///    halo (white ceiling → white halo). We estimate `B_real` from
///    the temporal plate and recover `F = (C - (1-α)·B_real) / α`,
///    then composite `α·F + (1-α)·bg_at_output`. Skipped at very low
///    α where the division is ill-conditioned (the contribution is
///    negligible there anyway).
///
/// Output pixels whose remapped source falls outside the source frame
/// are pure background — the silhouette ends at the source bounds and
/// the strip vacated on the trailing edge is filled by `bg`.
#[allow(clippy::too_many_arguments)]
pub(super) fn asymmetric_blend(
    frame: &mut [u8],
    bg: &[u8],
    fg: &[u8],
    plate: &[u8],
    mask: &[f32],
    w: usize,
    h: usize,
    f: Framing,
) {
    debug_assert_eq!(bg.len(), frame.len());
    debug_assert_eq!(fg.len(), frame.len());
    debug_assert_eq!(plate.len(), frame.len());
    let inv_zoom = 1.0 / f.zoom.max(1e-4);
    let wf = w as f32;
    let hf = h as f32;
    for y in 0..h {
        let src_yf = f.src_anchor_y + (y as f32 + 0.5 - f.dst_anchor_y) * inv_zoom - 0.5;
        let row = y * w;
        for x in 0..w {
            let dst_pi = (row + x) * 4;
            let src_xf = f.src_anchor_x + (x as f32 + 0.5 - f.dst_anchor_x) * inv_zoom - 0.5;

            // Vacated strip: remapped source outside the source frame.
            // Pure background.
            if src_xf <= -1.0 || src_xf >= wf || src_yf <= -1.0 || src_yf >= hf {
                frame[dst_pi] = bg[dst_pi];
                frame[dst_pi + 1] = bg[dst_pi + 1];
                frame[dst_pi + 2] = bg[dst_pi + 2];
                frame[dst_pi + 3] = 255;
                continue;
            }

            let (m_raw, fg_rgb) = sample_fg_bilinear(fg, mask, w, h, src_xf, src_yf);
            let m = refine_mask(m_raw);
            if m <= 0.0 {
                frame[dst_pi] = bg[dst_pi];
                frame[dst_pi + 1] = bg[dst_pi + 1];
                frame[dst_pi + 2] = bg[dst_pi + 2];
                frame[dst_pi + 3] = 255;
                continue;
            }

            // Decontaminate fg: subtract the (1-α) contribution of the
            // real wall (estimated from the plate) before blending over
            // the virtual bg. Skip at low α — the division blows up
            // and the contribution is tiny anyway.
            let plate_rgb = sample_rgb_bilinear(plate, w, h, src_xf, src_yf);
            let f_clean = if m >= DECONTAM_MIN_ALPHA {
                let inv_m = 1.0 / m;
                let one_minus_m = 1.0 - m;
                [
                    ((fg_rgb[0] - one_minus_m * plate_rgb[0]) * inv_m).clamp(0.0, 255.0),
                    ((fg_rgb[1] - one_minus_m * plate_rgb[1]) * inv_m).clamp(0.0, 255.0),
                    ((fg_rgb[2] - one_minus_m * plate_rgb[2]) * inv_m).clamp(0.0, 255.0),
                ]
            } else {
                fg_rgb
            };

            let inv = 1.0 - m;
            frame[dst_pi] = (f_clean[0] * m + bg[dst_pi] as f32 * inv) as u8;
            frame[dst_pi + 1] = (f_clean[1] * m + bg[dst_pi + 1] as f32 * inv) as u8;
            frame[dst_pi + 2] = (f_clean[2] * m + bg[dst_pi + 2] as f32 * inv) as u8;
            frame[dst_pi + 3] = 255;
        }
    }
}
