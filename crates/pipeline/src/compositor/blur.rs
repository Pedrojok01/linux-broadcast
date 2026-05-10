//! Two-pass separable box blur on RGBA8 frames.
//!
//! The vertical pass operates on the transposed image so that the inner
//! loop strides one pixel at a time instead of `w` pixels — keeps the
//! L1 cache hot at 720p / 1080p where a column stride blows the working
//! set.

/// Two-pass separable box blur on RGBA8 in `buf`, using `tmp` as the
/// horizontal-pass intermediate. Alpha is forced to 255.
///
/// Frames smaller than the kernel still produce a result — the kernel
/// edge-clamps and the output ends up close to a flat colour — but
/// production frames are always 720p+, well above the largest radius.
pub(super) fn box_blur_rgba(buf: &mut [u8], tmp: &mut [u8], w: usize, h: usize, r: usize) {
    blur_horizontal(buf, tmp, w, h, r);
    blur_vertical(tmp, buf, w, h, r);
}

fn blur_horizontal(src: &[u8], dst: &mut [u8], w: usize, h: usize, r: usize) {
    let win = (2 * r + 1) as f32;
    let inv_win = 1.0 / win;
    let max_x = w as isize - 1;
    for y in 0..h {
        let row = y * w * 4;
        for c in 0..3 {
            let mut sum = 0.0_f32;
            for k in 0..(2 * r + 1) {
                let xi = (k as isize - r as isize).clamp(0, max_x) as usize;
                sum += src[row + xi * 4 + c] as f32;
            }
            for x in 0..w {
                dst[row + x * 4 + c] = round_u8(sum * inv_win);
                let x_out = (x as isize - r as isize).clamp(0, max_x) as usize;
                let x_in = (x as isize + r as isize + 1).clamp(0, max_x) as usize;
                sum += src[row + x_in * 4 + c] as f32 - src[row + x_out * 4 + c] as f32;
            }
        }
        for x in 0..w {
            dst[row + x * 4 + 3] = 255;
        }
    }
}

fn blur_vertical(src: &[u8], dst: &mut [u8], w: usize, h: usize, r: usize) {
    let win = (2 * r + 1) as f32;
    let inv_win = 1.0 / win;
    let max_y = h as isize - 1;
    let stride = w * 4;
    for x in 0..w {
        let col = x * 4;
        for c in 0..3 {
            let mut sum = 0.0_f32;
            for k in 0..(2 * r + 1) {
                let yi = (k as isize - r as isize).clamp(0, max_y) as usize;
                sum += src[yi * stride + col + c] as f32;
            }
            for y in 0..h {
                dst[y * stride + col + c] = round_u8(sum * inv_win);
                let y_out = (y as isize - r as isize).clamp(0, max_y) as usize;
                let y_in = (y as isize + r as isize + 1).clamp(0, max_y) as usize;
                sum += src[y_in * stride + col + c] as f32 - src[y_out * stride + col + c] as f32;
            }
        }
        for y in 0..h {
            dst[y * stride + col + 3] = 255;
        }
    }
}

#[inline]
fn round_u8(v: f32) -> u8 {
    // Round-half-away-from-zero: `as u8` truncates, which biases the
    // blurred output ~0.5 LSB per channel. The summed sample values are
    // already non-negative, so a simple `+ 0.5` works.
    (v + 0.5).clamp(0.0, 255.0) as u8
}
