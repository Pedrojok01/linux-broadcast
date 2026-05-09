//! Auto-framing: smoothed horizontal recenter + light foreground zoom.
//!
//! The compositor blends `fg * mask + bg * (1 - mask)`. Auto-framing
//! reparameterizes the per-output-pixel *foreground sample point* so the
//! silhouette is recentered horizontally and slightly enlarged. The
//! background plane (blurred original frame, or replacement image) is
//! sampled at unshifted output coordinates, so it stays put while the
//! person slides over it. The strip vacated on the trailing edge gets
//! filled by the background — that's just the `mask = 0` case of the
//! existing blend.
//!
//! The zoom is a static [`FG_ZOOM`] (no UI control). Anchors are picked
//! so a 1.15× zoom doesn't crop heads:
//! - **Horizontal:** silhouette centroid → frame center. Mass-weighted
//!   so a hand on a desk doesn't dominate (see `horizontal_midpoint`).
//! - **Vertical:** silhouette *top edge* → [`TOP_HEADROOM`] of the way
//!   down the frame, but capped so the source bottom always maps to
//!   (at least) the output bottom. Centering on the vertical centroid
//!   would crop the head when zoomed; pinning the top edge keeps the
//!   head in frame regardless of how much body is visible. The cap
//!   matters at low [`FG_ZOOM`]: without it, the silhouette wouldn't
//!   reach the output bottom and a band of the (unshifted) background
//!   plane would peek through under the body.
//!
//! When no foreground is detected, [`BBoxSmoother::update`] returns
//! `None` and the feeder skips framing for that frame. When auto-framing
//! is on but the user picks `Background::None`, the feeder skips framing
//! entirely — there's no background plane, so a translated/zoomed
//! foreground would leave a hole.

use crate::Mask;

/// Mask probability above which a pixel counts as foreground for the
/// centroid computation. Conservative — well below the 0.5 threshold
/// used for compositing — so faint silhouette edges still influence the
/// recentering.
const FG_THRESHOLD: f32 = 0.5;

/// Per-row mask-mass threshold (as a fraction of row width) above which
/// a row is considered "head-top." Sized so a stray edge pixel doesn't
/// pull the top anchor up but the actual top of the head registers.
const HEAD_TOP_ROW_FRACTION: f32 = 0.02;

/// EMA factor applied to the smoothed normalized `cx`. 0.10 at 30 fps
/// lags real motion by ~0.3 s — fast enough to feel like centering,
/// slow enough to ignore single-frame mask noise.
pub const DEFAULT_ALPHA: f32 = 0.10;
/// Horizontal deadzone in normalized coords. Below this, hold the
/// previous `cx`. Sized so a stationary user with a slightly noisy
/// mask never produces visible micro-shift, but tight enough that the
/// EMA tail settles to within ~3 px of true center on a 1280-wide
/// frame.
pub const DEFAULT_DEADZONE_X: f32 = 0.002;
/// Vertical deadzone in normalized coords. The head-top edge is noisier
/// than the centroid (single mask row), so this is intentionally larger
/// than [`DEFAULT_DEADZONE_X`] to avoid head-bobble.
pub const DEFAULT_DEADZONE_Y: f32 = 0.005;

/// Static foreground zoom factor applied whenever auto-frame is on.
/// A subtle "lean-in" effect; deliberately not user-adjustable. Past
/// ~1.2× the segmentation edges and any pixelation become visible.
pub const FG_ZOOM: f32 = 1.05;
/// Where the silhouette's top edge should land in the output frame, as
/// a fraction of frame height. Small enough to feel like a tight
/// portrait crop, large enough that mask noise above the head doesn't
/// poke past the top edge.
pub const TOP_HEADROOM: f32 = 0.08;

/// Smoothed framing anchor in normalized source coords. Output of
/// [`BBoxSmoother::update`].
#[derive(Debug, Clone, Copy)]
pub struct Anchor {
    /// Mass-weighted horizontal centroid in `[0, 1]`.
    pub cx_norm: f32,
    /// Topmost row with significant mask mass, in `[0, 1]`.
    pub top_y_norm: f32,
}

/// Smoothed silhouette tracker. One instance per running pipeline.
pub struct BBoxSmoother {
    /// Last published `cx` in normalized `[0, 1]`. `None` until the
    /// first detection — and reset back to `None` on engagement
    /// boundaries (toggle off, exited Live) so the next engagement
    /// snaps to the new position instead of panning from a stale one.
    current_cx: Option<f32>,
    /// Last published top-edge `y` in normalized `[0, 1]`.
    current_top_y: Option<f32>,
    alpha: f32,
    deadzone_x: f32,
    deadzone_y: f32,
}

impl Default for BBoxSmoother {
    fn default() -> Self {
        Self::new()
    }
}

impl BBoxSmoother {
    pub fn new() -> Self {
        Self {
            current_cx: None,
            current_top_y: None,
            alpha: DEFAULT_ALPHA,
            deadzone_x: DEFAULT_DEADZONE_X,
            deadzone_y: DEFAULT_DEADZONE_Y,
        }
    }

    /// Compute the smoothed framing anchor for this mask. Returns
    /// `None` when no foreground pixel exceeds [`FG_THRESHOLD`] — feeder
    /// skips framing in that case so the output is the unshifted
    /// composited frame.
    pub fn update(&mut self, mask: &Mask) -> Option<Anchor> {
        let target_cx = horizontal_midpoint(mask)?;
        let target_top = top_edge(mask).unwrap_or(target_cx); // fall back if all rows below threshold
        let cx = ema_step(self.current_cx, target_cx, self.alpha, self.deadzone_x);
        let top_y = ema_step(self.current_top_y, target_top, self.alpha, self.deadzone_y);
        self.current_cx = Some(cx);
        self.current_top_y = Some(top_y);
        Some(Anchor {
            cx_norm: cx,
            top_y_norm: top_y,
        })
    }

    /// Drop smoother state. Call on engagement boundaries (toggling
    /// auto-frame off, dropping out of Live) so the next engagement
    /// snaps to the new position instead of drifting from a stale one.
    pub fn reset(&mut self) {
        self.current_cx = None;
        self.current_top_y = None;
    }
}

fn ema_step(prev: Option<f32>, target: f32, alpha: f32, deadzone: f32) -> f32 {
    match prev {
        None => target,
        Some(p) => {
            if (target - p).abs() < deadzone {
                p
            } else {
                p + (target - p) * alpha
            }
        }
    }
}

/// Mass-weighted horizontal centroid of the silhouette in normalized
/// source-frame coordinates: `Σ(x · m) / Σ(m)` over all pixels with
/// `m > FG_THRESHOLD`. `None` if no pixel exceeds the threshold.
///
/// Centroid is preferred over bbox midpoint because the bbox is
/// dominated by whichever extremity sticks out furthest — a hand on a
/// desk shifts the bbox edge by ~10% of the frame even though it adds
/// only a few pixels of mass. The centroid weights by silhouette area,
/// so the resulting `cx` tracks the torso rather than the extremities.
fn horizontal_midpoint(mask: &Mask) -> Option<f32> {
    if mask.width == 0 || mask.height == 0 {
        return None;
    }
    let w = mask.width as usize;
    let h = mask.height as usize;
    let mut sum_xm = 0.0_f64;
    let mut sum_m = 0.0_f64;
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let m = mask.data[row + x];
            if m > FG_THRESHOLD {
                let m = m as f64;
                sum_xm += (x as f64 + 0.5) * m;
                sum_m += m;
            }
        }
    }
    if sum_m == 0.0 {
        return None;
    }
    Some((sum_xm / sum_m / mask.width as f64) as f32)
}

/// Topmost row whose total mask mass exceeds
/// `HEAD_TOP_ROW_FRACTION × width`, in normalized `[0, 1]`. `None` if no
/// row qualifies. Row mass = sum of probabilities, so a row with 2% full
/// pixels qualifies just as much as one with 4% half-prob pixels.
fn top_edge(mask: &Mask) -> Option<f32> {
    if mask.width == 0 || mask.height == 0 {
        return None;
    }
    let w = mask.width as usize;
    let h = mask.height as usize;
    let row_min_mass = (w as f32) * HEAD_TOP_ROW_FRACTION;
    for y in 0..h {
        let row = y * w;
        let mass: f32 = mask.data[row..row + w].iter().sum();
        if mass >= row_min_mass {
            return Some((y as f32 + 0.5) / h as f32);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask_with_rect(w: u32, h: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> Mask {
        let mut data = vec![0.0f32; (w * h) as usize];
        for y in y0..y1 {
            for x in x0..x1 {
                data[(y * w + x) as usize] = 1.0;
            }
        }
        Mask {
            data,
            width: w,
            height: h,
        }
    }

    fn empty_mask(w: u32, h: u32) -> Mask {
        Mask {
            data: vec![0.0; (w * h) as usize],
            width: w,
            height: h,
        }
    }

    #[test]
    fn empty_mask_returns_none() {
        let mut s = BBoxSmoother::new();
        assert!(s.update(&empty_mask(50, 50)).is_none());
    }

    #[test]
    fn first_detection_snaps_to_anchor() {
        let mut s = BBoxSmoother::new();
        // Silhouette occupying x ∈ [20, 60), y ∈ [10, 90) of a 100×100 mask.
        // Centroid x = (20 + 59 + 1) / 2 / 100 = 0.40.
        // Top row = 10, normalized = 10.5/100 = 0.105.
        let mask = mask_with_rect(100, 100, 20, 10, 60, 90);
        let a = s.update(&mask).expect("detection");
        assert!((a.cx_norm - 0.40).abs() < 1e-3, "cx={}", a.cx_norm);
        assert!(
            (a.top_y_norm - 0.105).abs() < 1e-3,
            "top_y={}",
            a.top_y_norm
        );
    }

    #[test]
    fn deadzone_holds_against_micro_shift() {
        let mut s = BBoxSmoother::new();
        // First call seeds — silhouette midpoint at 0.5.
        let m1 = mask_with_rect(1000, 100, 400, 10, 600, 90);
        let first = s.update(&m1).unwrap();
        // Shift midpoint by 1/1000 = 0.001, well below the 0.002 deadzone.
        let m2 = mask_with_rect(1000, 100, 401, 10, 601, 90);
        let second = s.update(&m2).unwrap();
        assert_eq!(
            first.cx_norm, second.cx_norm,
            "tiny x shift should be deadzoned",
        );
    }

    #[test]
    fn ema_blends_partial_pan_toward_target() {
        let mut s = BBoxSmoother::new();
        let _ = s.update(&mask_with_rect(100, 100, 20, 10, 60, 90));
        let after_seed = s.current_cx.unwrap();
        let _ = s.update(&mask_with_rect(100, 100, 40, 10, 80, 90));
        let after_step = s.current_cx.unwrap();
        assert!(
            after_step > after_seed && after_step < 0.6,
            "after_step={after_step} (seed={after_seed}, target=0.6)",
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut s = BBoxSmoother::new();
        let _ = s.update(&mask_with_rect(100, 100, 20, 10, 60, 90));
        assert!(s.current_cx.is_some());
        assert!(s.current_top_y.is_some());
        s.reset();
        assert!(s.current_cx.is_none());
        assert!(s.current_top_y.is_none());
        // After reset, the next detection should snap (no EMA blend).
        let a = s.update(&mask_with_rect(100, 100, 70, 10, 90, 90)).unwrap();
        assert!((a.cx_norm - 0.80).abs() < 1e-3, "cx={}", a.cx_norm);
    }

    #[test]
    fn top_edge_picks_first_substantial_row() {
        // Row 0 has only 1 pixel of mass (well below the 2% × 100 = 2.0
        // threshold). Real silhouette starts at row 10 with 40 pixels.
        // Top should track row 10, not row 0.
        let mut data = vec![0.0f32; 100 * 100];
        data[5] = 1.0; // stray noise pixel in row 0
        for y in 10..90 {
            for x in 30..70 {
                data[y * 100 + x] = 1.0;
            }
        }
        let mask = Mask {
            data,
            width: 100,
            height: 100,
        };
        let mut s = BBoxSmoother::new();
        let a = s.update(&mask).unwrap();
        assert!(
            (a.top_y_norm - 0.105).abs() < 1e-3,
            "top_y={}",
            a.top_y_norm,
        );
    }
}
