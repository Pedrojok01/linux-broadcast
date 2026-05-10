//! Auto-framing: snap-on-first horizontal recenter + light foreground zoom.
//!
//! On the first detection after construction (or after [`AnchorLock::reset`])
//! we capture the silhouette anchor and hold it forever. Subsequent calls
//! return that locked anchor unchanged — small natural movements of a seated
//! user don't produce visible recentering. To pick a new anchor, the caller
//! resets the lock (the feeder does this when auto-frame is toggled off and
//! on again, and also on Live exit).
//!
//! This is the GMeet auto-frame UX: snap once at engagement, stay put. An
//! earlier EMA-based design panned every frame to track the centroid; users
//! reported it as a "I'm always moving" loop where the recentering and the
//! user's natural compensation kept feeding each other.
//!
//! The zoom is a static [`FG_ZOOM`] (no UI control). Anchors are picked
//! so the zoom doesn't crop heads:
//! - **Horizontal:** silhouette centroid → frame centre. Mass-weighted
//!   so a hand on a desk doesn't dominate (see `horizontal_midpoint`).
//! - **Vertical:** silhouette *top edge* → [`TOP_HEADROOM`] of the way
//!   down the frame, but capped so the source bottom always maps to
//!   (at least) the output bottom. Centring on the vertical centroid
//!   would crop the head when zoomed; pinning the top edge keeps the
//!   head in frame regardless of how much body is visible. The cap
//!   matters at low [`FG_ZOOM`]: without it the crop window would
//!   extend below the source frame and edge-clamped sampling would
//!   smear the bottom row across a visible band.
//!
//! When no foreground is detected on the very first update, the lock
//! never engages and [`AnchorLock::update`] returns `None` — the feeder
//! passes `framing = None` to the compositor and the frame is composited
//! without any zoom that tick. Once locked, the lock holds even through
//! later frames where the segmenter loses the silhouette.

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

/// Minimum zoom factor when auto-frame is on. Applied when the user is
/// already perfectly centred — a barely-perceptible "lean-in" that
/// keeps the framing from feeling completely flat.
pub const FG_ZOOM: f32 = 1.03;
/// Maximum zoom factor allowed when adaptive zoom kicks in to recentre
/// an off-centre user. Past ~1.2× the wall starts visibly zooming and
/// segmentation edges / pixelation become noticeable. Users
/// significantly past this offset get clamped (appear off-centre but
/// still in frame) rather than zoom even further.
pub const FG_ZOOM_MAX: f32 = 1.20;
/// Where the silhouette's top edge should land in the output frame, as
/// a fraction of frame height. Small enough to feel like a tight
/// portrait crop, large enough that mask noise above the head doesn't
/// poke past the top edge.
pub const TOP_HEADROOM: f32 = 0.08;

/// Framing anchor in normalised source coords. Output of
/// [`AnchorLock::update`].
#[derive(Debug, Clone, Copy)]
pub struct Anchor {
    /// Mass-weighted horizontal centroid in `[0, 1]`.
    pub cx_norm: f32,
    /// Topmost row with significant mask mass, in `[0, 1]`.
    pub top_y_norm: f32,
}

/// Snap-on-first silhouette tracker. One instance per running pipeline.
///
/// Captures the first detection's anchor and returns it unchanged on
/// every subsequent `update()`. `reset()` clears the lock so the next
/// detection seeds a new anchor.
pub struct AnchorLock {
    /// Anchor captured on the first detection after construction or
    /// `reset()`. `None` until that first detection succeeds.
    locked: Option<Anchor>,
}

impl Default for AnchorLock {
    fn default() -> Self {
        Self::new()
    }
}

impl AnchorLock {
    pub fn new() -> Self {
        Self { locked: None }
    }

    /// Compute the framing anchor for this mask.
    ///
    /// Once locked, returns the cached anchor on every call. If not yet
    /// locked, computes the centroid + top edge from this mask, locks
    /// to that, and returns it. Returns `None` only when the lock is
    /// not yet engaged AND no foreground exceeds [`FG_THRESHOLD`].
    pub fn update(&mut self, mask: &Mask) -> Option<Anchor> {
        if let Some(a) = self.locked {
            return Some(a);
        }
        let cx = horizontal_midpoint(mask)?;
        let top = top_edge(mask).unwrap_or(cx); // fall back if all rows below threshold
        let a = Anchor {
            cx_norm: cx,
            top_y_norm: top,
        };
        self.locked = Some(a);
        Some(a)
    }

    /// Drop the lock. The next [`update`](Self::update) call will seed a
    /// new anchor from the next mask that has any detectable foreground.
    /// Called by the feeder on engagement boundaries (auto-frame
    /// toggled off, pipeline exited Live).
    pub fn reset(&mut self) {
        self.locked = None;
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
        let mut s = AnchorLock::new();
        assert!(s.update(&empty_mask(50, 50)).is_none());
    }

    #[test]
    fn first_detection_snaps_to_anchor() {
        let mut s = AnchorLock::new();
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
    fn lock_holds_against_subsequent_movement() {
        // Snap to silhouette at centroid 0.4, then move it to centroid 0.7.
        // The lock must keep returning the original anchor.
        let mut s = AnchorLock::new();
        let first = s.update(&mask_with_rect(100, 100, 20, 10, 60, 90)).unwrap();
        // First anchor: centroid at 0.40.
        assert!((first.cx_norm - 0.40).abs() < 1e-3, "cx={}", first.cx_norm);
        // Now feed a very different silhouette (centroid 0.70).
        let second = s.update(&mask_with_rect(100, 100, 50, 10, 90, 90)).unwrap();
        assert_eq!(first.cx_norm, second.cx_norm);
        assert_eq!(first.top_y_norm, second.top_y_norm);
    }

    #[test]
    fn lock_holds_through_lost_detection() {
        // Once locked, the lock survives a frame with no foreground
        // (segmenter blink, user briefly leaving frame). Returns the
        // cached anchor instead of None.
        let mut s = AnchorLock::new();
        let first = s.update(&mask_with_rect(100, 100, 20, 10, 60, 90)).unwrap();
        let after_blank = s.update(&empty_mask(100, 100)).unwrap();
        assert_eq!(first.cx_norm, after_blank.cx_norm);
        assert_eq!(first.top_y_norm, after_blank.top_y_norm);
    }

    #[test]
    fn reset_clears_lock_and_re_snaps() {
        let mut s = AnchorLock::new();
        let _ = s.update(&mask_with_rect(100, 100, 20, 10, 60, 90));
        assert!(s.locked.is_some());
        s.reset();
        assert!(s.locked.is_none());
        // After reset, the next detection seeds a new lock — at the
        // *new* silhouette position, not the old one.
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
        let mut s = AnchorLock::new();
        let a = s.update(&mask).unwrap();
        assert!(
            (a.top_y_norm - 0.105).abs() < 1e-3,
            "top_y={}",
            a.top_y_norm,
        );
    }
}
