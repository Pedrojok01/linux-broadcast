/// Default EMA weight on the current frame for the per-frame MediaPipe
/// models. Values in `[0.6, 0.8]` look natural — present frame dominates
/// so motion stays responsive while the remaining history weight damps
/// shimmer at the silhouette.
pub const DEFAULT_ALPHA: f32 = 0.7;

/// EMA weight for RVM. RVM has its own recurrent temporal smoothing
/// baked into the network (the 4 r*i / r*o state tensors), so stacking
/// a heavy EMA on top mostly just blurs edges during motion. 0.95 keeps
/// a tiny residual flicker damp without softening the matte.
pub const RVM_ALPHA: f32 = 0.95;

/// Exponential moving-average smoothing of the per-pixel mask across frames.
///
/// MediaPipe's selfie segmentation is per-frame, so the mask flickers slightly
/// between adjacent frames. EMA `mask_t = α·m_now + (1-α)·m_prev` knocks
/// down most of the perceptible flicker for two lines of code.
pub struct MaskSmoother {
    alpha: f32,
    prev: Option<Vec<f32>>,
}

impl MaskSmoother {
    /// `alpha` is the weight of the new frame; 0.6–0.8 looks natural.
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            prev: None,
        }
    }

    /// Smooth `mask` in place using the previous frame's mask.
    pub fn smooth(&mut self, mask: &mut [f32]) {
        match &self.prev {
            Some(prev) if prev.len() == mask.len() => {
                let a = self.alpha;
                let inv = 1.0 - a;
                for i in 0..mask.len() {
                    mask[i] = a * mask[i] + inv * prev[i];
                }
            }
            _ => {}
        }
        // Reuse the previous-frame Vec when its size matches; only
        // allocate (or grow) when the mask resolution changes. At RVM
        // frame resolution this saves a multi-MB allocation per tick.
        let prev = self.prev.get_or_insert_with(|| Vec::with_capacity(mask.len()));
        if prev.len() != mask.len() {
            prev.resize(mask.len(), 0.0);
        }
        prev.copy_from_slice(mask);
    }

    pub fn reset(&mut self) {
        self.prev = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn first_call_passes_through() {
        let mut s = MaskSmoother::new(0.5);
        let mut m = vec![0.1, 0.4, 0.7, 1.0];
        let original = m.clone();
        s.smooth(&mut m);
        assert_eq!(m, original);
    }

    #[test]
    fn reset_drops_state() {
        let mut s = MaskSmoother::new(0.5);
        let mut a = vec![0.0; 4];
        s.smooth(&mut a);
        s.reset();
        let mut b = vec![0.9_f32; 4];
        let original = b.clone();
        s.smooth(&mut b);
        assert_eq!(b, original, "after reset, next smooth must pass through");
    }

    #[test]
    fn ema_converges_to_constant() {
        let mut s = MaskSmoother::new(0.7);
        for _ in 0..100 {
            let mut m = vec![0.42_f32; 8];
            s.smooth(&mut m);
        }
        let mut probe = vec![0.42_f32; 8];
        s.smooth(&mut probe);
        for v in probe {
            assert!((v - 0.42).abs() < 1e-5, "got {v}");
        }
    }

    #[test]
    fn alpha_zero_freezes_output() {
        // α = 0 → output equals prev forever after the first call.
        let mut s = MaskSmoother::new(0.0);
        let mut first = vec![0.3_f32, 0.6, 0.9];
        s.smooth(&mut first);
        let prev = first.clone();
        let mut next = vec![0.0_f32, 0.0, 0.0];
        s.smooth(&mut next);
        assert_eq!(next, prev, "α=0 should ignore the new frame entirely");
    }

    proptest! {
        #[test]
        fn output_bounded_for_bounded_input(
            inputs in proptest::collection::vec(
                proptest::collection::vec(0.0f32..=1.0, 4..=4),
                1..=10,
            ),
            alpha in 0.0f32..=1.0,
        ) {
            let mut s = MaskSmoother::new(alpha);
            for frame in inputs {
                let mut m = frame;
                s.smooth(&mut m);
                for &v in &m {
                    prop_assert!((0.0..=1.0).contains(&v), "out of range: {v}");
                }
            }
        }
    }
}
