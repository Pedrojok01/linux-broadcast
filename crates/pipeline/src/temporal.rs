/// Default EMA weight on the current frame. Values in `[0.6, 0.8]` look
/// natural — present frame dominates so motion stays responsive while the
/// remaining history weight damps shimmer at the silhouette.
pub const DEFAULT_ALPHA: f32 = 0.7;

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
        self.prev = Some(mask.to_vec());
    }

    pub fn reset(&mut self) {
        self.prev = None;
    }
}
