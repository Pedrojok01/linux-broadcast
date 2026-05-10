//! Long-running per-pixel estimate of the room behind the user, and the
//! cold-start `bg_mean` colour used as a fallback.

/// Per-pixel EMA learning rate when a sample is confidently background.
/// At 30 fps this gives an effective half-life of roughly one second —
/// fast enough to track gentle lighting drift, slow enough that a brief
/// segmenter false-negative on the body doesn't bake the user into the
/// plate.
const PLATE_LEARN_RATE: f32 = 0.05;

/// `bg_certainty = 1 - mask α` must exceed this to fold a pixel into the
/// plate. Set high (0.9 → α < 0.1) so even silhouette-edge pixels never
/// contribute, eliminating the "ghost me" artefact that would otherwise
/// be exposed at the original source position when auto-frame remaps the
/// live silhouette to a new output position.
const PLATE_BG_THRESHOLD: f32 = 0.9;

/// Cumulative-weight threshold above which the plate is trusted at a
/// pixel. Below this, the bg materializer falls back to the global
/// `bg_mean` colour. ~6-12 confident updates' worth, so cold start
/// transitions to "real plate" within roughly half a second per pixel
/// once exposed.
pub(super) const PLATE_CONF_THRESHOLD: f32 = 0.3;

/// Long-running per-pixel estimate of the room behind the user.
///
/// On each non-`None` composite the plate folds in the current frame
/// at pixels where the segmentation mask reports confidently-bg
/// (`1 - α > PLATE_BG_THRESHOLD`), using a small EMA. After a few
/// seconds of natural movement the plate is a person-free copy of the
/// actual scene. `Background::Blur` uses it as the input to the blur,
/// so the blurred bg shows the actual wall rather than a smeared
/// version of the person — and the tight threshold ensures the
/// silhouette never bakes into the plate, so auto-frame doesn't
/// expose a "ghost me" at the original position when it remaps the
/// live silhouette.
///
/// Per-pixel `conf` accumulates the weights ever applied at that
/// pixel, capped at 1.0. `conf < PLATE_CONF_THRESHOLD` marks
/// "haven't seen real bg here yet" and the materializer fills with
/// `bg_mean` instead. `reset()` zeros both buffers — the feeder calls
/// it on Live exit so the next engagement starts learning fresh in
/// case the camera or lighting changed during the idle window.
pub(super) struct BgPlate {
    pub(super) rgba: Vec<u8>,
    pub(super) conf: Vec<f32>,
    pub(super) width: u32,
    pub(super) height: u32,
}

impl BgPlate {
    pub(super) fn new() -> Self {
        Self {
            rgba: Vec::new(),
            conf: Vec::new(),
            width: 0,
            height: 0,
        }
    }

    /// Drop all accumulated state. Next call to [`Self::update`] starts
    /// from a cold plate; the `bg_mean` cold-start fallback in the
    /// materializer covers the visual gap until the EMA re-converges.
    pub(super) fn reset(&mut self) {
        self.rgba.clear();
        self.conf.clear();
        self.width = 0;
        self.height = 0;
    }

    /// Fold one frame into the plate. `mask` is the upsampled
    /// foreground probability at frame resolution. Pixels with
    /// `mask >= PLATE_BG_THRESHOLD` are skipped entirely; the rest are
    /// EMA'd at a rate proportional to their bg-certainty.
    pub(super) fn update(&mut self, frame: &[u8], mask: &[f32], width: u32, height: u32) {
        let n_px = (width as usize) * (height as usize);
        if self.width != width || self.height != height {
            self.rgba.clear();
            self.rgba.resize(n_px * 4, 0);
            self.conf.clear();
            self.conf.resize(n_px, 0.0);
            self.width = width;
            self.height = height;
        }
        for (i, &m) in mask.iter().take(n_px).enumerate() {
            let bg_certainty = 1.0 - m.clamp(0.0, 1.0);
            if bg_certainty <= PLATE_BG_THRESHOLD {
                continue;
            }
            let alpha = PLATE_LEARN_RATE * bg_certainty;
            let inv = 1.0 - alpha;
            let pi = i * 4;
            self.rgba[pi] = (self.rgba[pi] as f32 * inv + frame[pi] as f32 * alpha) as u8;
            self.rgba[pi + 1] =
                (self.rgba[pi + 1] as f32 * inv + frame[pi + 1] as f32 * alpha) as u8;
            self.rgba[pi + 2] =
                (self.rgba[pi + 2] as f32 * inv + frame[pi + 2] as f32 * alpha) as u8;
            self.rgba[pi + 3] = 255;
            self.conf[i] = (self.conf[i] + alpha).min(1.0);
        }
    }
}

/// Mask-weighted mean RGB of the input frame: pixels with low mask
/// (confidently background) dominate; foreground pixels contribute
/// little. Used as the cold-start fallback colour wherever the
/// temporal plate hasn't accumulated enough confidence yet. f64
/// accumulators keep the 1280×720 sum precise.
pub(super) fn compute_bg_mean(frame: &[u8], mask: &[f32], n_px: usize) -> [u8; 3] {
    let mut bg_sum = [0.0_f64; 3];
    let mut bg_w = 0.0_f64;
    for i in 0..n_px {
        let w = (1.0 - mask[i].clamp(0.0, 1.0)) as f64;
        bg_w += w;
        bg_sum[0] += frame[i * 4] as f64 * w;
        bg_sum[1] += frame[i * 4 + 1] as f64 * w;
        bg_sum[2] += frame[i * 4 + 2] as f64 * w;
    }
    if bg_w > 1.0 {
        [
            (bg_sum[0] / bg_w) as u8,
            (bg_sum[1] / bg_w) as u8,
            (bg_sum[2] / bg_w) as u8,
        ]
    } else {
        // Frame is essentially all foreground — there's no bg
        // signal to estimate from. Mid-grey is the least-bad
        // default and only applies to a degenerate input.
        [128, 128, 128]
    }
}
