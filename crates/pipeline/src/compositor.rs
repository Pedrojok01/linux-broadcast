//! Composite the segmented foreground over a chosen background plane.
//!
//! Steps, in order:
//! 1. **Mask prep** — bilinear-upsample the model mask to frame resolution
//!    when needed. RVM already returns frame-resolution masks, so the
//!    upsample short-circuits.
//! 2. **Background plate update** — for any non-`None` mode, fold the
//!    current frame into a long-running per-pixel EMA that only
//!    samples pixels we're *very* confident are background (per-pixel
//!    mask α < `1 - PLATE_BG_THRESHOLD`). Over a few seconds of natural
//!    movement the plate becomes a person-free copy of the actual room.
//!    Used as the source for the `Blur` mode bg plane (blurring the live
//!    frame would smear the person into the bg). See `BgPlate`.
//! 3. **Background plane** — produced once per frame from the active
//!    [`Background`]:
//!    - `Blur`: a two-pass separable box blur of the temporal plate
//!      (`bg_mean`-filled where plate confidence is low, e.g. cold
//!      start).
//!    - `Image`: a cached scaled-to-cover RGBA copy of the user's
//!      library image.
//!    - `None`: skips the blend entirely — the feeder pushes the
//!      camera frame straight to `appsrc`. With auto-frame on, falls
//!      through to step 5 (no blend, just crop the raw camera frame).
//! 4. **Blend** — `out = fg * mask + bg * (1 - mask)` per pixel, in RGBA8.
//!    - **Image + framing**: asymmetric remap. The background image is
//!      sampled at unshifted output coords (the virtual room stays
//!      put), while foreground RGBA + mask are bilinearly sampled at
//!      the remapped source position so the silhouette gets recentered
//!      and lightly zoomed in front of the static bg. Alpha
//!      decontamination uses the temporal plate as a `B_estimate` to
//!      remove the user's real wall colour from soft silhouette edges
//!      (no white halo around the user).
//!    - All other paths use a clean in-place fast path. Foreground =
//!      current frame.
//! 5. **Auto-frame** (Blur and None only) — when framing is on, snapshot
//!    the composite (or the raw frame, for None) and bilinearly resample
//!    a cropped window of it back over the output. Bg and fg move
//!    together → no plate ghost is exposed (Blur), no halo seam, no
//!    layer ambiguity. Image bg uses the asymmetric remap above instead
//!    so the virtual wall genuinely stays put.
//!
//! Buffers (resizers, working planes, blur scratch, plate, fg scratch)
//! live on `Compositor` to avoid per-frame allocations. The blur
//! kernel falls back to a single pass below `BLUR_TWO_PASS_THRESHOLD`
//! and switches to two passes above it, pushing the box-kernel output
//! closer to a Gaussian without paying for the second pass at low
//! strengths.
//!
//! The temporal plate decouples the blur source from the live frame:
//! blurring the live frame would always carry a soft outline of the
//! person, even though the live silhouette gets composited cleanly on
//! top. With auto-frame on, that outline would be exposed at the
//! *original* source position once the silhouette is remapped — the
//! "ghost me" artefact. The tight `PLATE_BG_THRESHOLD` ensures only
//! pixels we're highly confident are wall ever update the plate, so
//! the ghost never bakes in to begin with. Cold-start fills with a
//! global background-mean colour for the ~1-2 seconds it takes the EMA
//! to converge.

use anyhow::{Result, anyhow};
use fast_image_resize::{
    self as fr, FilterType, ResizeAlg, ResizeOptions,
    images::{Image, ImageRef},
};

use crate::Mask;

/// Minimum kernel radius (px) at `strength = 0.0`. Anything weaker than
/// this looked muddy rather than blurred — segmentation imperfections at
/// the silhouette edge stay visible against a barely-defocused plate, so
/// the slider's 0% now starts where a usable blur actually begins.
pub const BLUR_MIN_RADIUS: usize = 12;
/// Maximum kernel radius (px) at `strength = 1.0`. Past ~32 px the
/// background becomes unreadable for any text or facial cues.
pub const BLUR_MAX_RADIUS: usize = 32;
/// Radius (px) at and above which the second blur pass kicks in to push
/// the box-kernel output closer to a Gaussian.
const BLUR_TWO_PASS_THRESHOLD: usize = 8;

/// Affine remap describing the auto-frame transform. The same field set
/// drives two different paths inside [`Compositor::composite`]:
///
/// - **Image + framing** (asymmetric remap): foreground RGBA + mask α
///   are sampled at the remapped source position; the background image
///   is sampled at unshifted output coordinates. Result: the virtual
///   room stays still and the silhouette slides over it.
/// - **Blur + framing and None + framing** (post-composite crop+rescale):
///   the same formula maps every output pixel to a source position in
///   the *composited* frame (or the raw camera, for None), and we
///   bilinearly resample. Both layers move together — wall zooms with
///   the user. In Blur this is invisible because the wall is blurred;
///   in None it just looks like a normal PTZ zoom.
///
/// Used by auto-frame to recentre horizontally on the silhouette
/// centroid and apply a static [`crate::framing::FG_ZOOM`] anchored at
/// the head-top.
#[derive(Debug, Clone, Copy)]
pub struct Framing {
    /// Source pixel coordinates of the anchor point (the location in the
    /// composited frame that should land at `dst_anchor` in the output).
    pub src_anchor_x: f32,
    pub src_anchor_y: f32,
    /// Output pixel coordinates the anchor lands at.
    pub dst_anchor_x: f32,
    pub dst_anchor_y: f32,
    /// Zoom factor, ≥ 1.0.
    pub zoom: f32,
}

impl Framing {
    /// True when the framing has no visible effect (zoom 1.0 and
    /// anchors coincide), so the compositor can skip the crop+rescale
    /// pass entirely.
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
    /// Gaussian-blur the temporal background plate and use it as the
    /// background. `strength` is in `[0.0, 1.0]` and maps to a kernel
    /// radius from `BLUR_MIN_RADIUS` (barely-visible) to
    /// `BLUR_MAX_RADIUS` (strong). The plate is a per-pixel EMA of
    /// the source frame restricted to confidently-bg samples (see
    /// `BgPlate`), so the bg plane is the actual room behind the
    /// user — never the user themselves.
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
const PLATE_CONF_THRESHOLD: f32 = 0.3;

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
struct BgPlate {
    rgba: Vec<u8>,
    conf: Vec<f32>,
    width: u32,
    height: u32,
}

impl BgPlate {
    fn new() -> Self {
        Self {
            rgba: Vec::new(),
            conf: Vec::new(),
            width: 0,
            height: 0,
        }
    }

    /// Drop all accumulated state. Next call to [`Self::update`] starts
    /// from a cold plate; the `bg_mean` cold-start fallback in
    /// [`Compositor::run_blur`] covers the visual gap until the EMA
    /// re-converges.
    fn reset(&mut self) {
        self.rgba.clear();
        self.conf.clear();
        self.width = 0;
        self.height = 0;
    }

    /// Fold one frame into the plate. `mask` is the upsampled
    /// foreground probability at frame resolution. Pixels with
    /// `mask >= PLATE_BG_THRESHOLD` are skipped entirely; the rest are
    /// EMA'd at a rate proportional to their bg-certainty.
    fn update(&mut self, frame: &[u8], mask: &[f32], width: u32, height: u32) {
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
    /// `Background::Image` asymmetric blend can sample foreground RGBA
    /// from a remapped source position while writing to the in-place
    /// output. Allocated lazily on first Image+framing call.
    fg_scratch: Vec<u8>,
    /// Snapshot of the clean composite (after the in-place blend, before
    /// the post-composite crop+rescale). Used by `Background::Blur` and
    /// `Background::None` when framing is on, so the crop reads from a
    /// stable source while writing to the in-place output. Allocated
    /// lazily.
    composite_scratch: Vec<u8>,
    /// Materialized bg plate at frame resolution (plate values where
    /// confidence is high, `bg_mean` fallback elsewhere). Used as the
    /// `B_estimate` for alpha decontamination in `asymmetric_blend` and
    /// as the input to the box blur in `Background::Blur`.
    plate_materialized: Vec<u8>,
    /// Background plane for `Background::Blur`: starts as a copy of
    /// `plate_materialized`, then box-blurred in place. Reused across
    /// frames to dodge a 1280×720×4 = 3.6 MB allocation per tick.
    blur_out: Vec<u8>,
    /// Scratch buffer used as the intermediate for the separable box
    /// blur (horizontal pass writes here, vertical pass reads from it).
    blur_tmp: Vec<u8>,
    /// Long-running estimate of the room behind the user. Updated
    /// every non-`None` composite. See `BgPlate`.
    plate: BgPlate,
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
            composite_scratch: Vec::new(),
            plate_materialized: Vec::new(),
            blur_out: Vec::new(),
            blur_tmp: Vec::new(),
            plate: BgPlate::new(),
        }
    }

    /// Discard the temporal background plate. Called by the feeder on
    /// Live exit so the next engagement starts learning from scratch
    /// (lighting and camera position may have changed during the idle
    /// window). Cheap — just clears two `Vec`s.
    pub fn reset_bg_plate(&mut self) {
        self.plate.reset();
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
    /// The framing path depends on the active background mode:
    /// - **`Background::None`, no framing**: true bytewise passthrough.
    /// - **`Background::None`, framing on**: digital crop+rescale of the
    ///   raw camera frame around the silhouette anchor. No mask, no
    ///   plate, no blend.
    /// - **`Background::Blur`, no framing**: clean in-place blend over
    ///   the blurred plate.
    /// - **`Background::Blur`, framing on**: same blend, then a
    ///   post-composite crop+rescale of the entire output. Both the
    ///   blurred bg and the silhouette move together — any plate ghost
    ///   moves with the silhouette and is never exposed.
    /// - **`Background::Image`, no framing**: clean in-place blend over
    ///   the scaled-to-cover image.
    /// - **`Background::Image`, framing on**: asymmetric remap. The bg
    ///   image is sampled at unshifted output coords (the virtual room
    ///   stays put) while the foreground RGBA + mask are sampled at
    ///   the remapped source position. Alpha decontamination using the
    ///   plate as a `B_estimate` removes the wall colour from soft
    ///   silhouette edges, so there's no white halo around the user.
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

        let framing = framing.filter(|f| !f.is_identity());
        let w = width as usize;
        let h = height as usize;

        // Background::None + no framing: true bytewise passthrough.
        // No segmentation, no plate update, no blend.
        if matches!(background, Background::None) && framing.is_none() {
            return Ok(());
        }

        // Background::None + framing: pure digital crop on the raw
        // camera frame. The camera *is* the wall, so we just snapshot
        // the input and crop+rescale around the silhouette anchor.
        // No mask, no plate, no blend needed.
        if let (Background::None, Some(f)) = (background, framing) {
            self.ensure_composite_scratch(frame.len());
            self.composite_scratch.copy_from_slice(frame);
            crop_and_rescale_in_place(frame, &self.composite_scratch, w, h, f);
            return Ok(());
        }

        self.prepare_mask(mask, width, height)?;

        // Fold this frame into the temporal background plate. Done for
        // every non-None path so switching modes finds a primed plate
        // instead of a cold start. Cost is one O(N) pass; trivial next
        // to segmentation.
        self.plate
            .update(frame, &self.upsampled_mask, width, height);

        // Materialize the plate first whenever the frame will need it.
        // Blur always consumes it as the blur input. Image+framing
        // consumes it as the `B_estimate` for alpha decontamination
        // (Image+no-framing doesn't need it). Blur+framing no longer
        // needs the plate for decontamination — the post-composite crop
        // moves both layers together — but Blur always needs it for the
        // blur, regardless of framing.
        let needs_plate = matches!(background, Background::Blur { .. })
            || matches!(
                (background, framing),
                (Background::Image { .. }, Some(_))
            );
        if needs_plate {
            self.materialize_plate(frame, width, height);
        }

        match background {
            Background::None => unreachable!(),
            Background::Blur { strength } => {
                // Blur path: clean fg-over-blurred-plate blend, in place.
                // If framing is on, snapshot the composite and
                // crop+rescale the whole thing — Plate ghost (if any)
                // moves with the silhouette and is never exposed.
                self.run_blur(width, height, *strength);
                blend(frame, &self.blur_out, &self.upsampled_mask);
                if let Some(f) = framing {
                    self.ensure_composite_scratch(frame.len());
                    self.composite_scratch.copy_from_slice(frame);
                    crop_and_rescale_in_place(frame, &self.composite_scratch, w, h, f);
                }
            }
            Background::Image {
                rgba: bg_rgba,
                width: bw,
                height: bh,
            } => {
                // Image path: keep the asymmetric remap when framing is
                // on (bg image stays put, fg slides over it — the
                // distinctive virtual-room effect). No framing → clean
                // in-place blend.
                self.ensure_bg_scaled(bg_rgba, *bw, *bh, width, height)?;
                if let Some(f) = framing {
                    self.ensure_fg_scratch(frame.len());
                    self.fg_scratch.copy_from_slice(frame);
                    asymmetric_blend(
                        frame,
                        &self.bg_scaled,
                        &self.fg_scratch,
                        &self.plate_materialized,
                        &self.upsampled_mask,
                        w,
                        h,
                        f,
                    );
                } else {
                    blend(frame, &self.bg_scaled, &self.upsampled_mask);
                }
            }
        }

        Ok(())
    }

    fn ensure_fg_scratch(&mut self, n: usize) {
        if self.fg_scratch.len() != n {
            self.fg_scratch.resize(n, 0);
        }
    }

    fn ensure_composite_scratch(&mut self, n: usize) {
        if self.composite_scratch.len() != n {
            self.composite_scratch.resize(n, 0);
        }
    }

    /// Fill `self.plate_materialized` with the bg plate at frame
    /// resolution. Pixels with `plate.conf > PLATE_CONF_THRESHOLD` get
    /// the EMA value (the actual room behind the user); the rest fall
    /// back to a global `bg_mean` colour computed from the current
    /// frame, used during cold start. Caller must have run
    /// `plate.update` already.
    ///
    /// Used as input to the `Background::Blur` blur pass and as the
    /// `B_estimate` for alpha decontamination in `asymmetric_blend`.
    fn materialize_plate(&mut self, frame: &[u8], width: u32, height: u32) {
        let w = width as usize;
        let h = height as usize;
        let n = (w * h) * 4;

        self.plate_materialized.resize(n, 0);

        let bg_mean = compute_bg_mean(frame, &self.upsampled_mask, w * h);

        let plate_rgba = &self.plate.rgba;
        let plate_conf = &self.plate.conf;
        let plate_ready =
            self.plate.width == width && self.plate.height == height && !plate_rgba.is_empty();
        let out = &mut self.plate_materialized;
        for (i, &conf) in plate_conf.iter().take(w * h).enumerate() {
            let pi = i * 4;
            let use_plate = plate_ready && conf > PLATE_CONF_THRESHOLD;
            if use_plate {
                out[pi] = plate_rgba[pi];
                out[pi + 1] = plate_rgba[pi + 1];
                out[pi + 2] = plate_rgba[pi + 2];
            } else {
                out[pi] = bg_mean[0];
                out[pi + 1] = bg_mean[1];
                out[pi + 2] = bg_mean[2];
            }
            out[pi + 3] = 255;
        }
    }

    /// Build the bg plane for `Background::Blur` into `self.blur_out`:
    /// copy the materialized plate (caller must have populated it),
    /// then run a one- or two-pass separable box blur over it.
    fn run_blur(&mut self, width: u32, height: u32, strength: f32) {
        let w = width as usize;
        let h = height as usize;
        let n = (w * h) * 4;

        self.blur_out.resize(n, 0);
        self.blur_tmp.resize(n, 0);
        debug_assert_eq!(self.plate_materialized.len(), n);
        self.blur_out.copy_from_slice(&self.plate_materialized);

        let s = strength.clamp(0.0, 1.0);
        let span = (BLUR_MAX_RADIUS - BLUR_MIN_RADIUS) as f32;
        let radius = (BLUR_MIN_RADIUS as f32 + s * span).round() as usize;
        if radius == 0 {
            return;
        }

        box_blur_rgba(&mut self.blur_out, &mut self.blur_tmp, w, h, radius);
        if radius >= BLUR_TWO_PASS_THRESHOLD {
            box_blur_rgba(&mut self.blur_out, &mut self.blur_tmp, w, h, radius);
        }
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

/// Alpha composite of foreground over background using `mask`. In-place
/// fast path: foreground = current `frame`, output overwrites it. Used
/// when no auto-frame transform is supplied.
fn blend(frame: &mut [u8], bg: &[u8], mask: &[f32]) {
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
fn asymmetric_blend(
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
fn refine_mask(m: f32) -> f32 {
    const HALO_LO: f32 = 0.10;
    if m <= HALO_LO {
        0.0
    } else {
        ((m - HALO_LO) / (1.0 - HALO_LO)).min(1.0)
    }
}

/// Minimum α at which alpha decontamination is applied. Below this the
/// `1/α` division amplifies bilinear noise badly without changing the
/// visible output (the foreground contribution is already vanishingly
/// small). Picked so the noise stays under one channel quantization step.
const DECONTAM_MIN_ALPHA: f32 = 0.05;

/// Bilinear RGB sample (no mask) used for the plate `B_estimate` lookup
/// in `asymmetric_blend`. Edge taps are clamped to source bounds.
#[inline]
fn sample_rgb_bilinear(buf: &[u8], w: usize, h: usize, sx: f32, sy: f32) -> [f32; 3] {
    let x0 = sx.floor();
    let y0 = sy.floor();
    let fx = (sx - x0).clamp(0.0, 1.0);
    let fy = (sy - y0).clamp(0.0, 1.0);
    let xi0 = (x0 as isize).clamp(0, w as isize - 1) as usize;
    let xi1 = ((x0 as isize) + 1).clamp(0, w as isize - 1) as usize;
    let yi0 = (y0 as isize).clamp(0, h as isize - 1) as usize;
    let yi1 = ((y0 as isize) + 1).clamp(0, h as isize - 1) as usize;

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

/// Crop a window of `scratch` and bilinearly rescale it into `frame`.
/// Both buffers are RGBA8 of dimensions `w × h`. Used by the
/// `Background::Blur` and `Background::None` framing paths to apply the
/// auto-frame transform after (Blur) or instead of (None) compositing.
/// Sampling indices are edge-clamped, but the caller's anchor math
/// (`lazy.rs` `min_dst_y` clamp + horizontal `min_src_x..max_src_x`
/// clamp) keeps the window strictly inside the source in normal
/// operation.
fn crop_and_rescale_in_place(frame: &mut [u8], scratch: &[u8], w: usize, h: usize, f: Framing) {
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

/// Bilinear sample of foreground RGB and mask α at fractional source
/// coords. Returns `(α in [0,1], rgb as f32)`. Edge taps are clamped to
/// the source bounds.
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
    let fx = (sx - x0).clamp(0.0, 1.0);
    let fy = (sy - y0).clamp(0.0, 1.0);
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

/// Mask-weighted mean RGB of the input frame: pixels with low mask
/// (confidently background) dominate; foreground pixels contribute
/// little. Used as the cold-start fallback colour wherever the
/// temporal plate hasn't accumulated enough confidence yet. f64
/// accumulators keep the 1280×720 sum precise.
fn compute_bg_mean(frame: &[u8], mask: &[f32], n_px: usize) -> [u8; 3] {
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

/// Two-pass separable box blur on RGBA8 in `buf`, using `tmp` as the
/// horizontal-pass intermediate. Alpha is forced to 255.
fn box_blur_rgba(buf: &mut [u8], tmp: &mut [u8], w: usize, h: usize, r: usize) {
    blur_horizontal(buf, tmp, w, h, r);
    blur_vertical(tmp, buf, w, h, r);
}

fn blur_horizontal(src: &[u8], dst: &mut [u8], w: usize, h: usize, r: usize) {
    let win = (2 * r + 1) as f32;
    for y in 0..h {
        let row = y * w * 4;
        for c in 0..3 {
            let mut sum = 0.0_f32;
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
        // bytewise short-circuit when no framing is supplied.
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
        // Identity framing must short-circuit and produce byte-identical
        // output to the no-framing call.
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
        // Asymmetric remap: the silhouette (blue, mask=1 at src x ∈ [4,8))
        // is translated by +8 px so it appears at output x ∈ [12, 16).
        // The bg plane (red image) is sampled at unshifted output coords,
        // so the rest of the output is solid red — including the strip
        // vacated on the trailing edge (output x ∈ [0, 8)) where the
        // remapped source falls outside the source frame.
        let (w, h) = (32, 8);
        let mut frame = solid_frame(w, h, [0, 0, 255, 255]); // raw input = blue
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
                let in_shifted = (12..16).contains(&x);
                if in_shifted {
                    assert_eq!(
                        &frame[pi..pi + 3],
                        &[0, 0, 255],
                        "expected blue fg at x={x}",
                    );
                } else {
                    // Both the rest of the source composite (red bg) and
                    // the edge-clamped vacated strip read from red pixels,
                    // so the whole non-shifted area is red.
                    assert_eq!(&frame[pi..pi + 3], &[255, 0, 0], "expected red bg at x={x}");
                }
            }
        }
    }

    #[test]
    fn framing_zoom_enlarges_silhouette() {
        // Asymmetric remap: foreground (blue square at src x,y ∈ [12,20))
        // is zoomed 2× around the frame center, so the silhouette extent
        // in output coords is x,y ∈ [8, 24). The bg plane (uniform red)
        // stays put.
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
        // → inside source silhouette → blue.
        let pi = (16 * w as usize + 10) * 4;
        assert_eq!(&frame[pi..pi + 3], &[0, 0, 255]);
        // Pixel at (28, 16): src_x = 16 + (28.5 - 16) * 0.5 - 0.5 = 21.75
        // → outside source silhouette (mask=0) → red.
        let pi = (16 * w as usize + 28) * 4;
        assert_eq!(&frame[pi..pi + 3], &[255, 0, 0]);
    }

    #[test]
    fn none_without_framing_is_bytewise_passthrough() {
        // None + None framing must remain a true bytewise short-circuit:
        // no segmentation, no plate update, no per-pixel rewrite.
        let (w, h) = (16, 4);
        let mut frame = vec![0u8; (w * h * 4) as usize];
        for (i, px) in frame.chunks_exact_mut(4).enumerate() {
            px[0] = (i % 251) as u8;
            px[1] = ((i * 7) % 251) as u8;
            px[2] = ((i * 13) % 251) as u8;
            px[3] = 255;
        }
        let original = frame.clone();
        let mask = mask_const(w, h, 1.0);
        let mut c = Compositor::new();
        c.composite(&mut frame, w, h, &mask, &Background::None, None)
            .unwrap();
        assert_eq!(frame, original);
    }

    #[test]
    fn none_with_framing_crops_raw_input() {
        // Background::None + non-identity framing crops + rescales the
        // raw camera frame around the anchor. Build an 8-wide horizontal
        // R-gradient and apply a 2× centred zoom; output should be the
        // middle half of the input stretched to full width.
        let (w, h) = (8u32, 4u32);
        let mut frame = vec![0u8; (w * h * 4) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let pi = (y * w as usize + x) * 4;
                frame[pi] = (x * 32) as u8; // R = 0, 32, 64, 96, 128, 160, 192, 224
                frame[pi + 3] = 255;
            }
        }
        // Mask is irrelevant in None mode (no segmentation consumed).
        let mask = mask_const(w, h, 0.0);
        let cx = w as f32 * 0.5;
        let cy = h as f32 * 0.5;
        let framing = Framing {
            src_anchor_x: cx,
            src_anchor_y: cy,
            dst_anchor_x: cx,
            dst_anchor_y: cy,
            zoom: 2.0,
        };
        let mut c = Compositor::new();
        c.composite(&mut frame, w, h, &mask, &Background::None, Some(framing))
            .unwrap();

        // src_x for output x is 4 + (x + 0.5 - 4) * 0.5 - 0.5 = 1.75 + 0.5·x.
        // x=0 → src_xf 1.75 → bilinear of R[1]=32 and R[2]=64 with fx=0.75
        //                   → 32·0.25 + 64·0.75 = 56.
        // x=7 → src_xf 5.25 → bilinear of R[5]=160 and R[6]=192 with fx=0.25
        //                   → 160·0.75 + 192·0.25 = 168.
        let r_at = |x: usize| frame[x * 4] as i32;
        assert!((r_at(0) - 56).abs() <= 2, "col 0 R = {}", r_at(0));
        assert!((r_at(7) - 168).abs() <= 2, "col 7 R = {}", r_at(7));
        // Sanity: must differ from the original gradient.
        assert_ne!(r_at(0), 0);
        assert_ne!(r_at(7), 224);
    }

    /// Build an Image bg whose pixels encode their column index in the R
    /// channel (R = x). Useful for asserting that bg stays put under
    /// asymmetric framing — every output column should keep its
    /// column-index R value where the silhouette is absent.
    fn column_index_image_bg(w: u32, h: u32) -> Background {
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let pi = (y * w as usize + x) * 4;
                rgba[pi] = x as u8;
                rgba[pi + 3] = 255;
            }
        }
        Background::Image {
            rgba,
            width: w,
            height: h,
        }
    }

    #[test]
    fn framing_keeps_image_bg_static() {
        // Image bg encodes column index in R. Source frame is solid
        // green (so any leakage of fg into bg-only territory would be
        // glaringly visible). Mask is fully zero — there's no fg at all.
        // Under asymmetric framing the bg must stay byte-identical to
        // the bg image at every output pixel, regardless of how the
        // (non-existent) silhouette would be remapped.
        let (w, h) = (16, 4);
        let mut frame = solid_frame(w, h, [0, 200, 0, 255]); // raw camera = green
        let mask = mask_const(w, h, 0.0); // no foreground
        let framing = Framing {
            src_anchor_x: 4.0,
            src_anchor_y: h as f32 * 0.5,
            dst_anchor_x: 12.0,
            dst_anchor_y: h as f32 * 0.5,
            zoom: 1.2,
        };
        let mut c = Compositor::new();
        c.composite(
            &mut frame,
            w,
            h,
            &mask,
            &column_index_image_bg(w, h),
            Some(framing),
        )
        .unwrap();
        for y in 0..h as usize {
            for x in 0..w as usize {
                let pi = (y * w as usize + x) * 4;
                assert_eq!(
                    frame[pi] as usize, x,
                    "bg must be unchanged at ({x},{y}); got R={}",
                    frame[pi]
                );
                assert_eq!(frame[pi + 1], 0, "G must be 0 (no fg leak)");
                assert_eq!(frame[pi + 2], 0, "B must be 0 (no fg leak)");
            }
        }
    }

    #[test]
    fn refine_mask_kills_low_alpha_halo() {
        // Mask is uniformly 0.05 — the kind of soft-tail value RVM
        // produces well outside the actual silhouette. With the asymmetric
        // path on, refine_mask should clamp this to 0, so the output is
        // pure background (no green leak from the camera frame).
        let (w, h) = (16, 4);
        let mut frame = solid_frame(w, h, [0, 200, 0, 255]); // green camera
        let mask = mask_const(w, h, 0.05);
        let framing = Framing {
            src_anchor_x: w as f32 * 0.5,
            src_anchor_y: h as f32 * 0.5,
            dst_anchor_x: w as f32 * 0.5 + 1.0, // tiny shift so framing is non-identity
            dst_anchor_y: h as f32 * 0.5,
            zoom: 1.0,
        };
        let mut c = Compositor::new();
        c.composite(&mut frame, w, h, &mask, &red_image_bg(w, h), Some(framing))
            .unwrap();
        // Inspect a column safely inside the source bounds (so the
        // out-of-bounds early-out doesn't carry the assertion). Output
        // x = 4 → src_x = 8 + (4.5 - 9) * 1 - 0.5 = 3.0, well inside.
        for y in 0..h as usize {
            let pi = (y * w as usize + 4) * 4;
            assert_eq!(&frame[pi..pi + 3], &[255, 0, 0], "halo leaked at y={y}");
        }
    }

    #[test]
    fn refine_mask_preserves_high_alpha_silhouette() {
        // Mask = 1.0 everywhere. refine_mask(1.0) saturates to 1.0, so the
        // output should be pure foreground (the raw camera frame) where
        // the remapped source is in bounds.
        let (w, h) = (16, 4);
        let mut frame = solid_frame(w, h, [0, 200, 0, 255]);
        let mask = mask_const(w, h, 1.0);
        let framing = Framing {
            src_anchor_x: w as f32 * 0.5,
            src_anchor_y: h as f32 * 0.5,
            dst_anchor_x: w as f32 * 0.5 + 1.0,
            dst_anchor_y: h as f32 * 0.5,
            zoom: 1.0,
        };
        let mut c = Compositor::new();
        c.composite(&mut frame, w, h, &mask, &red_image_bg(w, h), Some(framing))
            .unwrap();
        // Output x = 4 → in-bounds src; output should be the green fg.
        for y in 0..h as usize {
            let pi = (y * w as usize + 4) * 4;
            assert_eq!(&frame[pi..pi + 3], &[0, 200, 0], "fg lost at y={y}");
        }
    }

    #[test]
    fn blur_framing_does_not_alias() {
        // Smoke test: Blur + non-identity framing must complete without
        // panic and must actually transform the frame (proves the new
        // post-composite crop path fires). We don't assert specific
        // pixel values — Blur's box-blur kernel is `BLUR_MIN_RADIUS` px
        // even at strength 0, which means a tiny 16×8 frame would just
        // be smeared into a uniform colour and the test would say
        // nothing useful. Use a "large enough" frame and a translation
        // framing, then check the output is not byte-identical to a
        // no-framing run.
        let (w, h) = (64, 64);
        let mut frame_a = vec![0u8; (w * h * 4) as usize];
        for (i, px) in frame_a.chunks_exact_mut(4).enumerate() {
            px[0] = (i % 251) as u8;
            px[1] = ((i * 5) % 251) as u8;
            px[2] = ((i * 11) % 251) as u8;
            px[3] = 255;
        }
        let mut frame_b = frame_a.clone();
        let mask = mask_const(w, h, 0.5);
        let bg = Background::Blur { strength: 0.0 };

        let mut c1 = Compositor::new();
        c1.composite(&mut frame_a, w, h, &mask, &bg, None).unwrap();
        let framing = Framing {
            src_anchor_x: 20.0,
            src_anchor_y: 20.0,
            dst_anchor_x: w as f32 * 0.5,
            dst_anchor_y: h as f32 * 0.5,
            zoom: 1.2,
        };
        let mut c2 = Compositor::new();
        c2.composite(&mut frame_b, w, h, &mask, &bg, Some(framing))
            .unwrap();
        assert_ne!(
            frame_a, frame_b,
            "Blur+framing must change the output vs Blur+no-framing"
        );
    }

    #[test]
    fn blur_framing_crops_whole_composite() {
        // With Blur+framing the *whole* composite (foreground silhouette
        // AND blurred bg) should crop together. Test at zoom=1 with a
        // pure horizontal translation: the silhouette and any bg
        // structure must shift by the same amount.
        //
        // Setup: 32×8 frame, mask=1.0 only in a vertical stripe at
        // src x ∈ [4, 8). Outside the stripe, mask=0 → output equals
        // bg (blurred plate). On the very first frame the plate hasn't
        // been primed yet, so the bg falls back to bg_mean — but
        // bg_mean here is computed from a uniform red frame, so it's
        // a constant red. The blend produces "red everywhere except
        // blue stripe at [4,8)". Translate by +8 → stripe at [12,16),
        // red elsewhere.
        let (w, h) = (32, 8);
        let mut frame = solid_frame(w, h, [0, 0, 200, 255]); // raw camera = blue
        // Add a deterministic "wall" by overwriting pixels outside the
        // stripe with red — simulates the camera frame showing a wall
        // around the user. Mask will tag those as bg.
        for y in 0..h as usize {
            for x in 0..w as usize {
                if !(4..8).contains(&x) {
                    let pi = (y * w as usize + x) * 4;
                    frame[pi] = 200;
                    frame[pi + 1] = 0;
                    frame[pi + 2] = 0;
                }
            }
        }
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
        c.composite(
            &mut frame,
            w,
            h,
            &mask,
            &Background::Blur { strength: 0.0 },
            Some(framing),
        )
        .unwrap();

        // Spot-check the centre of the shifted silhouette is dominantly
        // blue, and a far-away bg pixel is dominantly red. The blur
        // softens the transition, so we accept any pixel whose blue >
        // red as "fg-dominant" and vice versa.
        let center_pi = (4 * w as usize + 13) * 4; // shifted silhouette centre
        assert!(
            frame[center_pi + 2] > frame[center_pi],
            "shifted silhouette centre at out x=13 should be blue-dominant; got rgb=({}, {}, {})",
            frame[center_pi],
            frame[center_pi + 1],
            frame[center_pi + 2],
        );
        let far_pi = (4 * w as usize + 28) * 4; // well outside the shift
        assert!(
            frame[far_pi] > frame[far_pi + 2],
            "far bg pixel at out x=28 should be red-dominant; got rgb=({}, {}, {})",
            frame[far_pi],
            frame[far_pi + 1],
            frame[far_pi + 2],
        );
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
