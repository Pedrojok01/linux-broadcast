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

mod blend;
mod blur;
mod plate;
mod sample;

#[cfg(test)]
mod tests;

use anyhow::{Result, anyhow};
use fast_image_resize::{
    self as fr, FilterType, ResizeAlg, ResizeOptions,
    images::{Image, ImageRef},
};

use crate::Mask;
use blend::{asymmetric_blend, blend};
use blur::box_blur_rgba;
use plate::{BgPlate, PLATE_CONF_THRESHOLD, compute_bg_mean};
use sample::crop_and_rescale_in_place;

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
            || matches!((background, framing), (Background::Image { .. }, Some(_)));
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
