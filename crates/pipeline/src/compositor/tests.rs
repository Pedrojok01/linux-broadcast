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
