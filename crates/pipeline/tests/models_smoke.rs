//! Phase 3 — model smoke tests.
//!
//! Validates that each ONNX model loads and produces a sensible mask on
//! a synthetic input. Gated behind `#[ignore]` because each test loads
//! a real ONNX model and runs inference (~2-5 s each).
//!
//! Run with:
//!   cargo test --workspace -- --ignored
//! or:
//!   cargo nextest run --workspace --run-ignored only

use lb_pipeline::{ModelKind, Segmenter};

const MODEL_BINARY_ONNX: &[u8] = include_bytes!("../../../models/selfie_segmenter.onnx");
const MODEL_MULTICLASS_ONNX: &[u8] = include_bytes!("../../../models/selfie_multiclass.onnx");
const MODEL_RVM_ONNX: &[u8] = include_bytes!("../../../models/rvm.onnx");

/// Build a 256×256 RGBA gradient as a deterministic synthetic frame.
fn gradient_frame(w: u32, h: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            v.push((x * 255 / w.max(1)) as u8);
            v.push((y * 255 / h.max(1)) as u8);
            v.push(((x + y) * 255 / (w + h).max(1)) as u8);
            v.push(255);
        }
    }
    v
}

#[test]
#[ignore]
fn binary_segmenter_produces_valid_mask() {
    let mut s = Segmenter::from_bytes(ModelKind::SelfieBinary, MODEL_BINARY_ONNX).unwrap();
    let frame = gradient_frame(256, 256);
    let mask = s.segment(&frame, 256, 256).unwrap();
    assert_eq!(mask.width, 256);
    assert_eq!(mask.height, 256);
    assert_eq!(mask.data.len(), 256 * 256);
    for &v in &mask.data {
        assert!(v.is_finite(), "non-finite mask value: {v}");
        assert!((0.0..=1.0).contains(&v), "out of range: {v}");
    }
}

#[test]
#[ignore]
fn multiclass_segmenter_produces_valid_mask() {
    let mut s = Segmenter::from_bytes(ModelKind::SelfieMulticlass, MODEL_MULTICLASS_ONNX).unwrap();
    let frame = gradient_frame(256, 256);
    let mask = s.segment(&frame, 256, 256).unwrap();
    assert_eq!(mask.width, 256);
    assert_eq!(mask.height, 256);
    assert_eq!(mask.data.len(), 256 * 256);
    for &v in &mask.data {
        assert!(v.is_finite());
        assert!((0.0..=1.0).contains(&v));
    }
}

#[test]
#[ignore]
fn rvm_segmenter_recurrent_state_resets() {
    // Two segments, then reset(), then two more on the same input.
    // Recurrent state must come back to zero, so the post-reset
    // sequence reproduces the pre-reset sequence exactly.
    let (w, h) = (256u32, 256u32);
    let frame = gradient_frame(w, h);
    let mut s = Segmenter::from_bytes(ModelKind::Rvm, MODEL_RVM_ONNX).unwrap();

    let m1 = s.segment(&frame, w as usize, h as usize).unwrap();
    let _ = s.segment(&frame, w as usize, h as usize).unwrap();
    s.reset();
    let m1_again = s.segment(&frame, w as usize, h as usize).unwrap();

    assert_eq!(m1.data.len(), m1_again.data.len());
    // After reset, the very first segment of the same input should
    // match the very first segment from a fresh state. Allow a small
    // tolerance for non-determinism in ORT internal threading.
    let max_diff = m1
        .data
        .iter()
        .zip(&m1_again.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "post-reset first frame should match fresh first frame; max diff {max_diff}",
    );
}
