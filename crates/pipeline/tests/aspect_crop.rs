//! Aspect-preserving crop test.
//!
//! The source graph center-crops the camera frame to the output aspect
//! ratio (via `aspectratiocrop`) before `videoscale`, so a target whose
//! ratio differs from the camera's is never *stretched*. This verifies the
//! crop directly: a 16:9 input fed at a non-16:9 target comes out carrying
//! the target aspect, with the height untouched and only the width cropped
//! — i.e. cropped, not squashed.
//!
//! Requires `gstreamer1.0-plugins-good` (videotestsrc + videocrop). No
//! `/dev/video0` or `v4l2loopback` needed.

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;

/// Run a 16:9 `videotestsrc` frame through `aspectratiocrop` at `target`
/// and return the cropped frame's `(width, height)`.
fn cropped_dims(in_w: i32, in_h: i32, target_w: i32, target_h: i32) -> (i32, i32) {
    gst::init().unwrap();
    let pipeline = gst::Pipeline::new();
    let src = gst::ElementFactory::make("videotestsrc")
        .property("num-buffers", 1i32)
        .build()
        .unwrap();
    let in_caps = gst::Caps::builder("video/x-raw")
        .field("width", in_w)
        .field("height", in_h)
        .build();
    let in_filter = gst::ElementFactory::make("capsfilter")
        .property("caps", &in_caps)
        .build()
        .unwrap();
    let crop = gst::ElementFactory::make("aspectratiocrop")
        .property("aspect-ratio", gst::Fraction::new(target_w, target_h))
        .build()
        .unwrap();
    let sink = gst_app::AppSink::builder().sync(false).build();
    pipeline
        .add_many([&src, &in_filter, &crop, sink.upcast_ref()])
        .unwrap();
    gst::Element::link_many([&src, &in_filter, &crop, sink.upcast_ref()]).unwrap();

    pipeline.set_state(gst::State::Playing).unwrap();
    let sample = sink
        .try_pull_sample(gst::ClockTime::from_seconds(5))
        .expect("a cropped sample");
    let s = sample.caps().unwrap().structure(0).unwrap().to_owned();
    pipeline.set_state(gst::State::Null).unwrap();

    (
        s.get::<i32>("width").unwrap(),
        s.get::<i32>("height").unwrap(),
    )
}

#[test]
fn crop_converts_16_9_input_to_target_aspect_without_squashing() {
    // The regression case: 16:9 camera, 1280×900 (64:45) output.
    let (in_w, in_h) = (1280, 720);
    let (tw, th) = (1280, 900);
    let (w, h) = cropped_dims(in_w, in_h, tw, th);

    // Cropped frame carries the TARGET aspect, so the downstream uniform
    // videoscale to tw×th introduces no distortion.
    assert_eq!(w * th, h * tw, "cropped {w}x{h} aspect != target {tw}x{th}");
    // Target is narrower than 16:9, so width is cropped and height is left
    // intact — proving a horizontal crop, not a vertical squash.
    assert_eq!(
        h, in_h,
        "height changed ({in_h} → {h}); should crop width only"
    );
    assert!(w < in_w, "width not cropped ({in_w} → {w})");
}

#[test]
fn matching_aspect_is_a_passthrough() {
    // 16:9 input, 16:9 target → no crop at all.
    let (w, h) = cropped_dims(1280, 720, 1280, 720);
    assert_eq!((w, h), (1280, 720));
}
