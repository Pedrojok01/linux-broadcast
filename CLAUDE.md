# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NVIDIA Broadcast-style virtual webcam for Linux. Captures a webcam frame, runs MediaPipe Selfie Segmentation (Landscape) on CPU via `tract`, composites the person over a blurred background or a still image, and writes the result to a `v4l2loopback` virtual camera that Zoom / Meet / Teams / Firefox / OBS consume.

The project is a **Rust** rewrite of an earlier Python prototype. The Python code lives on the `legacy-python` branch — useful for reading composite math but not for execution.

Out of scope: audio / microphone effects.

## Common commands

```bash
# One-time host setup (also handled by the .deb postinst once we ship)
sudo modprobe v4l2loopback video_nr=10 card_label="Linux Broadcast" exclusive_caps=1 max_buffers=2

# Build & run (Phase 1: hardcoded /dev/video0 → /dev/video10, blur)
cargo run --release

# Override source / sink devices without editing code
LB_SOURCE=/dev/video2 LB_SINK=/dev/video10 cargo run --release

# Lint / format
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all

# Verify the virtual cam from another terminal
cheese -d /dev/video10
```

System dev packages required to build:

```bash
sudo apt install -y \
  build-essential pkg-config \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libxkbcommon-dev libwayland-dev \
  libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev \
  v4l2loopback-dkms \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
```

Pin `v4l2loopback-dkms >= 0.12.8` — version 0.12.7 fails to build on kernel 6.8+ (Ubuntu 24.04 / Mint).

## Architecture

Cargo workspace with two crates:

- **`crates/pipeline`** (lib `lb_pipeline`) — the entire video pipeline. Headless, no GUI deps.
- **`crates/app`** (bin `linux-broadcast`) — `egui` GUI that drives the pipeline. Owns config persistence and the v4l2loopback first-run wizard.

### Frame pipeline (one GStreamer graph, one inference loop)

```
v4l2src device=$SOURCE
  ! videoconvert ! videoscale
  ! video/x-raw,format=RGBA,width=W,height=H,framerate=FPS/1
  ! appsink (callback)
                        │
                        ▼
   Segmenter (tract, MediaPipe selfie_segmenter, 256×256)
                        │
                        ▼
   MaskSmoother (EMA α=0.7 across frames)
                        │
                        ▼
   Compositor (bilinear-upsample mask, blur OR image background)
                        │
                        ▼
appsrc
  ! videoconvert ! video/x-raw,format=YUY2,...
  ! v4l2sink device=$SINK
```

The `appsink` callback is wired with `connect_new_sample` (callback-based, not the `AppSinkStream` API — see gstreamer-rs issue #346 for the race). Each callback also drains a `crossbeam-channel` of `Command`s from the GUI so live setting changes (e.g. switching from blur to a new background image) take effect on the next frame without restarting the graph.

### Cross-cutting conventions

- **Pixel format inside the pipeline is RGBA8** end-to-end. Conversion to/from camera-native (YUYV/MJPEG) happens in GStreamer's `videoconvert` elements; Rust code never touches non-RGBA pixels.
- **The mask is `f32` everywhere.** Round-trips to `u8` cost quality and don't save memory at our resolutions.
- **Inference runs at the model's native 256×144.** Only the upsample+composite step touches frame-resolution pixels. This is what keeps CPU usage low — see `compositor::Compositor::upsample_mask`.
- **No CUDA, no torch, no OpenCV.** `tract` is pure Rust; resizing is `fast_image_resize`; blur is a hand-rolled separable box blur. Single static binary modulo libc + GStreamer.
- **Reusable buffers in `Segmenter` and `Compositor`** — both keep their working buffers in `self` to avoid per-frame allocations.

### Model

Bundled at compile time via `include_bytes!("../../../models/selfie_segmenter.onnx")` from `crates/app/src/main.rs`. Sourced from `onnx-community/mediapipe_selfie_segmentation` on Hugging Face — pre-converted, no `tf2onnx` step needed. The general 256×256 variant is what's published; the 256×144 landscape variant would have been slightly more aspect-friendly but isn't on this repo.

Per [google-ai-edge/mediapipe#6134](https://github.com/google-ai-edge/mediapipe/issues/6134), the raw model output requires post-processing — sigmoid for the single-channel landscape variant, softmax across the channel axis if you swap in a 2-channel variant. `Segmenter::segment` handles both.

### Files at a glance

- `crates/pipeline/src/segmenter.rs` — `tract` session + RGBA→f32 NHWC pre, sigmoid/softmax post.
- `crates/pipeline/src/compositor.rs` — bilinear mask upsample, separable box blur, RGBA composite.
- `crates/pipeline/src/temporal.rs` — `MaskSmoother` EMA. ~10 lines, biggest single quality win.
- `crates/pipeline/src/pipeline.rs` — GStreamer graph + appsink callback.
- `crates/app/src/main.rs` — Phase 1 entry point (hardcoded; GUI lands in Phase 2).

### Adding a new background mode

1. Add a variant to `Background` in `compositor.rs`.
2. Add a branch to `Compositor::composite` that produces the new background plane (frame-sized RGBA8) and reuses the `out = fg*mask + bg*(1-mask)` formula.
3. Plumb a `Command` from the GUI via `Pipeline::cmd_sender`.

## Phase status

- **Phase 0** ✅ — Python preserved on `legacy-python`, `main` wiped.
- **Phase 1** 🚧 — vertical slice scaffolded; needs MediaPipe ONNX in `models/`, GStreamer dev libs installed, and a smoke run against `/dev/video10`.
- **Phase 2** — `egui` GUI with camera picker / blur toggle / background picker / Start-Stop, settings persisted to `~/.config/linux-broadcast/config.toml`.
- **Phase 3** — quality polish: edge feather, light wrap, cover-fit background scaling.
- **Phase 4** — `.deb` (with `v4l2loopback-dkms` postinst modprobe + `/etc/modules-load.d/`) and Flatpak (`flatpak-spawn --host pkexec modprobe …` first-run wizard, OBS PR #4552 pattern).
- **Phase 5** — GitHub Actions release workflow + benchmarks.

The detailed plan lives at `/home/pedrojok/.claude/plans/ok-now-build-a-expressive-flurry.md` (developer-local).
