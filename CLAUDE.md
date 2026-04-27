# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Background-replacement virtual webcam for Linux. Captures a webcam frame, runs MediaPipe Selfie Segmentation on CPU via `ort` (ONNX Runtime), composites the person over a blurred background, a saved image, or passes the frame through unchanged, and writes the result to a `v4l2loopback` virtual camera that Zoom / Meet / Teams / Firefox / OBS consume.

The project is a Rust rewrite of an earlier Python prototype. The Python code lives on the `legacy-python` branch — useful for reading composite math but not for execution.

Out of scope: audio / microphone effects.

## Common commands

```bash
# One-time host setup (also handled by the .deb postinst once we ship)
sudo modprobe -r v4l2loopback 2>/dev/null
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Linux Broadcast" \
  exclusive_caps=1 max_buffers=2

# GUI (default)
cargo run --release -p linux-broadcast

# Headless (uses the saved config, no window)
LB_HEADLESS=1 cargo run --release -p linux-broadcast

# Dump the bundled window icon to /tmp/lb-icon.png (sanity check)
LB_DUMP_ICON=1 cargo run --release -p linux-broadcast

# Lint / format
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all

# Verify the virtual cam from another terminal (only works while the
# pipeline is running — exclusive_caps=1 hides /dev/video10 otherwise).
ffplay -fflags nobuffer -f v4l2 -input_format yuyv422 \
  -video_size 1280x720 /dev/video10
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

Pin `v4l2loopback-dkms ≥ 0.12.8` — version 0.12.7 fails to build on kernel 6.8+ (Ubuntu 24.04 / Mint).

## Architecture

Cargo workspace with two crates:

- **`crates/pipeline`** (lib `lb_pipeline`) — the entire video pipeline. Headless, no GUI deps.
- **`crates/app`** (bin `linux-broadcast`) — `eframe`/`egui` GUI that drives the pipeline. Owns config persistence, the saved-background library, the theme, and the desktop-entry installer.

### Frame pipeline (one GStreamer graph, one inference loop)

```
v4l2src device=$SOURCE
  ! videoscale ! videoconvert
  ! video/x-raw,format=RGBA,width=W,height=H
  ! appsink (callback, sync=false, async=false)
                        │
                        ▼
   Background::None?  ── yes ──► passthrough (skip seg + composite)
                        │ no
                        ▼
   Segmenter (ort) — one of:
     ├─ ModelKind::SelfieBinary     (256×256 NCHW, prob output)
     ├─ ModelKind::SelfieMulticlass (256×256 NHWC, 6-class logits)
     └─ ModelKind::Rvm              (frame-resolution, recurrent state)
                        │
                        ▼
   MaskSmoother (EMA α=0.7 across frames)
                        │
                        ▼
   Compositor (bilinear-upsample mask → blur | image | …)
                        │
                        ▼
appsrc (live, framerate=FPS/1)
  ! videoconvert ! video/x-raw,format=YUY2,…
  ! v4l2sink device=$SINK (sync=false, async=false)
```

Key non-obvious settings on the GStreamer side:

- **`v4l2sink async=false`** — without it the pipeline deadlocks: v4l2sink stays in PAUSED waiting for a preroll buffer from appsrc, which has nothing to push because new_sample only fires once we're in PLAYING.
- **`appsink sync=false`** + a `new_preroll` callback that pulls and discards the preroll buffer, so live frames flow.
- **`v4l2src do-timestamp=true`** — without it the output stream's PTS is wrong and v4l2sink's pacing slips.
- **Caps strategy:** the source-side capsfilter pins **only RGBA + width/height** (no framerate). Forcing 30/1 caused negotiation failures with the C920 (camera reports `30000/1001`). The appsrc-side caps **do** declare framerate, otherwise the sink-side videoconvert asserts `fps_n == out_fps_n`.
- The appsink callback is wired with `set_callbacks` (callback-based, not `AppSinkStream` — see gstreamer-rs#346).

The callback also drains a `crossbeam-channel` of `Command`s from the GUI so live setting changes (mode swap, blur strength, picking a new background image) take effect on the next frame without restarting the graph. Camera, resolution, and **model** changes do require a graph restart, which the GUI handles by Stop+Start.

### Cross-cutting conventions

- **Pixel format inside the pipeline is RGBA8** end-to-end. Conversion to/from camera-native (YUYV/MJPEG) and to YUY2 for the v4l2sink happens in GStreamer's `videoconvert` elements; Rust code never touches non-RGBA pixels.
- **The mask is `f32` everywhere** between segmentation and composite. `u8` round-trips cost quality without saving memory.
- **Inference runs at the model's native 256×256.** Only the upsample + composite step touches frame-resolution pixels. This is what keeps CPU usage low at 720p/1080p.
- **`Background::None` is a true passthrough.** The new_sample callback short-circuits *before* segmentation and resets the EMA smoother so the next non-None frame starts clean.
- **No CUDA, no torch, no OpenCV.** ORT runs on the CPU EP only; resizing is `fast_image_resize`; blur is a hand-rolled separable box-blur. `ort` pulls `libonnxruntime.so` automatically via its `download-binaries` feature.
- **Reusable buffers in `Segmenter` and `Compositor`** — both keep their working buffers in `self` to avoid per-frame allocations.

### Models

Three ONNX files, all bundled at compile time via `include_bytes!` in `crates/app/src/main.rs`:

- **`models/selfie_segmenter.onnx`** (~450 KB). Sourced from `onnx-community/mediapipe_selfie_segmentation` on Hugging Face — pre-converted, no `tf2onnx` step needed. Input `[batch, 3, 256, 256]` NCHW, output `[batch, 1, 256, 256]` NCHW. The single-channel output is **already a probability** (final sigmoid baked into the graph) — `Segmenter::segment` clamps to `[0,1]` and forwards it. *Applying a second sigmoid was an early bug that clamped masks to `[0.5, 0.731]` and made the blur invisible — leaving this note here as a caution.*
- **`models/selfie_multiclass.onnx`** (~16 MB). Converted locally from MediaPipe's `selfie_multiclass_256x256.tflite` via `tf2onnx` (one-time, executed via `uvx`/`uv run`; no permanent Python deps). Input `[1, 256, 256, 3]` NHWC, output `[1, 256, 256, 6]` NHWC of raw logits. Six classes: `0` background, `1` hair, `2` body-skin, `3` face-skin, `4` clothes, `5` others. Foreground = `1 - softmax(logits)[0]` per pixel. Edge quality is noticeably better than the binary model on tricky scenes (similar luminance, hair detail).
- **`models/rvm.onnx`** (~15 MB). Robust Video Matting (MobileNetV3 backbone, fp32) from PeterL1n/RobustVideoMatting v1.0.0. Recurrent video matting: 6 inputs (`src` plus 4 recurrent state tensors `r1i…r4i` plus a `downsample_ratio` scalar), 6 outputs (`fgr`, `pha`, and the 4 next-frame states `r1o…r4o`). We use only `pha` as the mask; `fgr` is discarded. The output mask is at the *frame* resolution — the compositor's `prepare_mask` skips upsampling when the mask already matches the frame. Internal compute is scaled by `RVM_DOWNSAMPLE_RATIO = 0.4` (in `segmenter.rs`) which keeps compute around 480×270 internally on a 720p frame, ~30–40 ms/frame on a single x86 core. The recurrent state is held on the segmenter and reset by `Segmenter::reset()` whenever the user switches to `Background::None` or changes input dimensions, so re-engaging starts clean.

`Mask` is a public type carrying `(data, width, height)` so each model can declare its native mask resolution. The MediaPipe variants emit 256×256, RVM emits frame-resolution. The Compositor handles either.

The active model is chosen via `lb_pipeline::ModelKind` (re-exported as the GUI's serde-friendly `config::Model`). The GUI dropdown drives a Stop+Start cycle on change.

#### Re-fetching the bundled models

```bash
# Multiclass (TFLite → ONNX, one-time conversion via uv-managed Python).
curl -L -o /tmp/selfie_multiclass.tflite \
  https://huggingface.co/yolain/selfie_multiclass_256x256/resolve/main/selfie_multiclass_256x256.tflite
uv run --python 3.10 --with "tf2onnx==1.16.1" --with "tensorflow==2.14.0" --with "numpy<2" \
  python -m tf2onnx.convert \
    --tflite /tmp/selfie_multiclass.tflite \
    --output models/selfie_multiclass.onnx \
    --opset 18

# RVM (already ONNX from upstream).
curl -L -o models/rvm.onnx \
  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx
```

### Files at a glance

Pipeline (`crates/pipeline/src/`):
- `segmenter.rs` — `Segmenter` enum dispatching to a binary, multiclass, or RVM `ort::Session`. MediaPipe variants share a resize-to-256×256 step and differ on layout (NCHW vs NHWC) and post (clamp vs softmax). RVM holds 4 recurrent state tensors across calls and exposes `reset()` for clean re-engagement.
- `compositor.rs` — bilinear mask upsample, two-pass separable box blur (radius 4–32 px from `Background::Blur { strength }`), plain alpha composite via `blend()`.
- `temporal.rs` — `MaskSmoother` (EMA across frames). Note: experimental `sharpen_mask` / `feather_mask` / `light_wrap` were tried and reverted — the raw mask composites cleaner perceptually.
- `pipeline.rs` — GStreamer graph + appsink callback + bus sync handler + `Command` channel.
- `lib.rs` — `MODEL_W`/`MODEL_H` constants and the public re-exports.

App (`crates/app/src/`):
- `main.rs` — entry point; embeds both ONNXs and dispatches to GUI / headless / icon-dump.
- `theme.rs` — design tokens (colors, spacing, radii, control sizes) applied to `egui::Style`/`Visuals`. Inter and JetBrains Mono TTFs registered via `FontDefinitions`.
- `ui.rs` — `eframe::App` + sidebar / preview / footer layout, segmented mode control, blur-intensity slider, library grid, model picker.
- `cameras.rs` — enumerate `/dev/video*` and probe friendly names from `/sys/class/video4linux/<n>/name`.
- `backgrounds.rs` — saved-image library at `~/.local/share/linux-broadcast/backgrounds/`.
- `config.rs` — TOML config at `~/.config/linux-broadcast/config.toml` (`Mode`, `Model`, blur strength, source/sink, mirror).
- `icon.rs` — programmatically rasterizes the logo SVG to a 64×64 RGBA `IconData` (no `usvg` dep — the logo is four primitives).
- `desktop_install.rs` — drops `~/.local/share/icons/hicolor/64x64/apps/io.Pedrojok01.LinuxBroadcast.png` and a matching `.desktop` file on first launch so Wayland compositors (KDE / GNOME) can resolve the window's `app_id` to a real taskbar icon.

Design assets:
- `assets/fonts/` — Inter-Variable + JetBrainsMono-Regular.
- `DESIGN.md` — colour / spacing / type tokens + design rationale (intent doc; `theme.rs` is authoritative for values).

### Adding a new background mode

1. Add a variant to `Background` in `compositor.rs`.
2. Add a branch to `Compositor::composite` that produces the new background plane (frame-sized RGBA8) and reuses `out = fg*mask + bg*(1-mask)`.
3. Add a corresponding `Mode` to `app/src/config.rs`.
4. Add the segmented-control tab in `ui.rs::sidebar_scene` and wire it through `build_background`.
5. The pipeline picks up the new mode via the existing `Command::SetBackground` — no graph rebuild needed.

### Adding a new model

1. Add a `ModelKind` variant in `pipeline/src/segmenter.rs` and a matching `segment_*` function for its pre/post.
2. Bundle the ONNX in `models/` and `include_bytes!` it from `crates/app/src/main.rs`.
3. Extend the call to `Pipeline::start` to pass the new bytes.
4. Add a config-side `Model` variant (with serde) in `app/src/config.rs` and surface it in the GUI's `sidebar_model` dropdown.
5. The GUI auto-restarts the pipeline on model change, so no graph plumbing is required.

## Phase status

- **Phase 0** ✅ — Python preserved on `legacy-python`, `main` wiped.
- **Phase 1** ✅ — vertical slice live: 30 fps at 1280×720, blur + image backgrounds, headless mode.
- **Phase 2** ✅ — `egui` GUI: camera dropdown, model picker (binary / multiclass / RVM), mode tabs, blur-intensity slider, saved-background library, live preview pane, themed to the design tokens.
- **Phase 3** ⏳ — quality polish: mostly held until model swap saturates. Remaining: real horizontal-mirror toggle (UI exists, plumbing TODO), CPU/GPU usage in footer.
- **Phase 4** — `.deb` (with `v4l2loopback-dkms` postinst modprobe + `/etc/modules-load.d/`) and Flatpak (`flatpak-spawn --host pkexec modprobe …` first-run wizard, OBS PR #4552 pattern). The `desktop_install` module already handles per-user icon registration ahead of full packaging.
- **Phase 5** — GitHub Actions release workflow + benchmarks.

The original implementation plan lives at `/home/pedrojok/.claude/plans/ok-now-build-a-expressive-flurry.md` (developer-local) and is now mostly historical.
