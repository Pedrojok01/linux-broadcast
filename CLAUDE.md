# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Background-replacement virtual webcam for Linux. Captures a webcam frame, runs MediaPipe Selfie Segmentation on CPU via `ort` (ONNX Runtime), composites the person over a blurred background, a saved image, or passes the frame through unchanged, and writes the result to a `v4l2loopback` virtual camera that Zoom / Meet / Teams / Firefox / OBS consume.

The project is a Rust rewrite of an earlier Python prototype. The Python code lives on the `legacy-python` branch — useful for reading composite math but not for execution.

Out of scope: audio / microphone effects.

## Common commands

```bash
# Manual host setup — only needed when building from source. The .deb's
# postinst runs the same commands automatically and ships conffiles in
# /etc/modprobe.d/ + /etc/modules-load.d/ so the module persists.
sudo modprobe -r v4l2loopback 2>/dev/null
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="LinuxBroadcast" \
  exclusive_caps=1 max_buffers=2

# GUI (default)
cargo run --release -p linux-broadcast

# Headless (uses the saved config, no window). Both forms work; the
# autostart .desktop uses --headless.
cargo run --release -p linux-broadcast -- --headless
LB_HEADLESS=1 cargo run --release -p linux-broadcast

# Dump the bundled window icon to /tmp/lb-icon.png. Also how
# packaging/LinuxBroadcast.png is regenerated when the logo changes.
LB_DUMP_ICON=1 cargo run --release -p linux-broadcast

# Lint / format
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all

# Build a local .deb (target/debian/linux-broadcast_<ver>_amd64.deb).
cargo install cargo-deb   # one-time
cargo deb -p linux-broadcast

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
  libgtk-3-dev libxdo-dev libayatana-appindicator3-dev \
  v4l2loopback-dkms \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
```

`libgtk-3-dev` + `libxdo-dev` + `libayatana-appindicator3-dev` are pulled in by the `tray-icon` crate. At runtime only `libayatana-appindicator3-1` is needed (declared in the `.deb`'s `depends`).

Pin `v4l2loopback-dkms ≥ 0.12.8` — version 0.12.7 fails to build on kernel 6.8+ (Ubuntu 24.04 / Mint).

## Architecture

Cargo workspace with two crates:

- **`crates/pipeline`** (lib `lb_pipeline`) — the entire video pipeline. Headless, no GUI deps.
- **`crates/app`** (bin `linux-broadcast`) — `eframe`/`egui` GUI that drives the pipeline. Owns config persistence, the saved-background library, the theme, and the desktop-entry installer.

### Frame pipeline (two GStreamer graphs, one feeder, lazy by default)

The pipeline runs in **lazy producer mode**: the physical camera (`/dev/video0`) is only opened while a real consumer is reading the virtual cam (`/dev/video10`) or the GUI preview pane is visible. To do that cleanly without the virtual cam blinking out of conferencing-app device lists, the pipeline is split into two GStreamer graphs glued by a single Rust feeder thread.

```
                          ┌──────────────────────┐
                          │ consumer_watch       │  /proc/*/fd poll @ ~1.25 Hz
                          │ thread               │  → Vec<Consumer> on changes
                          └──────────┬───────────┘
                                     │
                                     ▼
   ┌────────────  feeder thread (lazy::Feeder) ─────────────────────────┐
   │  state machine: Idle → Activating(2s debounce) → Live              │
   │                       Live → Deactivating(3s debounce) → Idle      │
   │  demand = consumers ∪ gui_preview_active                           │
   │  owns: Segmenter, Compositor, MaskSmoother, Background slot        │
   │                                                                    │
   │  on enter Live:  build & start source graph                        │
   │  on exit  Live:  set source → Null (releases /dev/video0, LED off) │
   │                                                                    │
   │  while Live, every ~33 ms:                                         │
   │    sample = source_appsink.try_pull_sample(33 ms)                  │
   │    if sample: segment + composite + push to sink_appsrc            │
   └────────────────────────────────────────────────────────────────────┘
                              │ build/teardown                ▲ push_buffer
                              ▼                               │
   ┌──────────  source graph (built on Live, dropped on Idle)  ─┐
   │ v4l2src device=$SOURCE                                     │
   │   ! videoscale ! videoconvert                              │
   │   ! video/x-raw,format=RGBA,width=W,height=H               │
   │   ! appsink (sync=false, drop=true, max-buffers=2)         │
   └────────────────────────────────────────────────────────────┘

   ┌────────  sink graph (built once, always PLAYING)  ───────────┐
   │ appsrc (RGBA, framerate=FPS/1, is_live=true, Format::Time)   │
   │   ! videoconvert                                             │
   │   ! video/x-raw,format=YUY2,width=W,height=H,framerate=FPS/1 │
   │   ! v4l2sink device=$SINK (sync=false, async=false)          │
   └──────────────────────────────────────────────────────────────┘
```

Key non-obvious settings:

- **Sink stays in PLAYING permanently, and the feeder keeps pushing.** The sink graph never gets torn down: that keeps `/dev/video10` advertised as a CAPTURE device so it never blinks out of conferencing-app device lists. With `exclusive_caps=1`, **PLAYING alone is not enough** — v4l2loopback only flips on `V4L2_CAP_VIDEO_CAPTURE` after `v4l2sink` has called `VIDIOC_STREAMON`, which only happens once at least one buffer has flowed through. So while Idle the feeder still re-pushes the last composited frame (or a black frame on cold start) at `IDLE_PUSH_INTERVAL` (5 Hz). After the first push the kernel's `ready_for_capture` flag is sticky for the lifetime of the producer fd, so the rate could in principle be much lower; 5 Hz is just slow enough to be free and fast enough to dodge any WebRTC consumer's "no frames" timeout.
- **Pipeline starts at app launch, regardless of mode.** The sink graph is what makes `/dev/video10` visible to Meet/Chrome/OBS, so it must run from the moment the app starts — both `--headless` autostart and the GUI launch invoke `Pipeline::start` immediately. The lazy state machine still keeps `/dev/video0` (the LED) released until a real consumer reads. There is no "Start broadcasting" gesture for the sink anymore; opening the GUI preview pane is the only way to force *source* camera engagement without a real consumer attached.
- **`v4l2sink async=false`** — without it the pipeline deadlocks on the PLAYING transition: v4l2sink waits for a preroll buffer that the lazy feeder may never push (we go to PLAYING immediately, before any consumer arrives).
- **`v4l2src do-timestamp=true`** — without it the output stream's PTS is wrong and v4l2sink's pacing slips.
- **Caps strategy:** the source-side capsfilter pins **only RGBA + width/height** (no framerate). Forcing 30/1 caused negotiation failures with the C920 (camera reports `30000/1001`). The appsrc-side caps **do** declare framerate, otherwise the sink-side videoconvert asserts `fps_n == out_fps_n`.

Live setting changes (background mode swap, blur strength, picking a new image, GUI preview toggle, **auto-frame on/off**) flow over the existing `crossbeam-channel` of `Command`s and are applied by the feeder on the next tick — no graph rebuild. Camera, resolution, and **model** changes still require a Stop+Start cycle from the GUI.

#### Lazy mode constants and consumer detection

- **Activation debounce: 2 s** (`lazy::ACTIVATION_DEBOUNCE`). A consumer must remain present for this long before the camera lights. Sized to reject browser capability-probes (Chrome / Firefox open `/dev/video10` briefly for `ENUM_FMT` without intending to stream — typically <500 ms).
- **Deactivation debounce: 3 s** (`lazy::DEACTIVATION_DEBOUNCE`). After the last consumer leaves, we wait this long before releasing `/dev/video0`. Absorbs in-call camera-switcher flicker (the user toggles between cameras in Meet's picker).
- **Watcher poll interval: 800 ms** (`pipeline::WATCH_POLL_INTERVAL`). Fast enough that a real consumer is observed inside the activation debounce window.
- **Consumer detection mechanism: walk `/proc/*/fd/*`** in `consumer_watch::current_consumers`. Follows each fd symlink and counts those targeting `/dev/video10`, excluding our own PID. v4l2loopback does not expose a sysfs consumer-count attribute and the kernel does not fire fsnotify events on character-device opens, so userspace polling is the only portable signal. Cost: ~1–3 ms per poll.
- The GUI's preview pane acts as a **synthetic consumer**: while it's visible (window not minimised AND the *Show preview* toggle is on), the camera stays lit even with no real consumer attached. Hiding the window or turning off the preview toggle drops that signal and the camera goes back to sleep on the deactivation debounce. There is intentionally no "GUI window is open ⇒ camera on" coupling beyond this — anything stronger would defeat the NVIDIA-Broadcast-style "camera only runs when actually being used" promise the lazy mode exists to deliver.

### Cross-cutting conventions

- **Pixel format inside the pipeline is RGBA8** end-to-end. Conversion to/from camera-native (YUYV/MJPEG) and to YUY2 for the v4l2sink happens in GStreamer's `videoconvert` elements; Rust code never touches non-RGBA pixels.
- **The mask is `f32` everywhere** between segmentation and composite. `u8` round-trips cost quality without saving memory.
- **Inference runs at the model's native 256×256.** Only the upsample + composite step touches frame-resolution pixels. This is what keeps CPU usage low at 720p/1080p.
- **`Background::None` is a true passthrough.** The new_sample callback short-circuits *before* segmentation and resets the EMA smoother so the next non-None frame starts clean.
- **No CUDA, no torch, no OpenCV.** ORT runs on the CPU EP only; resizing is `fast_image_resize`; blur is a hand-rolled separable box-blur. `ort` pulls `libonnxruntime.so` automatically via its `download-binaries` feature.
- **Reusable buffers in `Segmenter` and `Compositor`** — both keep their working buffers in `self` to avoid per-frame allocations.

### Models

Two ONNX files, both bundled at compile time via `include_bytes!` in `crates/app/src/main.rs`. RVM is the default — it's what blur radius and auto-frame are tuned against; multiclass is kept as a low-CPU fallback.

- **`models/rvm.onnx`** (~15 MB). Robust Video Matting (MobileNetV3 backbone, fp32) from PeterL1n/RobustVideoMatting v1.0.0. Recurrent video matting: 6 inputs (`src` plus 4 recurrent state tensors `r1i…r4i` plus a `downsample_ratio` scalar), 6 outputs (`fgr`, `pha`, and the 4 next-frame states `r1o…r4o`). We use only `pha` as the mask; `fgr` is discarded. The output mask is at the *frame* resolution — the compositor's `prepare_mask` skips upsampling when the mask already matches the frame. Internal compute is scaled by `RVM_DOWNSAMPLE_RATIO` (currently 0.5 — roughly 640×360 internal on a 720p frame, ~40–60 ms/frame on one x86 core). The recurrent state is held on the segmenter and reset by `Segmenter::reset()` whenever the user switches to `Background::None` or changes input dimensions, so re-engaging starts clean.
- **`models/selfie_multiclass.onnx`** (~16 MB). Converted locally from MediaPipe's `selfie_multiclass_256x256.tflite` via `tf2onnx` (one-time, executed via `uvx`/`uv run`; no permanent Python deps). Input `[1, 256, 256, 3]` NHWC, output `[1, 256, 256, 6]` NHWC of raw logits. Six classes: `0` background, `1` hair, `2` body-skin, `3` face-skin, `4` clothes, `5` others. Foreground = `1 - softmax(logits)[0]` per pixel. Per-frame and fixed at 256×256, roughly 2–3× cheaper than RVM at 720p — the low-CPU fallback. Visible mask flicker around hair / glasses / low-contrast edges; blur and auto-frame are tuned for RVM, not for this.

`Mask` is a public type carrying `(data, width, height)` so each model declares its native mask resolution. Multiclass emits 256×256, RVM emits frame-resolution. The Compositor handles either.

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
- `segmenter.rs` — `Segmenter` enum dispatching to a multiclass or RVM `ort::Session`. Multiclass runs at a fixed 256×256 NHWC with softmax post; RVM holds 4 recurrent state tensors across calls and exposes `reset()` for clean re-engagement.
- `compositor.rs` — bilinear mask upsample, two-pass separable box blur (radius 4–32 px from `Background::Blur { strength }`), plain alpha composite via `blend()`.
- `temporal.rs` — `MaskSmoother` (EMA across frames).
- `pipeline.rs` — public `Pipeline` facade + `PipelineConfig`, `Command`, `PipelineState`. Builds the always-on sink graph at `start()`, spawns the consumer watcher and the feeder, returns. `build_source_pipeline` / `build_sink_pipeline` are factored helpers used by the feeder on every Live engagement.
- `lazy.rs` — feeder thread + state machine + debounce timers. Owns `Segmenter` / `Compositor` / `MaskSmoother` / `AnchorLock` and the source pipeline handle (only `Some` while Live). Drives the Idle ↔ Activating ↔ Live ↔ Deactivating transitions and the per-frame segment + (optional auto-frame) + composite + push.
- `framing.rs` — auto-framing math. Computes a `Framing` (foreground source-anchor + zoom) from the silhouette mask: mass-weighted horizontal centroid for `cx`, *top-edge row* for `cy` (centering on the vertical centroid would crop heads when zoomed). The `AnchorLock` snaps to the silhouette on the first detection and holds that anchor for the rest of the session — toggling auto-frame off+on (or a Live exit) resets the lock so the next frame can re-anchor. Returns `None` only when the lock is not yet engaged AND no foreground is detected; once locked, the lock survives later frames where the segmenter loses the silhouette. Foreground zoom is a static `FG_ZOOM` (no UI control). The compositor consumes the `Framing` to remap foreground sample points only — background plane stays fixed, and the `mask = 0` strip vacated on the trailing edge is filled by the existing blend.
- `consumer_watch.rs` — background `/proc/*/fd` poller. Public `Consumer { pid, name }` (re-exported as `lb_pipeline::Consumer`). Emits a fresh `Vec<Consumer>` on each set change; the feeder folds these into its demand signal.
- `lib.rs` — `MODEL_W`/`MODEL_H` constants and the public re-exports (`Pipeline`, `PipelineConfig`, `PipelineState`, `Consumer`, `Command`, `Background`, …).

App (`crates/app/src/`):
- `main.rs` — entry point; embeds the ONNX models and dispatches to either the icon dumper or `ui::run(headless)`. There is **no separate headless code path** — `--headless` (or `LB_HEADLESS=1`) just tells `ui::run` to start with the window hidden in the tray and auto-start the pipeline. Same UX as before for the autostart case (no window flash, broadcasting on login), single code path to maintain.
- `theme.rs` — design tokens (colors, spacing, radii, control sizes) applied to `egui::Style`/`Visuals`. Inter and JetBrains Mono TTFs registered via `FontDefinitions`.
- `ui.rs` — `eframe::App` + sidebar / preview / footer layout, segmented mode control, blur-intensity slider, library grid, model picker, **Settings** section (`Start on login`, `Show preview`, `Auto-frame`). `App::update` sends a `SetGuiPreviewActive` heartbeat (edge-triggered) so the preview pane counts as a synthetic consumer when visible — and explicitly clears it when the window is hidden in the tray, so a tray-only instance lets the camera drop to Idle on the lazy path. The footer renders `PipelineState` (`Idle` / `Standby (no consumer)` / `LIVE → name(pid)`). The window's close button is intercepted (`ViewportCommand::CancelClose` + `Visible(false)`) — only the tray's Quit menu sets `quit_requested` and lets the close through. In headless mode, `App::new` polls for `/dev/video10` (10 s timeout) before auto-starting the pipeline so it survives the autostart-vs-`systemd-modules-load.service` race on cold boot.
- `tray.rs` — system tray icon and menu (Show / Hide / Quit). Spawns a dedicated `lb-tray-gtk` thread that calls `gtk::init()` + `gtk::main()` because tray-icon's Linux backend (libayatana-appindicator) needs a GTK loop on *some* thread, and egui/winit don't host one. `MenuEvent`s are bridged into a crossbeam channel that `App::update` drains every frame. Install can fail on systems without a tray host — the failure is logged and the GUI keeps working without a tray entry.
- `cameras.rs` — enumerate `/dev/video*` and probe friendly names from `/sys/class/video4linux/<n>/name`.
- `backgrounds.rs` — saved-image library at `~/.local/share/linux-broadcast/backgrounds/`.
- `config.rs` — TOML config at `~/.config/linux-broadcast/config.toml` (`Mode`, `Model`, blur strength, source/sink, `start_on_login`, `show_preview`, `auto_frame`).
- `icon.rs` — programmatically rasterizes the logo SVG to a 64×64 RGBA `IconData` (no `usvg` dep — the logo is four primitives). Same code path produces `packaging/LinuxBroadcast.png` via `LB_DUMP_ICON=1`.
- `desktop_install.rs` — drops `~/.local/share/icons/hicolor/64x64/apps/LinuxBroadcast.png` and a matching `LinuxBroadcast.desktop` on first launch so Wayland compositors (KDE / GNOME) can resolve the window's `app_id` to a real taskbar icon. **Skipped** when the `.deb`-installed system entry at `/usr/share/applications/LinuxBroadcast.desktop` is present, to avoid duplicate menu entries.
- `autostart.rs` — install / uninstall / reconcile the opt-in `~/.config/autostart/LinuxBroadcast-autostart.desktop` (runs `<exec> --headless`). Driven by the sidebar toggle; reconciled on every GUI launch against the saved `start_on_login` flag.
- `lock.rs` — per-user `flock` at `$XDG_RUNTIME_DIR/linux-broadcast.lock` (config-dir fallback). **Acquired at process scope in `main.rs`** and held for the lifetime of the LB process — necessary because lazy-mode instances can sit Idle (no `/dev/video10` write contention) for arbitrarily long, so we'd otherwise allow two instances to coexist and race the moment a consumer arrived. A second LB launch finds the lock held and exits cleanly.

Design assets:
- `assets/fonts/` — Inter-Variable + JetBrainsMono-Regular.
- `DESIGN.md` — colour / spacing / type tokens + design rationale (intent doc; `theme.rs` is authoritative for values).

### Packaging (`packaging/` + `[package.metadata.deb]`)

`cargo deb -p linux-broadcast` reads `crates/app/Cargo.toml` and ships:

| Asset | Installed to | Notes |
|---|---|---|
| `target/release/linux-broadcast` | `/usr/bin/linux-broadcast` | The binary. |
| `packaging/LinuxBroadcast.desktop` | `/usr/share/applications/` | System launcher; `desktop_install.rs` skips its per-user clone when this exists. |
| `packaging/LinuxBroadcast.png` | `/usr/share/icons/hicolor/64x64/apps/` | Pre-rendered via `LB_DUMP_ICON=1` so `cargo deb` doesn't have to execute the binary at packaging time. Regenerate when the logo changes. |
| `packaging/linux-broadcast.modprobe.conf` | `/etc/modprobe.d/linux-broadcast.conf` | **conffile** — `options v4l2loopback devices=1 video_nr=10 card_label="LinuxBroadcast" exclusive_caps=1 max_buffers=2`. |
| `packaging/linux-broadcast.modules-load.conf` | `/etc/modules-load.d/linux-broadcast.conf` | **conffile** — single line `v4l2loopback`, makes the module reload on every boot. |

Maintainer scripts at `packaging/scripts/`:
- `postinst` — drops a stale module if loaded with different params, then `modprobe v4l2loopback` (options come from the modprobe.d drop-in, no need to pass them on the command line). Refreshes `update-desktop-database` and `gtk-update-icon-cache` so the menu entry shows up without logout. Module-load failure is logged but does **not** fail the install: DKMS may still be building, and the `modules-load.d` file guarantees the next boot loads it.
- `prerm` — best-effort `modprobe -r v4l2loopback` on uninstall. Failure is harmless (something else may be holding the device).
- `postrm` — refresh desktop / icon caches after the system files are gone.

Conffiles mean `apt purge` removes the `/etc/modprobe.d/` and `/etc/modules-load.d/` drop-ins; `apt remove` keeps them for a future re-install.

The release artefact lives at `target/debian/linux-broadcast_<ver>_amd64.deb`. CI release-on-tag is the next phase; for now we build locally and attach to GitHub Releases by hand.

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

### Tests

`cargo test -p lb_pipeline` covers the headless math and graph plumbing:

- `tests/models_smoke.rs` — loads each bundled ONNX through `Segmenter::from_bytes`, runs 1–2 inferences on a synthetic frame, and asserts the mask shape + value range match the per-model contract (256×256 for the MediaPipe variants, frame-size for RVM; RVM also covers `reset()` clearing recurrent state). Catches model/pre-post regressions without needing a real camera; runs by default on every `cargo test`.
- `tests/synthetic_graph.rs` — drives a fake `videotestsrc → … → appsink` source through the `Compositor` and back into the sink graph, verifying caps negotiation and the appsrc PTS pacing without touching `/dev/video0` or `/dev/video10`. Useful when refactoring `pipeline.rs` / `lazy.rs`.

The GUI crate has no tests — its surface is mostly egui layout, exercised by hand. Don't add UI snapshot tests without a strong reason; egui rendering is too version-sensitive to be worth the maintenance.

## Roadmap

- CPU/GPU usage in footer.
- ~~`.deb` packaging~~ ✅ — `cargo-deb` metadata in `crates/app/Cargo.toml`, conffiles in `packaging/`, postinst handles `modprobe` + cache refresh, opt-in autostart toggle in the GUI. Flatpak is **not pursued** — sandbox can't `modprobe` and the wizard pattern (`flatpak-spawn --host pkexec modprobe …`) is too fragile to recommend; users who can't install the `.deb` should build from source.
- ~~Auto-framing~~ ✅ — `framing.rs` + `AnchorLock`, opt-in via the *Auto-frame* setting. Currently a snap-on-first lock with a static zoom + horizontal recenter; could grow into a fuller PTZ if there's demand.
- GitHub Actions release-on-tag workflow that uploads the `.deb` to the release page (`cargo deb` already produces a reproducible artefact locally).
- Throughput benchmarks per model on a stable reference machine, published in the repo so contributors can spot regressions on a model swap.

