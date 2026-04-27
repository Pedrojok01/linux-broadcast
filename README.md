# LinuxBroadcast

A small, NVIDIA-Broadcast-style virtual webcam for Linux. Captures your camera, segments the foreground with MediaPipe / RVM, blurs or replaces the background, and exposes the result on a `v4l2loopback` device that Zoom, Meet, Teams, OBS, Firefox, and Chrome treat as a regular webcam.

```
  v4l2src           appsink                appsrc           v4l2sink
 ┌────────┐  RGBA  ┌────────┐  segment +  ┌────────┐  YUY2  ┌─────────────┐
 │/dev/v0 ├───────►│ Rust   ├────────────►│ Rust   ├───────►│ /dev/video10│
 └────────┘        │  ML    │  composite  │ push   │        └─────────────┘
                   └────────┘             └────────┘
```

**Why it exists.** Existing options on Linux are either heavy (Python + CUDA + OpenCV stacks) or shallow (basic chroma key with hard cuts). This is a single Rust binary that runs MediaPipe / RVM on CPU via ONNX Runtime, with a native `egui` UI, no Python, no CUDA, and edge quality close to NVIDIA Broadcast's.

## Features

- **Three switchable models** (chosen live in the GUI):
  - **Selfie binary** — fast (~5 ms inference, 450 KB).
  - **Selfie multiclass** — six-class output, sharper edges (~16 MB).
  - **RVM** (Robust Video Matting) — recurrent video matting, best edges, no flicker (~15 MB).
- **Three background modes** — passthrough, blur (intensity slider, 4–32 px radius), replace with a saved image.
- **Saved background library** — imports are copied to `~/.local/share/linux-broadcast/backgrounds/` so they survive across launches.
- **Live preview pane** in the GUI; settings persist to `~/.config/linux-broadcast/config.toml`.
- **No CUDA, no PyTorch, no Python** — single Rust binary, ~25 MB plus the bundled ONNX/font assets.

Out of scope: audio / microphone effects.

## Install & run

### 1. System dependencies

Tested on Ubuntu 24.04+ / Mint 22+ / Debian trixie+. The build needs GStreamer + a few X11/Wayland headers; the runtime additionally needs the `v4l2loopback` kernel module and the GStreamer plugin packages.

```bash
sudo apt install -y \
  build-essential pkg-config \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libxkbcommon-dev libwayland-dev \
  libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev \
  v4l2loopback-dkms \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
```

### 2. Create the virtual camera device

```bash
# Reload the module so our params actually take effect — modprobe is a
# no-op if the module is already loaded with different options.
sudo modprobe -r v4l2loopback 2>/dev/null
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="LinuxBroadcast" \
  exclusive_caps=1 max_buffers=2
```

To make this survive reboots, drop the same options into `/etc/modprobe.d/linux-broadcast.conf` and `linux-broadcast` into `/etc/modules-load.d/`.

### 3. Build & run

```bash
git clone https://github.com/Pedrojok01/linux-broadcast.git
cd linux-broadcast
cargo run --release -p linux-broadcast
```

ONNX Runtime's `libonnxruntime.so` is fetched automatically the first time you build (`ort` crate, `download-binaries` feature). The MediaPipe / RVM models and the Inter / JetBrains Mono fonts ship in-tree.

A headless mode is available for sanity checks and CI:

```bash
LB_HEADLESS=1 cargo run --release -p linux-broadcast
```

## Using it

1. Pick a physical camera in **Camera**.
2. Pick a model in **Model** — binary is fastest, multiclass sharper, RVM cleanest. Switching restarts the pipeline automatically.
3. **Set the scene** with the segmented control — `None` (passthrough), `Blur` (slider for intensity), or `Replace` (uses the active library tile).
4. Click **+ Import** to add background images; click any thumbnail to switch to it live; right-click → Remove deletes it.
5. Click **Start broadcasting**. Any conferencing app that picks `LinuxBroadcast` as its camera now sees the composited stream.

## Performance

Reference numbers on a **Logitech C920 + single x86 core, 1280×720**:

| Model | Inference / frame | Throughput |
|---|---|---|
| Selfie binary | ~5 ms | 30 fps (camera-bound) |
| Selfie multiclass | ~10 ms | 30 fps (camera-bound) |
| RVM (`downsample_ratio=0.4`) | ~30–40 ms | ~15 fps |

The MediaPipe variants leave plenty of headroom for 1080p; RVM at 1080p needs `downsample_ratio=0.25` (set in `crates/pipeline/src/segmenter.rs`).

## Troubleshooting

- **`/dev/video10` doesn't appear.** Stale module load — run `sudo modprobe -r v4l2loopback` first, then re-`modprobe` with the params above.
- **`/dev/video10` is "busy" or "not a video capture device".** That's `exclusive_caps=1` doing its job: the device only exposes CAPTURE while LinuxBroadcast is producing frames. Real apps see it; raw `ffplay` may not until the producer is running.
- **`apt install v4l2loopback-dkms` fails on kernel 6.8+.** You have the broken 0.12.7 — install ≥ 0.12.8 from upstream or your distro backports.
- **The window icon shows in the title bar but the taskbar entry stays generic on Wayland.** First launch installs `~/.local/share/icons/.../io.Pedrojok01.LinuxBroadcast.png` and a matching `.desktop` file; KDE may need `kbuildsycoca6 --noincremental` once or a re-login to refresh its sycoca cache.

## Repo layout

```
crates/
  pipeline/      # GStreamer graph + ort segmenter + compositor (no GUI deps)
  app/           # eframe/egui GUI, theme, config, background library
models/          # bundled ONNX (binary / multiclass / RVM)
assets/fonts/    # Inter + JetBrains Mono
DESIGN.md        # design tokens (colour, spacing, type) + rationale
CLAUDE.md        # in-depth dev notes (pipeline, gotchas, model details)
```

`CLAUDE.md` is the longer engineering map — pipeline plumbing, GStreamer settings that took a session to nail, model conventions, where to put new modes / models. Read it before non-trivial changes.

## Contributing

```bash
# Format
cargo fmt --all

# Lint (CI runs this with -D warnings)
cargo clippy --workspace --all-targets -- -D warnings

# Build
cargo build --workspace

# Headless smoke (uses the saved config; needs /dev/video10 + a real camera)
LB_HEADLESS=1 cargo run --release -p linux-broadcast
```

The toolchain is pinned via [`rust-toolchain.toml`](rust-toolchain.toml) (current stable + `rustfmt` + `clippy`); rustfmt config lives in [`rustfmt.toml`](rustfmt.toml). CI in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs `fmt --check`, `clippy -D warnings`, and a release build on every push and PR.

For substantive changes:
- Stay clippy-clean. CI fails on warnings.
- Keep all pixel data in RGBA8; the mask in `f32`. The pipeline assumes both end-to-end.
- New segmentation models: add a `ModelKind` variant + `segment_*` function in `crates/pipeline/src/segmenter.rs`, bundle the ONNX, surface it in the GUI's `Model` dropdown. CLAUDE.md has a step-by-step.
- New background modes: add a `Background` variant + branch in the compositor, plus a `Mode` in `app/src/config.rs` and a tab in `ui::sidebar_scene`. Live changes flow over the existing `Command::SetBackground` channel — no graph rebuild.

## License

MIT.
