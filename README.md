<div align="center">

<img src="./packaging/LinuxBroadcast.png" width="96" height="96" alt="LinuxBroadcast logo" />

<br><br>

<h1><strong>LinuxBroadcast</strong></h1>

<p>
  <em>Background replacement for your webcam, on Linux, in a single Rust binary.</em>
</p>

<p align="center">
  <a href="https://github.com/Pedrojok01/linux-broadcast/actions/workflows/ci.yml"><img src="https://github.com/Pedrojok01/linux-broadcast/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg" alt="License: GPL-3.0-or-later"></a>
  <img src="https://img.shields.io/badge/MSRV-1.88-orange.svg" alt="MSRV 1.88">
  <img src="https://img.shields.io/badge/platform-linux--x86__64-lightgrey" alt="Platform: Linux x86_64">
</p>

</div>

A small virtual webcam for Linux that segments the foreground with MediaPipe / RVM, blurs or replaces the background, and exposes the result on a `v4l2loopback` device that Zoom, Meet, Teams, OBS, Firefox, and Chrome treat as a regular webcam.

```mermaid
flowchart LR
    cam(["📷 Physical cam<br/><sub>/dev/video0</sub>"])
    seg["🧠 <b>Segment</b><br/><sub>MediaPipe · RVM<br/>ONNX Runtime</sub>"]
    comp["🎨 <b>Composite</b><br/><sub>blur · replace<br/>auto-frame</sub>"]
    vcam(["📺 Virtual cam<br/><sub>/dev/video10</sub>"])

    cam == RGBA ==> seg
    seg == mask ==> comp
    comp == YUY2 ==> vcam

    classDef device fill:#0f172a,stroke:#60a5fa,stroke-width:2px,color:#f8fafc;
    classDef rust fill:#7c2d12,stroke:#fb923c,stroke-width:2px,color:#fff7ed;
    class cam,vcam device;
    class seg,comp rust;
```

**Why it exists.** Existing options on Linux are either heavy (Python + CUDA + OpenCV stacks) or shallow (basic chroma key with hard cuts). This is a single Rust binary that runs MediaPipe / RVM on CPU via ONNX Runtime, with a native `egui` UI, no Python, no CUDA, and edge quality on par with the leading proprietary alternatives.

## Contents

- [Features](#features)
- [Install & run](#install--run)
  - [Option A — `.deb` (recommended)](#option-a--deb-recommended)
  - [Option B — Build from source](#option-b--build-from-source)
  - [Start on login](#start-on-login)
  - [Lazy mode (camera on demand)](#lazy-mode-camera-on-demand)
- [Using it](#using-it)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Repo layout](#repo-layout)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Two switchable models** (chosen live in the GUI):
  - **RVM** (Robust Video Matting) — recurrent video matting, best edges, no flicker (~15 MB). Default.
  - **Selfie multiclass** — per-frame, fixed 256×256 (~16 MB). Low-CPU fallback.
- **Three background modes** — passthrough, blur (intensity slider, 4–32 px radius), replace with a saved image.
- **Saved background library** — imports are copied to `~/.local/share/linux-broadcast/backgrounds/` so they survive across launches.
- **Auto-frame** — a smoothed horizontal recenter plus a light foreground zoom that keeps the silhouette centered as you move. Off by default; toggle in the sidebar's *Settings*. Skipped when the background mode is *None* (no plane to paint over).
- **Lazy producer mode** — the physical webcam LED only lights when an app is actually reading the virtual cam, like NVIDIA Broadcast. See [Lazy mode](#lazy-mode-camera-on-demand).
- **System tray** — the close button hides to the tray; `Quit` from the tray menu actually exits.
- **Live preview pane** in the GUI; settings persist to `~/.config/linux-broadcast/config.toml`.
- **No CUDA, no PyTorch, no Python** — single Rust binary, ~25 MB plus the bundled ONNX/font assets.

Out of scope: audio / microphone effects.

## Install & run

### Option A — `.deb` (recommended)

Tested on Ubuntu 24.04+ / Mint 22+ / Debian trixie+. The package depends on `v4l2loopback-dkms (≥ 0.12.8)` and the GStreamer plugin set; everything else is statically baked into the binary.

```bash
sudo apt install ./linux-broadcast_<version>_amd64.deb
```

That's it. Behind the scenes the package:

- Installs the kernel module via DKMS and loads it now (`postinst` runs `modprobe v4l2loopback`), so `/dev/video10` is available immediately.
- Drops `/etc/modprobe.d/linux-broadcast.conf` with the right options (`devices=1 video_nr=10 card_label="LinuxBroadcast" exclusive_caps=1 max_buffers=2`) and `/etc/modules-load.d/linux-broadcast.conf` so the module reloads on every boot — no manual `modprobe` ever required.
- Registers the app menu entry and icon under `/usr/share/applications/` and `/usr/share/icons/hicolor/64x64/apps/`.

`apt remove linux-broadcast` unloads the module and removes the launcher; `apt purge` additionally drops the `/etc/modprobe.d` and `/etc/modules-load.d` drop-ins (preserved as conffiles otherwise).

Want it always running so Zoom / Meet / Signal / Firefox just see "LinuxBroadcast" in their camera list at every login? Open the GUI and flip **Start on login** in the sidebar — see [Start on login](#start-on-login) below.

### Option B — Build from source

Use this when hacking on the code; the `.deb` is the right choice for everyday use.

#### 1. System dependencies

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

The `libgtk-3-dev` / `libxdo-dev` / `libayatana-appindicator3-dev` trio is pulled in by the `tray-icon` crate at build time; at runtime only `libayatana-appindicator3-1` is required (already declared in the `.deb`'s `Depends`).

#### 2. Create the virtual camera device (development only)

The `.deb` does this automatically; for source builds, set the module up by hand once:

```bash
# Reload the module so our params actually take effect — modprobe is a
# no-op if the module is already loaded with different options.
sudo modprobe -r v4l2loopback 2>/dev/null
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="LinuxBroadcast" \
  exclusive_caps=1 max_buffers=2
```

To make this survive reboots, drop the same options into `/etc/modprobe.d/linux-broadcast.conf` and `linux-broadcast` into `/etc/modules-load.d/` (or just install the `.deb`, which ships these files for you).

#### 3. Build & run

```bash
git clone https://github.com/Pedrojok01/linux-broadcast.git
cd linux-broadcast
cargo run --release -p linux-broadcast
```

ONNX Runtime's `libonnxruntime.so` is fetched automatically the first time you build (`ort` crate, `download-binaries` feature). The MediaPipe / RVM models and the Inter / JetBrains Mono fonts ship in-tree.

A headless mode is available for sanity checks and CI:

```bash
cargo run --release -p linux-broadcast -- --headless
# or, equivalently:
LB_HEADLESS=1 cargo run --release -p linux-broadcast
```

#### 4. Build a `.deb` locally

```bash
cargo install cargo-deb
cargo deb -p linux-broadcast
# artefact: target/debian/linux-broadcast_<version>_amd64.deb
```

### Start on login

LinuxBroadcast can run silently in the background at every login so the virtual cam is already up by the time you open Zoom / Meet / Signal / Firefox. It's **off by default** — flip the toggle in the sidebar's *Settings* section to enable it. The GUI writes (or removes) `~/.config/autostart/LinuxBroadcast-autostart.desktop`, which runs `linux-broadcast --headless` on login. No system files are touched and no root is needed; uninstalling the `.deb` won't remove your autostart entry, and disabling the toggle will.

### Lazy mode (camera on demand)

By default LinuxBroadcast only opens your physical camera when **something is actually reading the virtual cam** — Meet / Zoom / Signal / Firefox / `ffplay`. The rest of the time the LED is off and CPU is at idle, even with the GUI open. The transition is fast: opening Meet lights the camera within ~2 s; closing it releases the camera ~3 s later. Both windows debounce browser capability-probes and in-call camera-switcher flicker.

The footer shows what's happening: `● Idle` (no app reading), `● Standby (no consumer)` (LB is up but nothing wants the cam yet), or `● LIVE → firefox (12345)` while a real consumer is attached.

When the LinuxBroadcast GUI is open and visible, the **preview pane counts as a consumer** so you can see your composited self while configuring backgrounds. Minimising the window drops that signal and the camera goes back to sleep (assuming nothing else is reading).

## Using it

The pipeline is **always running** while LinuxBroadcast is open — there's no Start/Stop button. Conferencing apps see `LinuxBroadcast` in their camera list the moment LB starts, and the physical webcam only powers on when something actually reads the virtual cam (see [Lazy mode](#lazy-mode-camera-on-demand)).

1. Pick a physical camera in **Camera**.
2. Pick a model in **Model** — RVM is the default (best edges); multiclass is the low-CPU fallback. Switching restarts the pipeline automatically.
3. **Set the scene** with the segmented control — `None` (passthrough), `Blur` (slider for intensity), or `Replace` (uses the active library tile).
4. Click **+ Import** to add background images; click any thumbnail to switch to it live; right-click → Remove deletes it.
5. Toggle **Auto-frame**, **Show preview**, or **Start on login** in the sidebar's *Settings* section as needed. All three persist to `config.toml`.
6. Open Zoom / Meet / Signal / Firefox / OBS and pick `LinuxBroadcast` as the camera — that's it. Closing LB's window keeps it running in the tray; `Quit` from the tray menu actually shuts it down.

## Performance

Reference numbers on a **Logitech C920 + single x86 core, 1280×720**:

| Model | Inference / frame | Throughput |
|---|---|---|
| Selfie multiclass | ~10 ms | 30 fps (camera-bound) |
| RVM (`downsample_ratio=0.5`) | ~40–60 ms | ~15 fps |

Multiclass leaves plenty of headroom for 1080p; RVM at 1080p needs `downsample_ratio=0.25` (set in `crates/pipeline/src/segmenter.rs`).

## Troubleshooting

- **`/dev/video10` doesn't appear.** With the `.deb`, this is handled automatically: the postinst does `modprobe -r` first to drop any stale module, then reloads it with the right params. For source builds, run those two commands by hand (see [step 2 above](#2-create-the-virtual-camera-device-development-only)).
- **`/dev/video10` is "busy" or "not a video capture device".** That's `exclusive_caps=1` doing its job: the device only exposes CAPTURE while LinuxBroadcast is producing frames. Real apps see it; raw `ffplay` may not until the producer is running.
- **`apt install v4l2loopback-dkms` fails on kernel 6.8+.** You have the broken 0.12.7 — install ≥ 0.12.8 from upstream or your distro backports.
- **The window icon shows in the title bar but the taskbar entry stays generic on Wayland.** First launch installs `~/.local/share/icons/.../LinuxBroadcast.png` and a matching `.desktop` file; KDE may need `kbuildsycoca6 --noincremental` once or a re-login to refresh its sycoca cache.

## Repo layout

```
crates/
  pipeline/      # GStreamer graph + ort segmenter + compositor (no GUI deps)
  app/           # eframe/egui GUI, theme, config, background library
models/          # bundled ONNX (multiclass / RVM)
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

GPL-3.0-or-later. See [`LICENSE`](LICENSE) for the full text.

This project bundles [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) (`models/rvm.onnx`), released under GPL-3.0, which is what makes the entire binary GPL-3.0. The MediaPipe ONNX file in `models/` is Apache-2.0; the bundled fonts in `assets/fonts/` are SIL Open Font License 1.1. See [`models/README.md`](models/README.md) and [`assets/fonts/`](assets/fonts/) for per-asset attribution.
