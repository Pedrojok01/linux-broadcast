# Linux Broadcast

A small, background-replacement virtual webcam for Linux. Captures your camera, segments the foreground with MediaPipe, blurs or replaces the background, and exposes the result as a regular webcam that Zoom, Meet, Teams, OBS and Firefox just pick up.

- **No CUDA, no PyTorch, no Python.** Single Rust binary.
- Two MediaPipe models bundled, switchable at runtime in the GUI:
  - **Selfie binary** — fast (~450 KB, ~5 ms inference).
  - **Selfie multiclass** — sharper edges (~16 MB, six-class output).
- Holds 30 fps at 1280×720 on a Logitech C920 with a single x86 core.
- Native `egui` UI: live preview pane, blur-intensity slider, saved background-image library, model picker.

## Status

Phase 2 done — fully usable end-to-end. The previous Python prototype is preserved on the [`legacy-python`](https://github.com/Pedrojok01/linux-broadcast/tree/legacy-python) branch.

## Try it from source

```bash
# 1. Build deps
sudo apt install -y \
  build-essential pkg-config \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libxkbcommon-dev libwayland-dev \
  libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev \
  v4l2loopback-dkms \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav

# 2. Virtual camera device (until the .deb postinst exists)
#    NB: a stale module load is the most common cause of /dev/video10 not
#    appearing — `modprobe -r v4l2loopback` first if you change params.
sudo modprobe -r v4l2loopback 2>/dev/null
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Linux Broadcast" \
  exclusive_caps=1 max_buffers=2

# 3. Build & run the GUI
cargo run --release -p linux-broadcast

# 4. Or run headless (no window, hard-loops on the saved config)
LB_HEADLESS=1 cargo run --release -p linux-broadcast

# 5. Quick visual check from another terminal
ffplay -fflags nobuffer -f v4l2 -input_format yuyv422 \
  -video_size 1280x720 /dev/video10
```

The MediaPipe ONNXs (binary + multiclass) and the Inter / JetBrains Mono TTFs ship in-tree, so there's no separate download step. ONNX Runtime's `libonnxruntime.so` is fetched automatically the first time you build (`ort` crate, `download-binaries` feature).

> **Kernel 6.8+ note:** if `apt install v4l2loopback-dkms` fails to build, you have the broken 0.12.7 — install ≥ 0.12.8 from upstream or your distro backports.

> **`/dev/video10` is "busy" or "not a video capture device":** that's `exclusive_caps` doing its job — the device only exposes CAPTURE while linux-broadcast is producing frames. Apps see it correctly; raw `ffplay` may not until the producer is running.

## Using it

1. Run `cargo run --release -p linux-broadcast`.
2. Pick your physical camera in the **Camera** dropdown.
3. Pick a model in the **Model** dropdown — multiclass is sharper, binary is faster. Switching restarts the pipeline automatically.
4. **Set the scene** — pick `None` (passthrough), `Blur` (slider for intensity), or `Replace` (uses the active library tile).
5. Drop background images via the **+ Import** tile in the Library; they're copied to `~/.local/share/linux-broadcast/backgrounds/` so they're available next time. Click any tile to switch live; right-click → Remove deletes it.
6. **Start broadcasting**. The preview pane fills with the composited frame, and any conferencing app picking `Linux Broadcast` as its camera sees the same stream.
7. Settings persist automatically to `~/.config/linux-broadcast/config.toml`.

## Design

`DESIGN.md` documents the colour tokens, spacing scale, and type system. The actual values are hard-coded in `crates/app/src/theme.rs`.

## Architecture (one-line tour)

`v4l2src → videoconvert/scale → appsink → segmenter (binary | multiclass) → EMA smooth → composite → appsrc → videoconvert → v4l2sink → /dev/video10`

The whole pipeline lives in `crates/pipeline`; the GUI is `crates/app`. See `CLAUDE.md` for the deeper map.

## License

MIT.
