# Linux Broadcast

A small, NVIDIA Broadcast-style virtual webcam for Linux. Blurs your background or replaces it with an image, and exposes the result as a regular webcam that Zoom, Meet, Teams, OBS and Firefox just pick up.

- **No CUDA, no PyTorch, no Python.** Single Rust binary.
- Runs MediaPipe Selfie Segmentation on CPU — ~5 ms per frame at 256×144 inference.
- ~25 MB binary, no system Python or Qt runtime to install.

## Status

Early. Phase 1 (headless vertical slice) is scaffolded but not yet smoke-tested end-to-end. The previous Python prototype is preserved on the [`legacy-python`](https://github.com/Pedrojok01/linux-broadcast/tree/legacy-python) branch.

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
sudo modprobe v4l2loopback video_nr=10 card_label="Linux Broadcast" exclusive_caps=1 max_buffers=2

# 3. Drop the model in place (one-time; ~450 KB)
mkdir -p models
curl -L -o models/selfie_segmenter.onnx \
  https://huggingface.co/onnx-community/mediapipe_selfie_segmentation/resolve/main/onnx/model.onnx

# 4. Run
cargo run --release
# In another terminal, verify:
cheese -d /dev/video10
```

> **Kernel 6.8+ note:** if `apt install v4l2loopback-dkms` fails to build, you have version 0.12.7 — install 0.12.8+ from upstream or your distro backports.

## Where the quality comes from

| Lever | What it does |
|---|---|
| MediaPipe Selfie Segmentation | Per-pixel matting at 256×256, designed for exactly this use case. |
| EMA mask smoothing across frames | Removes ~80% of flicker without needing a video-recurrent model. |
| Inference at native model resolution | Only the upsample + composite touches full-res pixels — keeps CPU low at 720p/1080p. |
| `tract` pure-Rust ONNX runtime | Zero native deps → tiny binary, simple packaging, no `libonnxruntime.so`. |

## License

MIT.
