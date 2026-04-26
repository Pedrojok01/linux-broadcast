# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NVIDIA Broadcast-style virtual background app for Linux. Captures a webcam frame, runs a portrait-matting model, composites the person over a chosen background (or blur), and pipes the result into a `v4l2loopback` virtual camera that conferencing apps consume. A Windows "preview mode" exists where `VCam` is a no-op.

## Common commands

```bash
# Run the app (downloads any missing models on first launch via main.py)
python main.py

# One-time host setup: load v4l2loopback and create /dev/video10
sudo ./scripts/setup_v4l2loopback.sh

# Download models manually (RVM from GitHub, RMBG-2.0 from gated HF repo)
python scripts/download_model.py

# Tests
python -m unittest discover tests
python -m unittest tests.test_segment.TestSegmenter.test_onnx_segmenter   # single test

# Build standalone binary (output in dist/virtual-background)
pyinstaller virtual-background.spec
```

`scripts/download_model.py` reads `HF_TOKEN` from `.env` (via `python-dotenv`) for the gated `briaai/RMBG-2.0` repo. The MODNet `.pth` / `.onnx` files referenced by the GUI and tests are **not** downloaded by `download_models()` — only RVM and RMBG are. Tests that load MODNet will fail unless those files are placed in `models/` separately.

## Architecture

The pipeline is a single producer/consumer loop owned by `VideoThread` (a `QThread` in `app/gui.py`). It wires four collaborators that are intentionally decoupled and BGR-throughout:

```
Camera.get_frame ─► Segmenter.segment ─► apply_background ─► VCam.write_frame
                                                        └──► Qt signal → preview QLabel
```

- **`app/camera.py`** — thin `cv2.VideoCapture` wrapper. `Camera.list_cameras()` probes indices 0–9 with the platform-appropriate backend (`CAP_V4L2` on Linux, `CAP_MSMF` on Windows).
- **`app/segment.py`** — all four model backends + a `create_segmenter(model_path)` factory. The factory dispatches by **substring** of the path, not just extension: `"rmbg"` → `RMBGSegmenter`, `"rvm"` → `RVMSegmenter`, then `.pth` → `PyTorchSegmenter`, `.onnx` → `ONNXSegmenter`. Renaming a model file in a way that breaks those substrings will silently route it to the wrong class.
  - `RMBGSegmenter` resizes input to a fixed 1024×1024 before inference, then upsamples the matte back.
  - `RVMSegmenter` is stateful — it carries recurrent tensors (`self.rec`) between frames for temporal stability, so a single instance must be reused across the stream (which `VideoThread.run` already does).
  - All segmenters return a single-channel float matte at the original frame's H×W; downstream code assumes that contract.
- **`app/background.py`** — `apply_background(frame, mask, background, blur)` does the alpha composite. The blur path ignores the `background` argument and Gaussian-blurs the original frame instead.
- **`app/vcam.py`** — wraps `pyfakewebcam` (Linux only). Converts BGR→RGB before writing. On `win32`, constructor logs and returns; `write_frame` becomes a no-op so the GUI still shows a preview.
- **`app/gui.py`** — PySide6 `MainWindow`. Model choice is a radio group mapped to filenames at `toggle_video` time:
  - `pytorch` → `models/modnet_webcam.pth`
  - `onnx` → `models/modnet_webcam.onnx`
  - `rvm` → `models/rvm_mobilenetv3_fp16.onnx`
  - `rmbg` → `models/rmbg_2_0_fp16.onnx`
  - `virtual_device` is hardcoded to `/dev/video10` on Linux and `None` on Windows.
- **`app/settings.py`** — persists GUI state to `settings.json` in CWD. Settings are saved on every widget change.

### Cross-cutting conventions

- **BGR everywhere** in the pipeline; only convert to RGB at the boundaries (`VCam.write_frame`, Qt preview, model preprocessing inside each segmenter).
- **CWD-relative paths**. `main.py`, `settings.py`, and `download_model.py` all assume the process is launched from the repo root. Running from elsewhere breaks model loading and settings persistence.
- **CUDA via ONNX Runtime** is opportunistic — segmenters check `onnxruntime.get_available_providers()` and fall back to CPU. `requirements.txt` pins `onnxruntime-gpu==1.18.0` and `nvidia-cudnn-cu12==8.9.*`; mismatched CUDA on the host silently falls back to CPU.

### Adding a new segmentation model

1. Subclass or follow the segmenter contract: `__init__(model_path)` and `segment(bgr_frame) -> H×W float mask`.
2. Add a branch to `create_segmenter` — remember it matches by substring, so pick a unique token.
3. Add a radio button in `MainWindow.__init__`, persist the choice in `save_current_settings`, and map it to a filename in `toggle_video`.
4. If the model needs weights, extend `scripts/download_model.py` (use `download_hf_file` for HF-hosted, `download_file` for direct URLs).
