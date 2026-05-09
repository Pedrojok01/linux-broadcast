# Bundled segmentation models

LinuxBroadcast embeds three ONNX models at compile time via `include_bytes!`
(see `crates/app/src/main.rs`). They ship inside the binary; no first-run
download.

## `selfie_segmenter.onnx`

- **Task:** binary foreground / background segmentation, 256×256 input.
- **Source:** [`onnx-community/mediapipe_selfie_segmentation`](https://huggingface.co/onnx-community/mediapipe_selfie_segmentation) on Hugging Face. Pre-converted to ONNX from MediaPipe's TFLite weights.
- **Upstream:** [Google MediaPipe — Selfie Segmentation](https://developers.google.com/mediapipe/solutions/vision/image_segmenter).
- **License:** Apache License 2.0 — see [`Apache-2.0-LICENSE.txt`](Apache-2.0-LICENSE.txt).
- **SHA-256:** `3241ac4ad8aa35bdaf33946776db29f7c283a413aa0b0dacb9483594b4531aad`

## `selfie_multiclass.onnx`

- **Task:** 6-class semantic segmentation (background / hair / body-skin / face-skin / clothes / others), 256×256 input.
- **Source:** locally converted from `selfie_multiclass_256x256.tflite` via `tf2onnx` (see CLAUDE.md for the exact `uvx` recipe).
- **Upstream:** [Google MediaPipe — Image Segmenter (multiclass selfie)](https://developers.google.com/mediapipe/solutions/vision/image_segmenter).
- **License:** Apache License 2.0 — see [`Apache-2.0-LICENSE.txt`](Apache-2.0-LICENSE.txt).
- **SHA-256:** `534fd56e6c826e810a0e65d2e9a1d7bea4566c71ca354f3b2663e3a470acd738`

## `rvm.onnx`

- **Task:** Robust Video Matting (recurrent), MobileNetV3 backbone, fp32. Frame-resolution alpha matte output.
- **Source:** [`PeterL1n/RobustVideoMatting` v1.0.0 release — `rvm_mobilenetv3_fp32.onnx`](https://github.com/PeterL1n/RobustVideoMatting/releases/tag/v1.0.0).
- **License:** **GNU General Public License v3.0** — see [`RVM-LICENSE.txt`](RVM-LICENSE.txt). Bundling this file is what makes the LinuxBroadcast binary as a whole GPL-3.0; see the project's top-level [`LICENSE`](../LICENSE).
- **SHA-256:** `88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828`

## Re-fetching

```bash
# selfie_segmenter — pull the pre-converted ONNX from HF.
curl -L -o models/selfie_segmenter.onnx \
  https://huggingface.co/onnx-community/mediapipe_selfie_segmentation/resolve/main/onnx/model.onnx

# selfie_multiclass — TFLite → ONNX via uv-managed Python.
curl -L -o /tmp/selfie_multiclass.tflite \
  https://huggingface.co/yolain/selfie_multiclass_256x256/resolve/main/selfie_multiclass_256x256.tflite
uv run --python 3.10 --with "tf2onnx==1.16.1" --with "tensorflow==2.14.0" --with "numpy<2" \
  python -m tf2onnx.convert \
    --tflite /tmp/selfie_multiclass.tflite \
    --output models/selfie_multiclass.onnx \
    --opset 18

# rvm — already ONNX from upstream.
curl -L -o models/rvm.onnx \
  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx
```
