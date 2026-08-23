#!/usr/bin/env bash
# Builds the linux-broadcast-cuda add-on .deb: only the ONNX Runtime CUDA
# provider libraries, version-locked to the main package. The main binary
# (cuda build) loads them from /usr/lib/linux-broadcast/ when present and the
# user has CUDA 13 + cuDNN 9 installed; otherwise it runs on CPU.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')
ARCH=amd64
OUT="target/debian"

# 1. Find the CUDA 13 provider libs. Set ORT_PROVIDER_DIR to package a custom
# ONNX Runtime build (for example, one compiled for a newer GPU architecture).
# Without it, build normally and take the newest CUDA 13 provider from ort's
# cache. The cache can contain cuda-12 and older cuda-13 builds, hence the
# ABI check below.
if [ -n "${ORT_PROVIDER_DIR:-}" ]; then
  DIR="$ORT_PROVIDER_DIR"
else
  ORT_CUDA_VERSION=13 cargo build --release -p linux-broadcast --features cuda
  DIR=$(find "$HOME/.cache/ort.pyke.io" -name 'libonnxruntime_providers_cuda.so' \
    -exec sh -c 'readelf -d "$1" 2>/dev/null | grep -q libcublasLt.so.13 \
      && printf "%s %s\n" "$(stat -c %Y "$1")" "$(dirname "$1")"' _ {} \; \
    | sort -rn | head -1 | cut -d' ' -f2-)
fi

[ -n "${DIR:-}" ] && [ -f "$DIR/libonnxruntime_providers_cuda.so" ] \
  && [ -f "$DIR/libonnxruntime_providers_shared.so" ] \
  || { echo "error: CUDA 13 provider libraries were not found" >&2; exit 1; }
readelf -d "$DIR/libonnxruntime_providers_cuda.so" 2>/dev/null \
  | grep -q 'libcublasLt.so.13' \
  || { echo "error: provider is not built for CUDA 13" >&2; exit 1; }

# 3. Assemble the .deb tree.
PKG="linux-broadcast-cuda"
ROOT="target/cuda-addon"
rm -rf "$ROOT"
mkdir -p "$ROOT/DEBIAN" "$ROOT/usr/lib/linux-broadcast"
cp "$DIR/libonnxruntime_providers_shared.so" "$DIR/libonnxruntime_providers_cuda.so" \
   "$ROOT/usr/lib/linux-broadcast/"
SIZE=$(du -ks "$ROOT/usr" | cut -f1)

cat > "$ROOT/DEBIAN/control" <<EOF
Package: $PKG
Version: ${VERSION}-1
Architecture: $ARCH
Maintainer: Pedrojok01 <pedrojok@pm.me>
Depends: linux-broadcast (= ${VERSION}-1)
Section: video
Priority: optional
Installed-Size: $SIZE
Description: GPU acceleration add-on for LinuxBroadcast (NVIDIA CUDA)
 Drops the ONNX Runtime CUDA execution-provider libraries next to the
 LinuxBroadcast binary. With an NVIDIA RTX-class GPU and the CUDA 13 runtime
 plus cuDNN 9 installed, segmentation runs on the GPU; otherwise LinuxBroadcast
 transparently continues on the CPU.
 .
 Requires (user-installed, not auto-resolved): CUDA 13 runtime libraries and
 cuDNN 9 for CUDA 13. See /usr/share/doc/linux-broadcast/README.md.
EOF

mkdir -p "$OUT"
DEB="$OUT/${PKG}_${VERSION}-1_${ARCH}.deb"
dpkg-deb --build --root-owner-group "$ROOT" "$DEB"
echo "built: $DEB"
dpkg-deb -c "$DEB"
