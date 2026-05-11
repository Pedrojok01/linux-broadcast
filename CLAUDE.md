# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Background-replacement virtual webcam for Linux. Captures a webcam frame, runs MediaPipe / RVM segmentation on CPU via `ort` (ONNX Runtime), composites the foreground over a blurred background, a saved image, or passes the frame through unchanged, and writes the result to a `v4l2loopback` virtual camera that Zoom / Meet / Teams / Firefox / OBS consume.

Out of scope: audio / microphone effects.

## Common commands

```bash
# Manual host setup — only needed when building from source. The .deb's
# postinst runs the same commands and ships conffiles in /etc/modprobe.d/
# + /etc/modules-load.d/ so the module persists.
sudo modprobe -r v4l2loopback 2>/dev/null
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="LinuxBroadcast" \
  exclusive_caps=1 max_buffers=2

# GUI (default)
cargo run --release -p linux-broadcast

# Headless — starts hidden in tray and auto-starts the pipeline. Used by
# the autostart .desktop. Both forms work.
cargo run --release -p linux-broadcast -- --headless
LB_HEADLESS=1 cargo run --release -p linux-broadcast

# Dump the bundled window icon to /tmp/lb-icon.png. Same code path that
# regenerates packaging/LinuxBroadcast.png when the logo changes.
LB_DUMP_ICON=1 cargo run --release -p linux-broadcast

# Lint / format
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all

# Build a local .deb (target/debian/linux-broadcast_<ver>-1_amd64.deb).
cargo install cargo-deb   # one-time
cargo deb -p linux-broadcast

# Cut a release. Bumps Cargo.toml + path-dep + Cargo.lock, runs CI
# checks, commits, tags. --push to send to origin and let CI ship.
scripts/release.sh 0.1.3 --push

# Verify the virtual cam from another terminal (only works while the
# pipeline is running — exclusive_caps=1 hides /dev/video10 otherwise).
ffplay -fflags nobuffer -f v4l2 -input_format yuyv422 \
  -video_size 1280x720 /dev/video10
```

System dev packages are listed in [README §Build from source](README.md#option-b--build-from-source). Pin `v4l2loopback-dkms ≥ 0.12.8` — 0.12.7 fails to build on kernel 6.8+.

## Architecture

Cargo workspace, two crates:

- **`crates/pipeline`** (lib `lb_pipeline`) — the entire video pipeline, headless, no GUI deps.
- **`crates/app`** (bin `linux-broadcast`) — `eframe`/`egui` GUI driving the pipeline. Owns config persistence, the saved-background library, the theme, autostart, tray.

### Frame pipeline (two GStreamer graphs, one feeder, lazy by default)

`/dev/video0` is only opened while a real consumer is reading `/dev/video10` or the GUI preview pane is visible. To do that without `/dev/video10` blinking out of conferencing-app device lists, the pipeline is split into two GStreamer graphs glued by a Rust feeder thread.

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

Non-obvious settings — these took a session each to nail:

- **Sink stays PLAYING permanently and the feeder keeps pushing.** The sink graph is what advertises `/dev/video10` as a CAPTURE device. With `exclusive_caps=1`, PLAYING alone is not enough — v4l2loopback only flips on `V4L2_CAP_VIDEO_CAPTURE` after `v4l2sink` has called `VIDIOC_STREAMON`, which only happens once at least one buffer has flowed. So while Idle the feeder re-pushes the last composited frame (or a black frame on cold start) at `IDLE_PUSH_INTERVAL` (5 Hz). After the first push the kernel's `ready_for_capture` flag is sticky for the lifetime of the producer fd; 5 Hz is just slow enough to be free and fast enough to dodge any WebRTC consumer's no-frames timeout.
- **Pipeline starts at app launch in every mode.** Both `--headless` autostart and the GUI launch invoke `Pipeline::start` immediately. The lazy state machine still keeps `/dev/video0` released until a real consumer reads.
- **`v4l2sink async=false`** — without it the pipeline deadlocks on the PLAYING transition: v4l2sink waits for a preroll buffer that the lazy feeder may never push.
- **`v4l2src do-timestamp=true`** — without it the output stream's PTS is wrong and v4l2sink's pacing slips.
- **Caps strategy:** the source-side capsfilter pins **only RGBA + width/height** (no framerate). Forcing 30/1 caused negotiation failures with the C920 (camera reports `30000/1001`). The appsrc-side caps **do** declare framerate, otherwise the sink-side videoconvert asserts `fps_n == out_fps_n`.

Live setting changes (background mode, blur strength, image swap, GUI preview toggle, auto-frame on/off) flow over a `crossbeam-channel` of `Command`s and apply on the next feeder tick — no graph rebuild. Camera, resolution, and **model** changes still require Stop+Start.

#### Lazy mode constants and consumer detection

- **Activation debounce: 2 s** (`lazy::ACTIVATION_DEBOUNCE`). Sized to reject browser capability-probes (Chrome / Firefox open `/dev/video10` briefly for `ENUM_FMT` without intending to stream — typically <500 ms).
- **Deactivation debounce: 3 s** (`lazy::DEACTIVATION_DEBOUNCE`). Absorbs in-call camera-switcher flicker (the user toggles cameras in Meet's picker).
- **Watcher poll interval: 800 ms** (`pipeline::WATCH_POLL_INTERVAL`). Fast enough that a real consumer is observed inside the activation window.
- **Detection mechanism: walk `/proc/*/fd/*`** in `consumer_watch::current_consumers`. v4l2loopback exposes no sysfs consumer-count and the kernel doesn't fire fsnotify on character-device opens, so userspace polling is the only portable signal. ~1–3 ms per poll.
- The GUI preview pane is a **synthetic consumer**: while the window is visible AND the *Show preview* toggle is on, the camera stays lit. Hiding the window or turning off the toggle drops the signal. There's intentionally no "GUI window open ⇒ camera on" coupling beyond this — anything stronger defeats the lazy-mode promise.

### Cross-cutting conventions

- **RGBA8 end-to-end.** YUYV/MJPEG↔RGBA and RGBA→YUY2 happen in GStreamer `videoconvert`; Rust never touches non-RGBA pixels.
- **Mask is `f32`** between segmentation and composite. `u8` round-trips cost quality without saving memory.
- **Inference at native 256×256.** Only upsample + composite touch frame-resolution pixels — keeps CPU low at 720p/1080p.
- **`Background::None` is true passthrough.** new_sample short-circuits *before* segmentation and resets the EMA smoother so the next non-None frame starts clean.

### Models

Two ONNX files, both `include_bytes!`'d in `crates/app/src/main.rs`. **RVM is the default** — blur radius and auto-frame are tuned against it; **multiclass** is the low-CPU fallback (visible mask flicker at hair / glasses / low-contrast edges).

- **`models/rvm.onnx`** — Robust Video Matting, MobileNetV3 fp32. Recurrent: 4 state tensors held on the segmenter, `Segmenter::reset()` clears them on `Background::None` switch or input-dimension change. Internal compute scaled by `RVM_DOWNSAMPLE_RATIO` (currently 0.5 → ~640×360 internal on a 720p frame, ~40–60 ms/frame on one x86 core). Output mask is at frame resolution; the compositor's `prepare_mask` skips upsampling when sizes match.
- **`models/selfie_multiclass.onnx`** — MediaPipe selfie multiclass, 256×256 NHWC, 6 classes. Per-frame, ~2–3× cheaper than RVM at 720p.

`Mask` is a public `(data, width, height)` so each model declares its native resolution; the compositor handles either. Active model is `lb_pipeline::ModelKind` (re-exported as `config::Model`); the GUI dropdown drives a Stop+Start on change.

For SHA-256s, license details, and re-fetch / TFLite→ONNX recipes, see [`models/README.md`](models/README.md).

### Auto-framing

`framing.rs` computes a `Framing` (foreground anchor + zoom) from the silhouette mask: mass-weighted horizontal centroid for `cx`, **top-edge row** for `cy` (the vertical centroid would crop heads when zoomed). `AnchorLock` snaps to the silhouette on the first detection and holds that anchor for the rest of the session — toggling auto-frame off+on (or a Live exit) resets the lock. Returns `None` only when the lock is not yet engaged AND no foreground is detected; once locked, it survives later frames where the segmenter loses the silhouette. Foreground zoom is a static `FG_ZOOM` (no UI control). The compositor remaps foreground sample points only — background plane stays fixed; the `mask = 0` strip vacated on the trailing edge is filled by the existing blend.

### Process lifecycle gotchas

- **Single-instance lock at process scope.** `lock.rs` flocks `$XDG_RUNTIME_DIR/linux-broadcast.lock` (config-dir fallback) for the lifetime of the LB process — a lazy-mode instance can sit Idle for arbitrarily long with no `/dev/video10` write contention, so we'd otherwise allow two instances to coexist and race the moment a consumer arrives. A second launch finds the lock held and exits cleanly.
- **Headless `App::new` polls for `/dev/video10`** (10 s timeout) before auto-starting the pipeline, so it survives the autostart-vs-`systemd-modules-load.service` race on cold boot.
- **Window close hides to tray, not exits.** The close button is intercepted (`ViewportCommand::CancelClose` + `Visible(false)`); only the tray's *Quit* sets `quit_requested` and lets the close through. Hiding the window also clears the GUI-preview synthetic-consumer signal so a tray-only instance lets the camera drop to Idle.
- **Tray needs a GTK loop.** `tray.rs` spawns a dedicated `lb-tray-gtk` thread that runs `gtk::main()` because tray-icon's Linux backend (libayatana-appindicator) needs a GTK loop on *some* thread, and egui/winit don't host one. Install can fail on systems without a tray host — the failure is logged and the GUI keeps working without a tray entry.
- **`desktop_install.rs` is skipped** when the `.deb`-installed system entry at `/usr/share/applications/LinuxBroadcast.desktop` is present, to avoid duplicate menu entries.

### Packaging (`packaging/` + `[package.metadata.deb]`)

`cargo deb -p linux-broadcast` reads `crates/app/Cargo.toml` and ships:

| Asset | Installed to | Notes |
|---|---|---|
| `target/release/linux-broadcast` | `/usr/bin/linux-broadcast` | The binary. |
| `packaging/LinuxBroadcast.desktop` | `/usr/share/applications/` | System launcher; `desktop_install.rs` skips its per-user clone when this exists. |
| `packaging/LinuxBroadcast.png` | `/usr/share/icons/hicolor/64x64/apps/` | Pre-rendered via `LB_DUMP_ICON=1` so `cargo deb` doesn't execute the binary at packaging time. Regenerate when the logo changes. |
| `packaging/linux-broadcast.modprobe.conf` | `/etc/modprobe.d/linux-broadcast.conf` | **conffile** — `options v4l2loopback devices=1 video_nr=10 card_label="LinuxBroadcast" exclusive_caps=1 max_buffers=2`. |
| `packaging/linux-broadcast.modules-load.conf` | `/etc/modules-load.d/linux-broadcast.conf` | **conffile** — single line `v4l2loopback`, makes the module reload on every boot. |

Maintainer scripts at `packaging/scripts/`:
- `postinst` — drops a stale module if loaded with different params, then `modprobe v4l2loopback` (options come from the modprobe.d drop-in). Refreshes `update-desktop-database` and `gtk-update-icon-cache`. Module-load failure is logged but does **not** fail the install: DKMS may still be building, and the `modules-load.d` file guarantees the next boot loads it.
- `prerm` — best-effort `modprobe -r v4l2loopback` on uninstall. Failure is harmless.
- `postrm` — refresh desktop / icon caches after the system files are gone.

`apt purge` removes the conffiles; `apt remove` keeps them for re-install.

Pushing a `v*` tag triggers `.github/workflows/release.yml`: it checks the tag matches `[workspace.package].version`, runs `cargo deb`, and publishes a GitHub Release with auto-generated notes and the `.deb` attached.

### Packaging (`packaging/aur/` — Arch / AUR)

The `linux-broadcast-bin` AUR package repackages the GitHub Release `.deb` so Arch users get the same binary, conffiles, and desktop entry as Debian users.

| File | Purpose |
|---|---|
| `packaging/aur/PKGBUILD` | `pkgname=linux-broadcast-bin`, `provides=("linux-broadcast=$pkgver")`, `conflicts=('linux-broadcast')`. `package()` extracts `data.tar.{zst,xz}` from the `.deb` into `$pkgdir` and relocates `usr/share/doc/$pkg/copyright` → `usr/share/licenses/$pkg/LICENSE` to match Arch's FHS. `options=('!strip')` because `cargo --release` already strips. |
| `packaging/aur/linux-broadcast.install` | `post_install` / `post_upgrade` / `post_remove` hooks. Mirror `packaging/scripts/postinst` semantics: drop stale `v4l2loopback`, reload, refresh desktop / icon caches. |
| `packaging/aur/.SRCINFO` | **Must be byte-identical to `makepkg --printsrcinfo` output** for the current PKGBUILD. The `aur-lint` workflow diffs them on PRs. Regenerate via the docker one-liner in that workflow's error message. |

Two workflows automate the loop:

- **`.github/workflows/aur-lint.yml`** (PR-time gate on `packaging/aur/**`): runs `makepkg --printsrcinfo` inside `archlinux:base-devel`, diffs against the committed `.SRCINFO`, runs `namcap` on the PKGBUILD and fails on `E:` lines (advisory `W:`/`I:` are surfaced but non-fatal).
- **`release.yml`'s `aur` job** (`needs: deb`, runs on every `v*` tag push): waits up to ~5 min for the just-uploaded `.deb` URL to be reachable, downloads it, computes sha256, sed-rewrites `pkgver` + `sha256sums` in PKGBUILD, regenerates `.SRCINFO`, and pushes to AUR using `ssh-keyscan` + `git push` directly (the `AUR_SSH_KEY` repo secret is the ed25519 private key registered on the AUR account).
- **`.github/workflows/aur.yml`** is a `workflow_dispatch`-only safety valve (`gh workflow run aur.yml -f tag=v0.1.2`) for re-publishing to AUR without re-tagging. Uses the same inline ssh + git push as the release job.

Non-obvious bits:

- **AUR publishing lives in `release.yml`, not its own workflow.** Tried that first; `release: published` events fired by `GITHUB_TOKEN` don't trigger downstream workflows. Putting both jobs in the same workflow with `needs: deb` sidesteps the cross-workflow trigger entirely.
- **`sha256sums=('SKIP')` is committed.** Real sums only exist at release time — the `.deb` doesn't exist when the PKGBUILD is edited, and pinning a stale sum would block the workflow's substitution. The lint workflow is fine with `SKIP`; namcap warns about it advisorily.
- **`v4l2loopback-dkms` is an AUR dep itself.** Helpers (`yay`, `paru`) handle the transitive AUR install. Do not change to a non-AUR alias.
- **AUR repo init is one-time, manual.** `KSXGitHub/...-deploy-aur` only updates an existing AUR repo. The first push (via `git clone ssh://aur@aur.archlinux.org/linux-broadcast-bin.git` + push) reserves the package name.
- **AUR pushes are over SSH only**, IPv4-only on most consumer ISPs in practice. The CI runner doesn't care; for local pushes set `Host aur.archlinux.org / AddressFamily inet` in `~/.ssh/config`.

### Adding a new background mode

1. Add a variant to `Background` in `compositor.rs`.
2. Add a branch to `Compositor::composite` that produces the new background plane (frame-sized RGBA8) and reuses `out = fg*mask + bg*(1-mask)`.
3. Add a `Mode` to `app/src/config.rs`.
4. Add the segmented-control tab in `ui::sidebar_scene` and wire it through `build_background`.
5. The pipeline picks up the new mode via the existing `Command::SetBackground` — no graph rebuild.

### Adding a new model

1. Add a `ModelKind` variant in `pipeline/src/segmenter.rs` and a `segment_*` function for its pre/post.
2. Bundle the ONNX in `models/` and `include_bytes!` it from `crates/app/src/main.rs`.
3. Extend the `Pipeline::start` call to pass the new bytes.
4. Add a config-side `Model` variant (with serde) in `app/src/config.rs` and surface it in `sidebar_model`.
5. The GUI auto-restarts the pipeline on model change.

### Tests

`cargo test -p lb_pipeline`:

- `tests/models_smoke.rs` — loads each bundled ONNX through `Segmenter::from_bytes`, runs 1–2 inferences on a synthetic frame, asserts mask shape + value range. RVM also covers `reset()` clearing recurrent state.
- `tests/synthetic_graph.rs` — drives `videotestsrc → … → appsink` through the `Compositor` and back into the sink graph, verifying caps negotiation and PTS pacing without touching `/dev/video0` or `/dev/video10`.

The GUI crate has no tests — surface is mostly egui layout. Don't add UI snapshot tests without a strong reason; egui rendering is too version-sensitive.

## Roadmap

- CPU/GPU usage in footer.
- Throughput benchmarks per model on a stable reference machine, published in the repo so contributors can spot regressions on a model swap.
