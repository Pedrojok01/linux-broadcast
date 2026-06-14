# Pipeline throughput — measurement & single-thread optimisation

Investigation into the "RVM caps output FPS" hypothesis. Branch:
`perf/inference-decouple`.

**TL;DR** — the original theory (inference is the bottleneck) was only
*one third* of the story. Measuring the live loop revealed three stacked
ceilings, in order of impact:

1. **The Live loop was paced by a 100 ms idle ticker → hard 10 fps cap**,
   independent of how fast inference or compositing ran. This was the real
   bottleneck. Fixed with a one-line change (don't park on the idler while
   Live); output now scales with actual compute. **Up to 2.2× on its own.**
2. **Compositing is expensive** and runs *every* frame: ~35 ms for blur,
   **~73 ms for blur + auto-frame** (the crop/rescale pass roughly doubles
   it), ~21 ms for image-replace. This is a hard floor that inference
   threading cannot remove.
3. **RVM inference** ~66 ms/frame — the thing we originally targeted. Real,
   but secondary to #1, and reducible by skipping frames (#2 of the plan).

The single-threaded changes in this branch (idler fix + 1-in-N inference)
take the measured configs from a flat 10 fps to **17–22 fps** with no
threading. Whether step 3 (multi-threading) is worth it depends entirely
on the background mode — see [Verdict](#verdict-on-step-3-multi-threading).

---

## Method

- **Hardware:** Intel Core i5-14600KF (14 cores / 20 threads), CPU-only
  inference. *A weaker/laptop CPU will be more inference-bound than the
  numbers here — RVM single-frame latency dominates there.*
- **Resolution / model:** 1280×720, RVM (default), `RVM_DOWNSAMPLE_RATIO`
  = 0.5. Camera (Logitech, YUYV) confirmed capable of 30 fps at 720p, so
  it is *not* the limiter.
- **Harness:** `crates/pipeline/examples/bench.rs` starts the real
  pipeline headless; an external `ffmpeg -f v4l2 -i /dev/video10 -f null -`
  reads the loopback to force the feeder Live. Per-stage timings come from
  the feeder's own profiler (`crates/pipeline/src/profile.rs`, active under
  `LB_PROFILE=1`), which prints mean/max ms per stage and effective FPS
  over rolling 90-frame windows.
- **Numbers below are steady-state** — the first window of each run is
  discarded (it includes the ~5 s activation + ORT first-inference warmup).
- **Caveat:** segment/composite means wobble ±several ms run-to-run because
  sustained load lowers turbo headroom. The *FPS* figures and the
  structural conclusions are robust; don't read 2–3 ms stage deltas as
  signal.

`fps` = frames actually delivered to `/dev/video10` per second.
`segment` only runs on inference frames (`infer/s` = inferences/sec).
`composite` runs on **every** frame.

---

## Results

`interval` = `LB_INFER_INTERVAL` (run RVM once every N frames, reuse the
cached mask on the others). All times in ms (mean).

### Background = Blur, auto-frame OFF

| interval | BEFORE fps | AFTER fps | speedup | segment | composite | infer/s (after) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.7 | **9.7**  | 1.26× | 67 | 35 | 9.7 |
| 2 | 9.5 | **14.6** | 1.54× | 66 | 35 | 7.3 |
| 3 | 9.9 | **17.5** | 1.77× | 66 | 35 | 5.8 |

### Background = Blur, auto-frame ON

| interval | BEFORE fps | AFTER fps | speedup | segment | composite | infer/s (after) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.1 | **7.1**  | 1.00× | 67 | 73 | 7.1 |
| 2 | 8.4 | **9.3**  | 1.11× | 67 | 73 | 4.7 |
| 3 | 8.8 | **10.5** | 1.19× | 67 | 73 | 3.5 |

### Background = Image-replace, auto-frame ON  *(your saved config)*

| interval | BEFORE fps | AFTER fps | speedup | segment | composite | infer/s (after) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.0 | **11.0** | 1.10× | 68 | 22 | 11.0 |
| 2 | 10.0 | **15.7** | 1.57× | 78 | 23 | 7.8 |
| 3 | 10.0 | **22.1** | 2.21× | 68 | 21 | 7.4 |

Notice the BEFORE column for image is a flat **10.0** regardless of
interval — that is the 100 ms idler cap, fully masking the frame-skip
optimisation. Once the idler is gone, frame-skip does what it's supposed
to.

---

## The three bottlenecks

### 1. The 100 ms idler (the real ceiling) — *fixed*

The feeder's run loop blocked in `select!` on a `tick(100 ms)` idler
between every frame. While Live, with no commands or consumer-changes
arriving, the only thing waking the loop was that 100 ms tick → it pumped
exactly **10 frames/sec** no matter how cheap the work was (e.g. a 37 ms
tick still produced only 10 fps, wasting 63 ms parked).

**Fix:** while `State::Live`, park for `Duration::ZERO` instead of
`IDLE_TICK`; `pump_one_frame`'s blocking `try_pull_sample` is the natural
pace-maker (≈ camera FPS). When not Live, keep parking a full tick (no
frames to pull, stay power-efficient). One `select!` arm changed.

Side effect: inference got *faster* (≈ 86 → 66 ms) because the ORT thread
pool no longer parks during 60 ms idle gaps and pay wake-up latency each
frame.

### 2. Compositing cost (a hard per-frame floor) — *not addressed*

Composite runs on every output frame and is single-threaded:

| mode | composite (ms) | ⇒ max possible fps |
|---|---:|---:|
| Image-replace (+ frame) | ~21 | ~47 |
| Blur (no frame) | ~35 | ~28 |
| **Blur + auto-frame** | **~73** | **~14** |

Auto-frame's post-composite crop+rescale (a full-frame bilinear resample)
adds ~38 ms on top of blur's ~35 ms. **This is why Blur + auto-frame is
stuck near 13–14 fps and nothing on the inference side can move it.**

### 3. RVM inference (~66 ms) — *addressed by frame-skip / future threading*

Reducible two ways: run it less often (the `infer_interval` knob added
here, single-thread) or move it off the pump thread (step 3, threading).
The cost is mask freshness — see below.

---

## What the single-thread changes cost (quality)

`infer_interval > 1` composites a *fresh* camera frame against a mask that
is up to `interval − 1` frames old. The foreground (your face) still moves
at full FPS; only the silhouette **edge** lags during fast motion →
slight background bleed on the leading edge / clipped trailing edge.

- **Blur:** nearly invisible (soft edge + blurred background).
- **Image-replace:** more visible as a transient halo/cut on fast motion.
- **Static talking-head (the common case):** no perceptible difference.

`interval = 2` halves inference cost for a ~1-frame mask lag — a good
default trade. `interval = 3` is noticeably laggier on motion.

---

## Verdict on step 3 (multi-threading)

With inference moved to a worker thread, output FPS becomes bound by the
**composite** floor (#2). So the payoff is mode-dependent:

| mode | single-thread best (this branch) | with inference threading | threading worth it? |
|---|---:|---:|---|
| Blur, no frame | 17.5 (i3) | ~28 (composite-bound) | **Yes** |
| Image + frame | 22.1 (i3) | ~30 (camera-capped) | **Yes** |
| **Blur + frame** | 10.5 (i3) | **~14 (composite-bound)** | **No** — needs composite work, not threading |

**Recommendation:**

1. **Ship the idler fix unconditionally.** It fixes a genuine bug, costs
   nothing in quality, and is the single biggest win. *(Already in this
   branch.)*
2. **Keep frame-skip as an opt-in knob** (currently `LB_INFER_INTERVAL`;
   wire to a GUI "Performance / Balanced / Quality" preset later). Default
   `1` keeps current quality; `2` is the sweet spot if you want more FPS.
3. **Before doing step 3, decide what you actually need.** If 15–22 fps is
   acceptable (it usually is for a low-motion webcam), you may be done.
   If you want ~30 fps:
   - For **plain blur** and **image-replace**, inference threading gets
     you there.
   - For **blur + auto-frame**, threading does nothing — the lever is
     **optimising the composite** (parallelise the blur + the auto-frame
     resample, or do them at reduced resolution). That's arguably a better
     step 3 than threading, since it also lifts the threaded ceilings.

---

## Changes in this branch

- `crates/pipeline/src/profile.rs` — new. Per-stage timing under
  `LB_PROFILE`.
- `crates/pipeline/src/lazy.rs`:
  - **Idler fix** — `default(park)` with `park = ZERO` while Live.
  - **Frame-skip** — `infer_interval` (env `LB_INFER_INTERVAL`, default 1)
    with a cached mask/framing reused on skipped ticks; cache cleared on
    Live-exit and framing toggle. `compute_framing` extracted from
    `pump_one_frame`.
  - Profiler wired into the Live tick.
- `crates/pipeline/src/segmenter.rs` — `Mask` derives `Clone` (for caching).
- `crates/pipeline/examples/bench.rs` — new. Headless bench harness.

All single-threaded. No public API change. `LB_INFER_INTERVAL` is the only
new runtime knob; with it unset (=1) behaviour matches today's quality,
just without the 10 fps idler cap.

## Reproduce

```bash
# 1. stop any running instance so the producer can own /dev/video10
pkill -f 'linux-broadcast --headless'

# 2. build the bench
cargo build --release -p lb-pipeline --example bench

# 3. run one config; in a second terminal, read the loopback to go Live:
#    ffmpeg -nostdin -loglevel error -f v4l2 -i /dev/video10 -f null -
LB_PROFILE=1 LB_INFER_INTERVAL=2 LB_BENCH_BG=blur LB_BENCH_FRAMING=0 \
  LB_BENCH_SECS=42 ./target/release/examples/bench

# knobs: LB_BENCH_BG=blur|image|none  LB_BENCH_FRAMING=0|1  LB_INFER_INTERVAL=N
```
