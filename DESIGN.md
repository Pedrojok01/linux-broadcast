# LinuxBroadcast — Design System

A calm, dark, slightly-cool design for a Rust + egui virtual webcam app. Flat-with-strokes; no gradients or drop shadows that egui can't reproduce cleanly. The actual code that applies these tokens lives in `crates/app/src/theme.rs` — this file is the *intent* doc and the colour reference, not the source of truth.

## Principles

- **Single accent.** One signal colour (`#5BD4C0`, warm cyan) for selection, slider fill, and the ready/idle state of the primary button. The danger palette (`#E5685A`) is for live / stop only. Two colours, never compete.
- **Hierarchy via strokes, not shadows.** Three near-black tints (`bg`, `panel`, `panel_alt`/`panel_inset`) layered with hairline strokes. egui has no real drop-shadow primitive, so the design doesn't pretend.
- **Tone.** Slightly warm copy ("Set the scene", "Sending to /dev/video10") with technical detail kept where it earns trust — device paths, fps, model name.
- **Typography.** Inter (proportional) for UI; JetBrains Mono for device paths, status, footer. That contrast carries the entire "OS-tool" personality. Both are bundled at `assets/fonts/`.
- **Layout.** 320 px sidebar + preview pattern. Section headers in muted uppercase. Mode tabs (None / Blur / Replace) are a segmented control. Library is a 3-column grid with an inline dashed Import tile. Primary action pinned to the sidebar bottom and colour-coded by state.
- **Preview chrome.** Status pills (`LIVE`, resolution, sink path) live above the preview; a Mirror toggle + sink reminder hover in the bottom corners.

## Logo

Frame-within-a-frame mark with a single accent dot — the "broadcast frame" + "on-air tally" idea. Renders at 16 px. The 64×64 RGBA used for the window icon is generated programmatically at startup (`crates/app/src/icon.rs`); the original SVG is reproduced below for reference.

```svg
<svg viewBox="0 0 64 64" width="64" height="64">
  <rect x="6" y="6" width="52" height="52" rx="10" fill="#0E1116"/>
  <rect x="6" y="6" width="52" height="52" rx="10" fill="none" stroke="#2A313B" stroke-width="1.25"/>
  <rect x="18" y="18" width="28" height="28" rx="5" fill="none" stroke="#E6EAF0" stroke-width="2.25"/>
  <circle cx="44" cy="20" r="2.6" fill="#5BD4C0"/>
</svg>
```

The accent dot is the only colour element — drop it and the mark works in pure greyscale.

## Colour tokens

| Role | Hex | Use |
|---|---|---|
| `bg` | `#0B0E13` | App background |
| `panel` | `#11151C` | Header, sidebar, footer |
| `panel_alt` | `#161B23` | Hover, segmented active |
| `panel_inset` | `#0D1117` | Inputs, slider track, tile bg |
| `stroke` | `#222934` | Default hairline |
| `stroke_strong` | `#2E3744` | Hover stroke |
| `text` | `#E6EAF0` | Primary text |
| `text_weak` | `#9AA4B2` | Labels, secondary |
| `text_muted` | `#6B7585` | Section captions, mono details |
| `accent` | `#5BD4C0` | Selection, ready, focus |
| `accent_soft` | `#5BD4C022` | Primary-button fill (idle) |
| `danger` | `#E5685A` | Live / stop |
| `danger_soft` | `#E5685A22` | Primary-button fill (running) |
| `success` | `#7FCB8E` | Connected dot, footer running indicator |

## Spacing & shape scale

| Token | Value | Use |
|---|---|---|
| `radius.sm` | 4 px | Tabs, ghost buttons |
| `radius.md` | 8 px | Default widget rounding, primary button |
| `radius.lg` | 12 px | Window, preview surface |
| `space.xs` / `sm` / `md` / `lg` | 4 / 8 / 12 / 16 px | Standard paddings/gaps |
| `space.section_gap` | 18 px | Between sidebar sections |
| `space.panel_pad_y` | 14 px | Sidebar inner top/bottom margin |
| `control.primary_height` | 40 px | Start/Stop button |
| `control.thumb_radius` | 6 px | Library thumbnails |

## Type scale

| Style | Size | Family |
|---|---|---|
| Title (window heading) | 20 px | Inter |
| Heading (section labels in caps) | 11 px, 1.2 px tracking | Inter SemiBold |
| Body / Button | 13 px | Inter |
| Small | 11 px | Inter |
| Mono | 11 px | JetBrains Mono |

## Things deliberately omitted

- Fancy gradients, drop shadows, glassmorphism — egui can't reproduce them faithfully.
- Iconography for every row (kept icons to controls that benefit; labels carry the rest).
- Light theme — can be derived from these tokens later if needed.

## Where the tokens land in egui

`Color32` constants live in `theme::color`; spacing in `theme::space`; radii in `theme::radius`. `theme::apply()` registers Inter + JetBrains Mono via `FontDefinitions` and writes the full `egui::Style`/`Visuals` from those constants. Re-read `crates/app/src/theme.rs` if any value here looks out of date — code is authoritative.
