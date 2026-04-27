//! Render the LinuxBroadcast logo (`APP_DESIGN_TEMP/logo.svg`) into a 64×64
//! RGBA byte buffer suitable for `eframe::egui::IconData`.
//!
//! No `usvg`/`tiny-skia` dep — the logo is four primitives. We draw them at
//! 4× and downsample with `image::imageops` to get implicit anti-aliasing.

use eframe::egui::IconData;
use image::{ImageBuffer, Rgba};

const SUPER: u32 = 4;
const FINAL: u32 = 64;
const HI: u32 = FINAL * SUPER;

const PANEL: [u8; 4] = [0x0E, 0x11, 0x16, 0xFF];
const STROKE: [u8; 4] = [0x2A, 0x31, 0x3B, 0xFF];
const FRAME: [u8; 4] = [0xE6, 0xEA, 0xF0, 0xFF];
const ACCENT: [u8; 4] = [0x5B, 0xD4, 0xC0, 0xFF];

pub fn build() -> IconData {
    let mut buf: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(HI, HI);

    // Coordinates from the SVG (viewBox 0 0 64 64) scaled by SUPER.
    let s = SUPER as f32;
    fill_rounded_rect(&mut buf, 6.0 * s, 6.0 * s, 52.0 * s, 52.0 * s, 10.0 * s, PANEL);
    stroke_rounded_rect(&mut buf, 6.0 * s, 6.0 * s, 52.0 * s, 52.0 * s, 10.0 * s, 1.25 * s, STROKE);
    stroke_rounded_rect(&mut buf, 18.0 * s, 18.0 * s, 28.0 * s, 28.0 * s, 5.0 * s, 2.25 * s, FRAME);
    fill_circle(&mut buf, 44.0 * s, 20.0 * s, 2.6 * s, ACCENT);

    let small = image::imageops::resize(&buf, FINAL, FINAL, image::imageops::FilterType::Lanczos3);
    IconData {
        rgba: small.into_raw(),
        width: FINAL,
        height: FINAL,
    }
}

fn put(buf: &mut ImageBuffer<Rgba<u8>, Vec<u8>>, x: i32, y: i32, c: [u8; 4]) {
    if x >= 0 && (x as u32) < buf.width() && y >= 0 && (y as u32) < buf.height() {
        buf.put_pixel(x as u32, y as u32, Rgba(c));
    }
}

fn fill_rounded_rect(
    buf: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    r: f32,
    c: [u8; 4],
) {
    let x0 = x as i32;
    let y0 = y as i32;
    let x1 = (x + w) as i32;
    let y1 = (y + h) as i32;
    for py in y0..y1 {
        for px in x0..x1 {
            if !inside_rounded(px as f32 + 0.5, py as f32 + 0.5, x, y, w, h, r) {
                continue;
            }
            put(buf, px, py, c);
        }
    }
}

fn stroke_rounded_rect(
    buf: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    r: f32,
    weight: f32,
    c: [u8; 4],
) {
    let x0 = (x - weight) as i32;
    let y0 = (y - weight) as i32;
    let x1 = (x + w + weight) as i32;
    let y1 = (y + h + weight) as i32;
    let half = weight * 0.5;
    for py in y0..y1 {
        for px in x0..x1 {
            let cx = px as f32 + 0.5;
            let cy = py as f32 + 0.5;
            let d = signed_dist_rounded(cx, cy, x, y, w, h, r);
            if d.abs() <= half {
                put(buf, px, py, c);
            }
        }
    }
}

fn fill_circle(buf: &mut ImageBuffer<Rgba<u8>, Vec<u8>>, cx: f32, cy: f32, r: f32, c: [u8; 4]) {
    let x0 = (cx - r - 1.0) as i32;
    let y0 = (cy - r - 1.0) as i32;
    let x1 = (cx + r + 1.0) as i32;
    let y1 = (cy + r + 1.0) as i32;
    for py in y0..=y1 {
        for px in x0..=x1 {
            let dx = px as f32 + 0.5 - cx;
            let dy = py as f32 + 0.5 - cy;
            if dx * dx + dy * dy <= r * r {
                put(buf, px, py, c);
            }
        }
    }
}

fn inside_rounded(px: f32, py: f32, x: f32, y: f32, w: f32, h: f32, r: f32) -> bool {
    if px < x || px > x + w || py < y || py > y + h {
        return false;
    }
    // Inside inner box (no corner clipping).
    if px >= x + r && px <= x + w - r {
        return true;
    }
    if py >= y + r && py <= y + h - r {
        return true;
    }
    // Corner test.
    let cx = px.clamp(x + r, x + w - r);
    let cy = py.clamp(y + r, y + h - r);
    let dx = px - cx;
    let dy = py - cy;
    dx * dx + dy * dy <= r * r
}

fn signed_dist_rounded(px: f32, py: f32, x: f32, y: f32, w: f32, h: f32, r: f32) -> f32 {
    // Distance from point to rounded-rect outline (negative inside).
    let cx = px.clamp(x + r, x + w - r);
    let cy = py.clamp(y + r, y + h - r);
    let dx = px - cx;
    let dy = py - cy;
    let dist_to_corner_center = (dx * dx + dy * dy).sqrt();
    // If the point is inside the inner core (closer than r to the corner
    // center), we're inside the rounded rect → signed distance to outline.
    let outside = dist_to_corner_center - r;
    if px > x && px < x + w && py > y && py < y + h && outside < 0.0 {
        outside // negative
    } else {
        outside.abs()
    }
}
