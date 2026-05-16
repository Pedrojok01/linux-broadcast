//! Procedural "camera is loading" frame for the Idle-state push.
//!
//! Renders directly into a flat RGBA8 `&mut [u8]` so we don't need the
//! `image` crate in the pipeline crate. The dark fill + LinuxBroadcast
//! logo are pre-baked once into a scratch buffer at construction and
//! memcopied into every output frame; the only per-frame work is the
//! rotating 8-segment spinner (~200 anti-aliased pixels), which is
//! negligible. No text — conferencing apps mirror the local self-view,
//! so a symmetric mark reads identically to the user and to remote
//! participants.

const BG: [u8; 4] = [0x10, 0x14, 0x18, 0xFF];
const PANEL: [u8; 4] = [0x0E, 0x11, 0x16, 0xFF];
const STROKE: [u8; 4] = [0x2A, 0x31, 0x3B, 0xFF];
const FRAME: [u8; 4] = [0xE6, 0xEA, 0xF0, 0xFF];
const ACCENT: [u8; 4] = [0x5B, 0xD4, 0xC0, 0xFF];
const DOT: [u8; 3] = [0xC8, 0xCF, 0xD8];

/// Stateful procedural drawer for the Idle-state still. Keeps a scratch
/// copy of the static layer (dark fill + logo) so each `render` call is
/// one memcpy plus a tiny anti-aliased spinner pass.
pub(crate) struct IdleLoader {
    width: u32,
    height: u32,
    /// Pre-rendered dark background + logo at `(width × height × 4)`.
    /// Rebuilt only when dimensions change.
    static_layer: Vec<u8>,
    /// Spinner phase, incremented once per `render`. Wraps cleanly.
    phase: u32,
}

impl IdleLoader {
    pub(crate) fn new(width: u32, height: u32) -> Self {
        let mut s = Self {
            width,
            height,
            static_layer: Vec::new(),
            phase: 0,
        };
        s.rebuild_static();
        s
    }

    /// Draw the next loader frame into `out`. Resizes `out` if needed.
    pub(crate) fn render(&mut self, out: &mut Vec<u8>) {
        let n = (self.width as usize) * (self.height as usize) * 4;
        if out.len() != n {
            out.resize(n, 0);
        }
        out.copy_from_slice(&self.static_layer);
        self.draw_spinner(out);
        self.phase = self.phase.wrapping_add(1);
    }

    fn rebuild_static(&mut self) {
        let w = self.width;
        let h = self.height;
        let mut buf = vec![0u8; (w as usize) * (h as usize) * 4];
        fill_solid(&mut buf, w, BG);
        draw_logo(&mut buf, w, h);
        self.static_layer = buf;
    }

    fn draw_spinner(&self, out: &mut [u8]) {
        // 8-segment spinner below the logo. At ~10 Hz push rate, one
        // full revolution takes 0.8 s — easy to read as "loading".
        const N: usize = 8;
        // Dimmest head opacity (the leading dot) → brightest tail.
        // Inverted from a typical spinner because the trailing dots fade
        // out, giving the eye a clear sense of direction.
        const HEAD: f32 = 0.18;
        const TAIL: f32 = 1.0;

        let w = self.width as f32;
        let h = self.height as f32;
        // Anchor to the canvas centre so any aspect / size works without
        // bespoke offsets. Spinner sits ~12 % below center, logo above.
        let cx = w * 0.5;
        let cy = h * 0.5 + h * 0.22;
        let ring_r = (h * 0.04).max(20.0);
        let dot_r = (h * 0.012).max(5.0);

        for i in 0..N {
            let theta =
                (i as f32) * std::f32::consts::TAU / (N as f32) - std::f32::consts::FRAC_PI_2; // start at 12 o'clock
            let x = cx + theta.cos() * ring_r;
            let y = cy + theta.sin() * ring_r;
            // Distance backward from the current head (phase). The head
            // is the dot at index == phase mod N.
            let lag = (i as i32 - self.phase as i32).rem_euclid(N as i32) as f32;
            let t = lag / (N as f32 - 1.0); // 0 at head, 1 at tail
            let opacity = HEAD + (TAIL - HEAD) * t;
            fill_circle_aa(out, self.width, self.height, x, y, dot_r, DOT, opacity);
        }
    }
}

fn draw_logo(buf: &mut [u8], w: u32, h: u32) {
    // 280-px logo using the icon.rs primitives at the equivalent of a
    // 64-unit viewBox scaled to logo_size. Anchored 12 % above center.
    let logo_size = (h as f32 * 0.36).clamp(160.0, 360.0);
    let s = logo_size / 64.0;
    let cx = w as f32 * 0.5;
    let cy = h as f32 * 0.5 - h as f32 * 0.10;
    let x = cx - logo_size * 0.5;
    let y = cy - logo_size * 0.5;

    fill_rounded_rect(
        buf,
        w,
        h,
        x + 6.0 * s,
        y + 6.0 * s,
        52.0 * s,
        52.0 * s,
        10.0 * s,
        PANEL,
    );
    stroke_rounded_rect(
        buf,
        w,
        h,
        x + 6.0 * s,
        y + 6.0 * s,
        52.0 * s,
        52.0 * s,
        10.0 * s,
        1.25 * s,
        STROKE,
    );
    stroke_rounded_rect(
        buf,
        w,
        h,
        x + 18.0 * s,
        y + 18.0 * s,
        28.0 * s,
        28.0 * s,
        5.0 * s,
        2.25 * s,
        FRAME,
    );
    fill_circle(buf, w, h, x + 44.0 * s, y + 20.0 * s, 2.6 * s, ACCENT);
}

// ----- primitives over a flat RGBA8 slice -----------------------------------

fn fill_solid(buf: &mut [u8], _w: u32, c: [u8; 4]) {
    for px in buf.chunks_exact_mut(4) {
        px.copy_from_slice(&c);
    }
}

fn put(buf: &mut [u8], w: u32, h: u32, x: i32, y: i32, c: [u8; 4]) {
    if x < 0 || y < 0 || x as u32 >= w || y as u32 >= h {
        return;
    }
    let i = ((y as u32 * w + x as u32) * 4) as usize;
    buf[i..i + 4].copy_from_slice(&c);
}

#[allow(clippy::too_many_arguments)]
fn fill_rounded_rect(
    buf: &mut [u8],
    w: u32,
    h: u32,
    x: f32,
    y: f32,
    rw: f32,
    rh: f32,
    r: f32,
    c: [u8; 4],
) {
    let x0 = x as i32;
    let y0 = y as i32;
    let x1 = (x + rw) as i32;
    let y1 = (y + rh) as i32;
    for py in y0..y1 {
        for px in x0..x1 {
            if inside_rounded(px as f32 + 0.5, py as f32 + 0.5, x, y, rw, rh, r) {
                put(buf, w, h, px, py, c);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn stroke_rounded_rect(
    buf: &mut [u8],
    w: u32,
    h: u32,
    x: f32,
    y: f32,
    rw: f32,
    rh: f32,
    r: f32,
    weight: f32,
    c: [u8; 4],
) {
    let x0 = (x - weight) as i32;
    let y0 = (y - weight) as i32;
    let x1 = (x + rw + weight) as i32;
    let y1 = (y + rh + weight) as i32;
    let half = weight * 0.5;
    for py in y0..y1 {
        for px in x0..x1 {
            let cx = px as f32 + 0.5;
            let cy = py as f32 + 0.5;
            let d = signed_dist_rounded(cx, cy, x, y, rw, rh, r);
            if d.abs() <= half {
                put(buf, w, h, px, py, c);
            }
        }
    }
}

fn fill_circle(buf: &mut [u8], w: u32, h: u32, cx: f32, cy: f32, r: f32, c: [u8; 4]) {
    let x0 = (cx - r - 1.0) as i32;
    let y0 = (cy - r - 1.0) as i32;
    let x1 = (cx + r + 1.0) as i32;
    let y1 = (cy + r + 1.0) as i32;
    for py in y0..=y1 {
        for px in x0..=x1 {
            let dx = px as f32 + 0.5 - cx;
            let dy = py as f32 + 0.5 - cy;
            if dx * dx + dy * dy <= r * r {
                put(buf, w, h, px, py, c);
            }
        }
    }
}

/// Anti-aliased filled circle with separate opacity. Blends `color`
/// (RGB) over the underlying pixel using `opacity ∈ [0,1]` and a 1-px
/// edge smoothing band, so dots read cleanly at small radii.
#[allow(clippy::too_many_arguments)]
fn fill_circle_aa(
    buf: &mut [u8],
    w: u32,
    h: u32,
    cx: f32,
    cy: f32,
    r: f32,
    color: [u8; 3],
    opacity: f32,
) {
    let x0 = (cx - r - 1.0).floor() as i32;
    let y0 = (cy - r - 1.0).floor() as i32;
    let x1 = (cx + r + 1.0).ceil() as i32;
    let y1 = (cy + r + 1.0).ceil() as i32;
    for py in y0..=y1 {
        for px in x0..=x1 {
            if px < 0 || py < 0 || px as u32 >= w || py as u32 >= h {
                continue;
            }
            let dx = px as f32 + 0.5 - cx;
            let dy = py as f32 + 0.5 - cy;
            let d = (dx * dx + dy * dy).sqrt();
            // Smooth edge over a 1-px band centred on r.
            let edge = (r - d + 0.5).clamp(0.0, 1.0);
            if edge <= 0.0 {
                continue;
            }
            let a = opacity * edge;
            let i = ((py as u32 * w + px as u32) * 4) as usize;
            let inv = 1.0 - a;
            buf[i] = (buf[i] as f32 * inv + color[0] as f32 * a) as u8;
            buf[i + 1] = (buf[i + 1] as f32 * inv + color[1] as f32 * a) as u8;
            buf[i + 2] = (buf[i + 2] as f32 * inv + color[2] as f32 * a) as u8;
        }
    }
}

fn inside_rounded(px: f32, py: f32, x: f32, y: f32, w: f32, h: f32, r: f32) -> bool {
    if px < x || px > x + w || py < y || py > y + h {
        return false;
    }
    if px >= x + r && px <= x + w - r {
        return true;
    }
    if py >= y + r && py <= y + h - r {
        return true;
    }
    let cx = px.clamp(x + r, x + w - r);
    let cy = py.clamp(y + r, y + h - r);
    let dx = px - cx;
    let dy = py - cy;
    dx * dx + dy * dy <= r * r
}

fn signed_dist_rounded(px: f32, py: f32, x: f32, y: f32, w: f32, h: f32, r: f32) -> f32 {
    let cx = px.clamp(x + r, x + w - r);
    let cy = py.clamp(y + r, y + h - r);
    let dx = px - cx;
    let dy = py - cy;
    let dist_to_corner_center = (dx * dx + dy * dy).sqrt();
    let outside = dist_to_corner_center - r;
    if px > x && px < x + w && py > y && py < y + h && outside < 0.0 {
        outside
    } else {
        outside.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dump four spinner phases to /tmp so the loader can be inspected
    /// without the full pipeline running. Ignored by default — run with:
    /// `cargo test -p lb-pipeline idle_loader::tests::dump -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn dump_phases_to_tmp() {
        let mut loader = IdleLoader::new(1280, 720);
        let mut frame = Vec::new();
        for phase in [0_u32, 2, 4, 6] {
            loader.phase = phase;
            loader.render(&mut frame);
            let img: image::ImageBuffer<image::Rgba<u8>, _> =
                image::ImageBuffer::from_raw(1280, 720, frame.clone()).unwrap();
            let path = format!("/tmp/lb-loader-phase-{phase}.png");
            img.save(&path).unwrap();
            println!("wrote {path}");
        }
    }
}
