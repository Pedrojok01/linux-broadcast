//! Stateless drawing primitives used by multiple sidebar/header sections.

use eframe::egui::{self, Color32, Rounding, Stroke};

use crate::theme::{color, radius};

pub(super) fn pill(ui: &mut egui::Ui, text: &str, fg: Color32, bg_override: Option<Color32>) {
    let font = egui::FontId::monospace(10.5);
    let galley = ui.fonts(|f| f.layout_no_wrap(text.to_string(), font.clone(), fg));
    let pad = egui::vec2(10.0, 6.0);
    let has_dot = bg_override == Some(color::DANGER);
    // 14 px = dot (6 dia) + 8 px gap before the glyphs.
    let extra_left = if has_dot { 14.0 } else { 0.0 };
    let size = galley.size() + pad * 2.0 + egui::vec2(extra_left, 0.0);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let p = ui.painter();
    let bg = bg_override.unwrap_or(color::PANEL_ALT);
    p.rect(
        rect,
        Rounding::same(999.0),
        bg,
        Stroke::new(1.0, color::STROKE),
    );
    if has_dot {
        let dot = rect.left_center() + egui::vec2(10.0, 0.0);
        p.circle_filled(dot, 3.0, Color32::WHITE);
        p.text(
            rect.left_center() + egui::vec2(10.0 + extra_left, 0.0),
            egui::Align2::LEFT_CENTER,
            text,
            font,
            fg,
        );
    } else {
        p.text(rect.center(), egui::Align2::CENTER_CENTER, text, font, fg);
    }
    ui.add_space(6.0);
}

pub(super) fn floating_pill(ui: &mut egui::Ui, text: &str, accent: Color32) {
    let font = egui::FontId::monospace(10.5);
    let galley = ui.fonts(|f| f.layout_no_wrap(text.to_string(), font.clone(), color::TEXT));
    let pad = egui::vec2(10.0, 6.0);
    let (rect, _) = ui.allocate_exact_size(galley.size() + pad * 2.0, egui::Sense::hover());
    let p = ui.painter();
    p.rect(
        rect,
        Rounding::same(999.0),
        Color32::from_rgba_premultiplied(0x08, 0x0A, 0x0E, 0xC0),
        Stroke::new(1.0, color::STROKE_STRONG),
    );
    let dot = rect.left_center() + egui::vec2(10.0, 0.0);
    p.circle_filled(dot, 3.0, accent);
    p.text(
        rect.left_center() + egui::vec2(20.0, 0.0),
        egui::Align2::LEFT_CENTER,
        text.trim_start_matches("● "),
        font,
        color::TEXT,
    );
}

pub(super) fn ghost_button(ui: &mut egui::Ui, text: &str) -> egui::Response {
    let font = egui::FontId::proportional(11.0);
    let galley = ui.fonts(|f| f.layout_no_wrap(text.to_string(), font.clone(), color::TEXT_WEAK));
    let pad = egui::vec2(10.0, 6.0);
    let (rect, resp) = ui.allocate_exact_size(galley.size() + pad * 2.0, egui::Sense::click());
    let p = ui.painter();
    let (fill, stroke_color, text_color) = if resp.hovered() {
        (color::PANEL_ALT, color::STROKE_STRONG, color::TEXT)
    } else {
        (Color32::TRANSPARENT, color::STROKE, color::TEXT_WEAK)
    };
    p.rect(
        rect,
        Rounding::same(radius::SM),
        fill,
        Stroke::new(1.0, stroke_color),
    );
    p.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        text,
        font,
        text_color,
    );
    resp
}

/// Settings-section toggle row: title + subtitle on the left, pill
/// switch on the right. Returns `true` on click so the caller can flip
/// the underlying value and run any side effects.
pub(super) fn toggle_row(
    ui: &mut egui::Ui,
    active: bool,
    title: &str,
    subtitle: &str,
) -> bool {
    let row_h = 36.0;
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), row_h),
        egui::Sense::click(),
    );
    let p = ui.painter();
    let stroke_color = if active {
        color::ACCENT.linear_multiply(0.6)
    } else if resp.hovered() {
        color::STROKE_STRONG
    } else {
        color::STROKE
    };
    p.rect(
        rect,
        Rounding::same(radius::MD),
        color::PANEL_INSET,
        Stroke::new(1.0, stroke_color),
    );
    p.text(
        rect.left_center() + egui::vec2(12.0, -6.0),
        egui::Align2::LEFT_CENTER,
        title,
        egui::FontId::proportional(13.0),
        color::TEXT,
    );
    p.text(
        rect.left_center() + egui::vec2(12.0, 8.0),
        egui::Align2::LEFT_CENTER,
        subtitle,
        egui::FontId::proportional(10.5),
        color::TEXT_MUTED,
    );
    let switch_w = 32.0;
    let switch_h = 18.0;
    let switch_rect = egui::Rect::from_center_size(
        egui::pos2(rect.right() - 12.0 - switch_w / 2.0, rect.center().y),
        egui::vec2(switch_w, switch_h),
    );
    let track_color = if active {
        color::ACCENT
    } else {
        color::STROKE_STRONG
    };
    p.rect_filled(switch_rect, Rounding::same(switch_h / 2.0), track_color);
    let knob_x = if active {
        switch_rect.right() - switch_h / 2.0
    } else {
        switch_rect.left() + switch_h / 2.0
    };
    p.circle_filled(
        egui::pos2(knob_x, switch_rect.center().y),
        switch_h / 2.0 - 2.0,
        color::TEXT,
    );
    resp.clicked()
}

pub(super) fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max - 1).collect();
        out.push('…');
        out
    }
}

pub(super) fn draw_dashed_rect(p: &egui::Painter, rect: egui::Rect, stroke: Stroke) {
    let dash = 4.0;
    let gap = 3.0;
    let edges = [
        (rect.left_top(), rect.right_top()),
        (rect.right_top(), rect.right_bottom()),
        (rect.right_bottom(), rect.left_bottom()),
        (rect.left_bottom(), rect.left_top()),
    ];
    for (a, b) in edges {
        let dir = (b - a).normalized();
        let len = (b - a).length();
        let mut t = 0.0;
        while t < len {
            let s = a + dir * t;
            let e = a + dir * (t + dash).min(len);
            p.line_segment([s, e], stroke);
            t += dash + gap;
        }
    }
}
