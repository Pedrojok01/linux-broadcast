//! Design tokens (colors, spacing, radii, control sizes) applied to egui.

use eframe::egui::{
    self, Color32, FontData, FontDefinitions, FontFamily, FontId, Margin, Rounding, Stroke, Style,
    TextStyle, Visuals,
};

const INTER: &[u8] = include_bytes!("../../../assets/fonts/Inter-Variable.ttf");
const JETBRAINS_MONO: &[u8] = include_bytes!("../../../assets/fonts/JetBrainsMono-Regular.ttf");

/// Token palette — see `DESIGN.md` for the colour rationale.
pub mod color {
    use super::Color32;
    pub const BG: Color32 = Color32::from_rgb(0x0B, 0x0E, 0x13);
    pub const PANEL: Color32 = Color32::from_rgb(0x11, 0x15, 0x1C);
    pub const PANEL_ALT: Color32 = Color32::from_rgb(0x16, 0x1B, 0x23);
    pub const PANEL_INSET: Color32 = Color32::from_rgb(0x0D, 0x11, 0x17);

    pub const STROKE: Color32 = Color32::from_rgb(0x22, 0x29, 0x34);
    pub const STROKE_STRONG: Color32 = Color32::from_rgb(0x2E, 0x37, 0x44);

    pub const TEXT: Color32 = Color32::from_rgb(0xE6, 0xEA, 0xF0);
    pub const TEXT_WEAK: Color32 = Color32::from_rgb(0x9A, 0xA4, 0xB2);
    pub const TEXT_MUTED: Color32 = Color32::from_rgb(0x6B, 0x75, 0x85);

    pub const ACCENT: Color32 = Color32::from_rgb(0x5B, 0xD4, 0xC0);
    pub const ACCENT_SOFT: Color32 = Color32::from_rgba_premultiplied(0x12, 0x2A, 0x26, 0x22);

    pub const DANGER: Color32 = Color32::from_rgb(0xE5, 0x68, 0x5A);
    pub const DANGER_SOFT: Color32 = Color32::from_rgba_premultiplied(0x2A, 0x14, 0x12, 0x22);
    pub const SUCCESS: Color32 = Color32::from_rgb(0x7F, 0xCB, 0x8E);
}

/// Token spacing — see `DESIGN.md` for the scale.
pub mod space {
    pub const XS: f32 = 4.0;
    pub const SM: f32 = 8.0;
    pub const MD: f32 = 12.0;
    pub const LG: f32 = 16.0;
    pub const PANEL_PAD_Y: f32 = 14.0;
    pub const SECTION_GAP: f32 = 18.0;
}

pub mod radius {
    pub const SM: f32 = 4.0;
    pub const MD: f32 = 8.0;
    pub const LG: f32 = 12.0;
}

pub mod control {
    pub const PRIMARY_HEIGHT: f32 = 40.0;
    pub const THUMB_RADIUS: f32 = 6.0;
}

pub fn apply(ctx: &egui::Context) {
    install_fonts(ctx);
    install_style(ctx);
}

fn install_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    fonts
        .font_data
        .insert("inter".into(), FontData::from_static(INTER));
    fonts
        .font_data
        .insert("jbmono".into(), FontData::from_static(JETBRAINS_MONO));
    fonts
        .families
        .get_mut(&FontFamily::Proportional)
        .unwrap()
        .insert(0, "inter".into());
    fonts
        .families
        .get_mut(&FontFamily::Monospace)
        .unwrap()
        .insert(0, "jbmono".into());
    ctx.set_fonts(fonts);
}

fn install_style(ctx: &egui::Context) {
    let mut style: Style = (*ctx.style()).clone();

    style.text_styles = [
        (
            TextStyle::Heading,
            FontId::new(20.0, FontFamily::Proportional),
        ),
        (TextStyle::Body, FontId::new(13.0, FontFamily::Proportional)),
        (
            TextStyle::Button,
            FontId::new(13.0, FontFamily::Proportional),
        ),
        (
            TextStyle::Small,
            FontId::new(11.0, FontFamily::Proportional),
        ),
        (
            TextStyle::Monospace,
            FontId::new(11.0, FontFamily::Monospace),
        ),
    ]
    .into();

    style.spacing.item_spacing = egui::vec2(space::SM, space::SM);
    style.spacing.button_padding = egui::vec2(space::MD, 8.0);
    style.spacing.window_margin = Margin::symmetric(space::LG, space::PANEL_PAD_Y);
    style.spacing.menu_margin = Margin::same(space::SM);
    style.spacing.combo_height = 240.0;
    style.spacing.scroll.bar_width = 8.0;
    style.spacing.icon_width = 14.0;
    style.spacing.icon_width_inner = 8.0;

    let mut v = Visuals::dark();

    v.window_fill = color::BG;
    v.panel_fill = color::PANEL;
    v.extreme_bg_color = color::PANEL_INSET;
    v.faint_bg_color = color::PANEL_ALT;
    v.code_bg_color = color::PANEL_INSET;

    v.window_stroke = Stroke::new(1.0, color::STROKE);
    v.menu_rounding = Rounding::same(radius::MD);
    v.window_rounding = Rounding::same(radius::LG);

    v.override_text_color = Some(color::TEXT);
    v.hyperlink_color = color::ACCENT;

    v.selection.bg_fill = Color32::from_rgba_premultiplied(0x1F, 0x46, 0x40, 0x99);
    v.selection.stroke = Stroke::new(1.5, color::ACCENT);

    // Widget styles — keep all four states consistent.
    let widget_round = Rounding::same(radius::MD);

    v.widgets.noninteractive.bg_fill = color::PANEL;
    v.widgets.noninteractive.weak_bg_fill = color::PANEL;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, color::STROKE);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, color::TEXT);
    v.widgets.noninteractive.rounding = widget_round;
    v.widgets.noninteractive.expansion = 0.0;

    v.widgets.inactive.bg_fill = color::PANEL_INSET;
    v.widgets.inactive.weak_bg_fill = color::PANEL_INSET;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, color::STROKE);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, color::TEXT);
    v.widgets.inactive.rounding = widget_round;
    v.widgets.inactive.expansion = 0.0;

    v.widgets.hovered.bg_fill = color::PANEL_ALT;
    v.widgets.hovered.weak_bg_fill = color::PANEL_ALT;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, color::STROKE_STRONG);
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, color::TEXT);
    v.widgets.hovered.rounding = widget_round;
    v.widgets.hovered.expansion = 0.0;

    v.widgets.active.bg_fill = color::PANEL_ALT;
    v.widgets.active.weak_bg_fill = color::PANEL_ALT;
    v.widgets.active.bg_stroke = Stroke::new(1.5, color::ACCENT);
    v.widgets.active.fg_stroke = Stroke::new(1.0, color::TEXT);
    v.widgets.active.rounding = widget_round;
    v.widgets.active.expansion = 0.0;

    v.widgets.open.bg_fill = color::PANEL_ALT;
    v.widgets.open.weak_bg_fill = color::PANEL_ALT;
    v.widgets.open.bg_stroke = Stroke::new(1.0, color::STROKE_STRONG);
    v.widgets.open.fg_stroke = Stroke::new(1.0, color::TEXT);
    v.widgets.open.rounding = widget_round;

    v.dark_mode = true;

    style.visuals = v;
    ctx.set_style(style);
}

/// A small uppercase section caption (Camera, Library, …).
pub fn section_caption(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text.to_uppercase())
            .small()
            .strong()
            .color(color::TEXT_MUTED)
            .extra_letter_spacing(1.2),
    );
}
