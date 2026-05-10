use eframe::egui::{self, Color32, Margin, Rounding, Stroke};

use super::App;
use super::widgets::{floating_pill, pill};
use crate::theme::{color, radius};

impl App {
    pub(super) fn preview_pane(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(
                egui::Frame::none()
                    .fill(color::BG)
                    .inner_margin(Margin::symmetric(20.0, 18.0)),
            )
            .show(ctx, |ui| {
                ui.spacing_mut().item_spacing.y = 14.0;

                // Header row: "Preview" + status pills
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Preview")
                            .size(13.0)
                            .strong()
                            .color(color::TEXT_WEAK),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        pill(
                            ui,
                            &format!("→ {}", self.cfg.sink_device),
                            color::TEXT_WEAK,
                            None,
                        );
                        pill(
                            ui,
                            &format!(
                                "{}×{} · {} fps",
                                self.cfg.width, self.cfg.height, self.cfg.framerate
                            ),
                            color::TEXT_WEAK,
                            None,
                        );
                        if self.running() {
                            pill(ui, "LIVE", color::TEXT, Some(color::DANGER));
                        }
                    });
                });

                // Preview surface
                let avail = ui.available_size_before_wrap();
                let (rect, _) = ui.allocate_exact_size(avail, egui::Sense::hover());
                let p = ui.painter();
                p.rect_filled(
                    rect,
                    Rounding::same(radius::LG),
                    color::PANEL_INSET, // letterbox
                );
                p.rect_stroke(
                    rect,
                    Rounding::same(radius::LG),
                    Stroke::new(1.0, color::STROKE),
                );

                if !self.cfg.show_preview {
                    p.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "Preview hidden",
                        egui::FontId::proportional(14.0),
                        color::TEXT_MUTED,
                    );
                } else if let Some(tex) = &self.preview_tex {
                    let [tw, th] = self.last_preview_size.unwrap_or([1280, 720]);
                    let aspect = tw as f32 / th as f32;
                    let mut w = rect.width();
                    let mut h = w / aspect;
                    if h > rect.height() {
                        h = rect.height();
                        w = h * aspect;
                    }
                    let centered = egui::Rect::from_center_size(rect.center(), egui::vec2(w, h));
                    let mut mesh = egui::Mesh::with_texture(tex.id());
                    mesh.add_rect_with_uv(
                        centered,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        Color32::WHITE,
                    );
                    p.add(egui::Shape::mesh(mesh));
                } else {
                    let msg = if self.running() {
                        "Waiting for first frame…"
                    } else {
                        "Press Start to begin streaming."
                    };
                    p.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        msg,
                        egui::FontId::proportional(14.0),
                        color::TEXT_MUTED,
                    );
                }

                // Hovering controls along the preview's bottom edge.
                if self.running() {
                    let pad = 12.0;
                    let bottom_rect = egui::Rect::from_min_max(
                        rect.left_bottom() + egui::vec2(pad, -36.0),
                        rect.right_bottom() + egui::vec2(-pad, -pad),
                    );
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(bottom_rect), |ui| {
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                            floating_pill(
                                ui,
                                &format!("● Sending to {}", self.cfg.sink_device),
                                color::DANGER,
                            );
                        });
                    });
                }
            });
    }
}
