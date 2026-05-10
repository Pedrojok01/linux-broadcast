use eframe::egui::{self, Rounding, Stroke, ViewportCommand};

use super::App;
use super::widgets::ghost_button;
use crate::theme::{color, space};

impl App {
    pub(super) fn header(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("header")
            .exact_height(56.0)
            .frame(
                egui::Frame::none()
                    .fill(color::PANEL)
                    .stroke(Stroke::new(1.0, color::STROKE)),
            )
            .show(ctx, |ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.add_space(space::LG);
                    // Brand mark — frame-within-a-frame logo
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(28.0, 28.0), egui::Sense::hover());
                    let p = ui.painter();
                    // outer rounded square
                    p.rect(
                        rect.shrink(2.0),
                        Rounding::same(7.0),
                        color::PANEL_INSET,
                        Stroke::new(1.25, color::STROKE_STRONG),
                    );
                    // inner square
                    p.rect_stroke(
                        egui::Rect::from_center_size(rect.center(), egui::vec2(14.0, 14.0)),
                        Rounding::same(3.5),
                        Stroke::new(2.0, color::TEXT),
                    );
                    // accent dot
                    p.circle_filled(rect.right_top() + egui::vec2(-7.0, 7.0), 2.4, color::ACCENT);

                    ui.add_space(space::MD);
                    ui.label(
                        egui::RichText::new("LinuxBroadcast")
                            .strong()
                            .size(14.0)
                            .color(color::TEXT),
                    );
                    ui.label(
                        egui::RichText::new("· virtual webcam")
                            .size(13.0)
                            .color(color::TEXT_WEAK),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(space::LG);
                        // No tray host (or it failed to install) → keep a
                        // reachable Quit button in the header so the user
                        // is never stranded.
                        if self.tray.is_none() && ghost_button(ui, "Quit").clicked() {
                            self.quit_requested = true;
                            ctx.send_viewport_cmd(ViewportCommand::Close);
                        }
                        if let Some(err) = &self.error {
                            let err = err.clone();
                            ui.label(egui::RichText::new(err).small().color(color::DANGER));
                        }
                    });
                });
            });
    }
}
