use eframe::egui::{self, Color32, Stroke};
use lb_pipeline::PipelineState;

use super::App;
use crate::theme::{color, space};

impl App {
    pub(super) fn footer(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("footer")
            .exact_height(32.0)
            .frame(
                egui::Frame::none()
                    .fill(color::PANEL)
                    .stroke(Stroke::new(1.0, color::STROKE)),
            )
            .show(ctx, |ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.add_space(space::LG);
                    let (dot_color, label, label_color) =
                        footer_status(self.running(), &self.pipeline_state);
                    ui.label(egui::RichText::new("●").color(dot_color).small());
                    ui.add_space(space::XS);
                    ui.label(egui::RichText::new(label).small().color(label_color));
                    ui.add_space(space::MD);
                    sep(ui);
                    ui.add_space(space::MD);
                    ui.label(
                        egui::RichText::new(format!("in  {}", self.cfg.source_device))
                            .monospace()
                            .small()
                            .color(color::TEXT_MUTED),
                    );
                    ui.add_space(space::MD);
                    ui.label(
                        egui::RichText::new(format!("→ {}", self.cfg.sink_device))
                            .monospace()
                            .small()
                            .color(color::TEXT_MUTED),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(space::LG);
                        ui.label(
                            egui::RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                                .monospace()
                                .small()
                                .color(color::TEXT_MUTED),
                        );
                    });
                });
            });
    }
}

/// Compute the footer status indicator for the current pipeline state.
/// Returns `(dot_color, label, label_color)`.
///
/// Three layers of nuance:
/// - pipeline not running at all → grey "Idle".
/// - running but lazy state is Idle (no consumer) → blue "Standby" so
///   the user sees the difference between "the app is on but the camera
///   is off" and "the app is off".
/// - running and Live → green "LIVE → name(pid)" if there's a real
///   consumer, otherwise green "LIVE (preview)" (the GUI's preview
///   heartbeat is the demand source).
fn footer_status(running: bool, state: &PipelineState) -> (Color32, String, Color32) {
    if !running {
        return (color::TEXT_MUTED, "Idle".to_string(), color::TEXT_MUTED);
    }
    match state {
        PipelineState::Live { consumers } if !consumers.is_empty() => {
            // Show the first consumer; if there are several, append a count.
            let first = &consumers[0];
            let label = if consumers.len() == 1 {
                format!("LIVE → {} ({})", first.name, first.pid)
            } else {
                format!(
                    "LIVE → {} ({}) + {}",
                    first.name,
                    first.pid,
                    consumers.len() - 1
                )
            };
            (color::SUCCESS, label, color::SUCCESS)
        }
        PipelineState::Live { .. } => {
            // Live without external consumers: the GUI preview heartbeat
            // is the demand source.
            (color::SUCCESS, "LIVE (preview)".to_string(), color::SUCCESS)
        }
        PipelineState::Idle => (
            color::ACCENT,
            "Standby (no consumer)".to_string(),
            color::ACCENT,
        ),
    }
}

fn sep(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(1.0, 14.0), egui::Sense::hover());
    ui.painter().vline(
        rect.center().x,
        rect.y_range(),
        Stroke::new(1.0, color::STROKE_STRONG),
    );
}
