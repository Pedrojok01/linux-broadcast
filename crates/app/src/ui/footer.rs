use eframe::egui::{self, Color32, Stroke};
use lb_pipeline::{Backend, PipelineState};

use super::App;
use crate::theme::{color, space};

impl App {
    pub(super) fn footer(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom("footer")
            .exact_size(32.0)
            .frame(
                egui::Frame::NONE
                    .fill(color::PANEL)
                    .stroke(Stroke::new(1.0, color::STROKE)),
            )
            .show(ui, |ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.add_space(space::LG);
                    let (dot_color, label, label_color) = footer_status(
                        self.running(),
                        &self.pipeline_state,
                        self.last_gui_preview_active == Some(true),
                    );
                    ui.label(egui::RichText::new("●").color(dot_color).small());
                    ui.add_space(space::XS);
                    ui.label(egui::RichText::new(label).small().color(label_color));
                    if self.backend == Backend::Gpu {
                        ui.add_space(space::SM);
                        gpu_badge(ui);
                    } else if let Some(reason) = &self.gpu_fallback_reason {
                        ui.add_space(space::SM);
                        gpu_fallback_badge(ui, reason);
                    }
                    ui.add_space(space::MD);
                    sep(ui);
                    ui.add_space(space::MD);
                    // Append the camera's native capture format while Live so
                    // the footer shows what the camera actually delivers (e.g.
                    // "MJPEG 1280×720"), not just the device path.
                    let capture_suffix = match &self.pipeline_state {
                        PipelineState::Live {
                            capture: Some(c), ..
                        } => format!(" · {} {}×{}", c.codec.label(), c.width, c.height),
                        _ => String::new(),
                    };
                    ui.label(
                        egui::RichText::new(format!(
                            "in  {}{}",
                            self.cfg.source_device, capture_suffix
                        ))
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
///   consumer, "LIVE (preview)" if the GUI's preview heartbeat is the
///   demand source, else "LIVE (releasing)": the deactivation grace period.
fn footer_status(
    running: bool,
    state: &PipelineState,
    preview: bool,
) -> (Color32, String, Color32) {
    if !running {
        return (color::TEXT_MUTED, "Idle".to_string(), color::TEXT_MUTED);
    }
    match state {
        PipelineState::Live { consumers, .. } if !consumers.is_empty() => {
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
        PipelineState::Live { .. } if preview => {
            (color::SUCCESS, "LIVE (preview)".to_string(), color::SUCCESS)
        }
        PipelineState::Live { .. } => (
            color::SUCCESS,
            "LIVE (releasing)".to_string(),
            color::SUCCESS,
        ),
        PipelineState::Idle => (
            color::ACCENT,
            "Standby (no consumer)".to_string(),
            color::ACCENT,
        ),
    }
}

/// Small accent-colored chip shown in the footer when the segmenter runs
/// on the GPU. Renders nothing on CPU (caller gates on `Backend::Gpu`).
fn gpu_badge(ui: &mut egui::Ui) {
    egui::Frame::NONE
        .fill(color::ACCENT_SOFT)
        .stroke(Stroke::new(1.0, color::ACCENT))
        .corner_radius(4)
        .inner_margin(egui::Margin::symmetric(6, 1))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new("GPU")
                    .monospace()
                    .small()
                    .strong()
                    .color(color::ACCENT),
            );
        });
}

/// Persistent, visible notice that CUDA started but failed on a real model
/// kernel and the pipeline recovered on CPU. The technical error stays in the
/// hover text so the compact footer remains readable.
fn gpu_fallback_badge(ui: &mut egui::Ui, reason: &str) {
    egui::Frame::NONE
        .fill(color::DANGER_SOFT)
        .stroke(Stroke::new(1.0, color::DANGER))
        .corner_radius(4)
        .inner_margin(egui::Margin::symmetric(6, 1))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new("GPU unavailable · CPU")
                    .monospace()
                    .small()
                    .strong()
                    .color(color::DANGER),
            )
            .on_hover_text(reason);
        });
}

fn sep(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(1.0, 14.0), egui::Sense::hover());
    ui.painter().vline(
        rect.center().x,
        rect.y_range(),
        Stroke::new(1.0, color::STROKE_STRONG),
    );
}
