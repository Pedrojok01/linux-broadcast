use anyhow::anyhow;
use eframe::egui::{self, Color32, Margin, Rounding, Stroke};
use lb_pipeline::Command;
use std::path::{Path, PathBuf};

use super::App;
use super::widgets::{draw_dashed_rect, toggle_row, truncate};
use crate::autostart;
use crate::backgrounds::{self, LibraryEntry};
use crate::config::{Mode, Model};
use crate::theme::{self, color, control, radius, space};

impl App {
    pub(super) fn sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("sidebar")
            .exact_width(320.0)
            .resizable(false)
            .frame(
                egui::Frame::none()
                    .fill(color::PANEL)
                    .stroke(Stroke::new(1.0, color::STROKE))
                    .inner_margin(Margin::symmetric(space::LG, space::PANEL_PAD_Y)),
            )
            .show(ctx, |ui| {
                // Reserve a fixed bottom strip for the primary action so it
                // can never be pushed off-screen by tall sections (Library,
                // Settings, …). The remaining top region holds the sections
                // inside a ScrollArea so short windows stay usable.
                let panel_rect = ui.max_rect();
                let action_h = control::PRIMARY_HEIGHT;
                let action_rect = egui::Rect::from_min_max(
                    egui::pos2(panel_rect.min.x, panel_rect.max.y - action_h),
                    panel_rect.max,
                );
                let scroll_rect = egui::Rect::from_min_max(
                    panel_rect.min,
                    egui::pos2(panel_rect.max.x, action_rect.min.y - space::MD),
                );

                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(scroll_rect), |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false; 2])
                        .show(ui, |ui| {
                            ui.spacing_mut().item_spacing.y = space::SECTION_GAP;
                            ui.style_mut().spacing.item_spacing.y = space::SECTION_GAP;
                            self.sidebar_camera(ui);
                            self.sidebar_model(ui);
                            self.sidebar_scene(ui);
                            self.sidebar_settings(ui);
                        });
                });

                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(action_rect), |ui| {
                    self.primary_action(ui);
                });
            });
    }

    fn sidebar_camera(&mut self, ui: &mut egui::Ui) {
        theme::section_caption(ui, "Camera");
        ui.add_space(space::SM);

        let current = self
            .cameras
            .iter()
            .find(|c| c.path == self.cfg.source_device)
            .cloned();

        // Custom-rendered "device select" row.
        let row_h = 44.0;
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), row_h),
            egui::Sense::click(),
        );
        let p = ui.painter();
        let stroke = if resp.hovered() {
            color::STROKE_STRONG
        } else {
            color::STROKE
        };
        p.rect(
            rect,
            Rounding::same(radius::MD),
            color::PANEL_INSET,
            Stroke::new(1.0, stroke),
        );
        // status dot
        let dot = rect.left_center() + egui::vec2(12.0, 0.0);
        p.circle_filled(dot, 3.5, color::SUCCESS);
        p.circle_stroke(
            dot,
            6.0,
            Stroke::new(1.0, color::SUCCESS.linear_multiply(0.25)),
        );
        // label
        let label = current
            .as_ref()
            .map(|c| c.label.clone())
            .unwrap_or_else(|| self.cfg.source_device.clone());
        let detail = format!(
            "{} · {}×{} · {} fps",
            self.cfg.source_device, self.cfg.width, self.cfg.height, self.cfg.framerate
        );
        p.text(
            rect.left_top() + egui::vec2(28.0, 8.0),
            egui::Align2::LEFT_TOP,
            truncate(&label, 28),
            egui::FontId::proportional(13.0),
            color::TEXT,
        );
        p.text(
            rect.left_top() + egui::vec2(28.0, 25.0),
            egui::Align2::LEFT_TOP,
            truncate(&detail, 36),
            egui::FontId::monospace(10.5),
            color::TEXT_MUTED,
        );
        // chevron
        let cx = rect.right_center() - egui::vec2(14.0, 0.0);
        for i in 0..2 {
            let dx = (i as f32 - 0.5) * 5.0;
            p.line_segment(
                [
                    cx + egui::vec2(dx, -2.5),
                    cx + egui::vec2(dx + 2.5 * (1.0 - 2.0 * i as f32), 1.5),
                ],
                Stroke::new(1.5, color::TEXT_MUTED),
            );
        }

        let popup_id = ui.make_persistent_id("camera_popup");
        if resp.clicked() {
            ui.memory_mut(|m| m.toggle_popup(popup_id));
        }
        let mut camera_changed = false;
        egui::popup_below_widget(
            ui,
            popup_id,
            &resp,
            egui::PopupCloseBehavior::CloseOnClick,
            |ui| {
                ui.set_min_width(rect.width());
                ui.set_max_height(280.0);
                for cam in &self.cameras {
                    let selected = cam.path == self.cfg.source_device;
                    let resp = ui.selectable_label(selected, &cam.label);
                    if resp.clicked() && !selected {
                        self.cfg.source_device = cam.path.clone();
                        camera_changed = true;
                    }
                }
            },
        );

        if camera_changed {
            self.save_settings();
            if self.running() {
                self.restart_pipeline();
            }
        }
    }

    fn sidebar_model(&mut self, ui: &mut egui::Ui) {
        theme::section_caption(ui, "Model");
        ui.add_space(space::SM);
        let current_label = self.cfg.model.label();
        let mut changed = false;
        egui::ComboBox::from_id_salt("model_combo")
            .width(ui.available_width())
            .selected_text(current_label)
            .show_ui(ui, |ui| {
                for &m in Model::ALL {
                    if ui
                        .selectable_label(self.cfg.model == m, m.label())
                        .clicked()
                        && self.cfg.model != m
                    {
                        self.cfg.model = m;
                        changed = true;
                    }
                }
            });
        if changed {
            self.save_settings();
            // The segmenter is built at pipeline start, so a model swap
            // requires a restart of the GStreamer graph.
            if self.running() {
                self.restart_pipeline();
            }
        }
    }

    fn sidebar_scene(&mut self, ui: &mut egui::Ui) {
        // Mode tabs (no caption, no separate import button — Library has its
        // own dashed Import tile, and the tabs themselves announce the
        // section's purpose).
        let mut mode_changed = false;
        let tab_gap: f32 = 6.0;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = tab_gap;
            let total_w = ui.available_width();
            let cell_w = (total_w - tab_gap * 2.0) / 3.0;
            for (label, mode) in [
                ("None", Mode::None),
                ("Blur", Mode::Blur),
                ("Replace", Mode::Replace),
            ] {
                let active = self.cfg.mode == mode;
                let (rect, resp) =
                    ui.allocate_exact_size(egui::vec2(cell_w, 30.0), egui::Sense::click());
                let p = ui.painter();
                let bg = if active {
                    color::PANEL_ALT
                } else if resp.hovered() {
                    color::PANEL_ALT.linear_multiply(0.5)
                } else {
                    color::PANEL_INSET
                };
                let stroke = if active && mode == Mode::Blur {
                    Stroke::new(1.0, color::ACCENT.linear_multiply(0.6))
                } else if active {
                    Stroke::new(1.0, color::STROKE_STRONG)
                } else {
                    Stroke::new(1.0, color::STROKE)
                };
                p.rect(rect, Rounding::same(radius::SM), bg, stroke);
                let text_color = if active && mode == Mode::Blur {
                    color::ACCENT
                } else if active {
                    color::TEXT
                } else {
                    color::TEXT_WEAK
                };
                p.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    label,
                    egui::FontId::proportional(12.0),
                    text_color,
                );
                if resp.clicked() && !active {
                    self.cfg.mode = mode;
                    mode_changed = true;
                }
            }
        });
        ui.add_space(space::SM);

        // Blur intensity slider — shown only in Blur mode (else it's confusing)
        if self.cfg.mode == Mode::Blur {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Blur intensity")
                        .small()
                        .color(color::TEXT_WEAK),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "{}%",
                            (self.cfg.blur_strength * 100.0).round() as i32
                        ))
                        .monospace()
                        .small()
                        .color(color::TEXT),
                    );
                });
            });
            // slim track + accent fill
            let (track_rect, resp) = ui.allocate_exact_size(
                egui::vec2(ui.available_width(), 14.0),
                egui::Sense::click_and_drag(),
            );
            let p = ui.painter();
            let track_y = track_rect.center().y;
            let track_l = track_rect.left() + 1.0;
            let track_r = track_rect.right() - 1.0;
            let bar = egui::Rect::from_min_max(
                egui::pos2(track_l, track_y - 2.0),
                egui::pos2(track_r, track_y + 2.0),
            );
            p.rect(
                bar,
                Rounding::same(2.0),
                color::PANEL_INSET,
                Stroke::new(1.0, color::STROKE),
            );
            // fill
            let fill_w = (track_r - track_l) * self.cfg.blur_strength;
            let fill = egui::Rect::from_min_max(
                egui::pos2(track_l, track_y - 2.0),
                egui::pos2(track_l + fill_w, track_y + 2.0),
            );
            p.rect_filled(fill, Rounding::same(2.0), color::ACCENT);
            // handle
            let handle_x = track_l + fill_w;
            p.circle_filled(egui::pos2(handle_x, track_y), 7.0, color::TEXT);
            p.circle_stroke(
                egui::pos2(handle_x, track_y),
                7.0,
                Stroke::new(2.0, color::ACCENT),
            );

            // drag interaction
            if let Some(pos) = resp.interact_pointer_pos() {
                let t = ((pos.x - track_l) / (track_r - track_l)).clamp(0.0, 1.0);
                if (t - self.cfg.blur_strength).abs() > 0.005 {
                    self.cfg.blur_strength = t;
                    mode_changed = true;
                }
            }
            ui.add_space(space::SM);
        }

        // Library grid (Replace mode only — owns the Import affordance).
        if self.cfg.mode == Mode::Replace {
            ui.add_space(space::XS);
            theme::section_caption(ui, "Library");
            ui.add_space(space::SM);

            let mut to_select: Option<PathBuf> = None;
            let mut to_remove: Option<PathBuf> = None;

            let library = self.library.clone();
            let active_path = self.cfg.background_path.clone();
            let cols = 3usize;
            let avail_w = ui.available_width();
            let gap = 8.0;
            let cell = ((avail_w - gap * (cols as f32 - 1.0)) / cols as f32).floor();
            let ctx = ui.ctx().clone();

            egui::ScrollArea::vertical()
                .max_height(220.0)
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(gap, gap);
                    let mut row: Vec<&LibraryEntry> = Vec::with_capacity(cols);
                    for entry in &library {
                        row.push(entry);
                        if row.len() == cols {
                            let r = std::mem::take(&mut row);
                            self.draw_library_row(
                                ui,
                                &ctx,
                                &r,
                                cell,
                                active_path.as_deref(),
                                &mut to_select,
                                &mut to_remove,
                            );
                        }
                    }
                    // Trailing partial row always gets a + Import tile —
                    // even when the library is empty, draw_library_row()
                    // will append it.
                    self.draw_library_row(
                        ui,
                        &ctx,
                        &row,
                        cell,
                        active_path.as_deref(),
                        &mut to_select,
                        &mut to_remove,
                    );
                });

            if let Some(p) = to_select {
                self.cfg.background_path = Some(p);
                self.cfg.mode = Mode::Replace;
                mode_changed = true;
            }
            if let Some(p) = to_remove {
                if let Err(e) = backgrounds::remove(&p) {
                    log::warn!("remove: {e:#}");
                }
                if self.cfg.background_path.as_deref() == Some(&p) {
                    self.cfg.background_path = None;
                    if self.cfg.mode == Mode::Replace {
                        self.cfg.mode = Mode::Blur;
                        mode_changed = true;
                    }
                }
                self.refresh_library();
            }
        }

        if mode_changed {
            self.save_settings();
            if self.running() {
                self.update_background_live();
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_library_row(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        row: &[&LibraryEntry],
        cell: f32,
        active_path: Option<&Path>,
        to_select: &mut Option<PathBuf>,
        to_remove: &mut Option<PathBuf>,
    ) {
        ui.horizontal(|ui| {
            for entry in row {
                let is_active = active_path == Some(&entry.path);
                let tex = self.thumbnail_for(ctx, &entry.path);
                let (rect, resp) =
                    ui.allocate_exact_size(egui::vec2(cell, cell), egui::Sense::click());
                let p = ui.painter();
                let stroke_color = if is_active {
                    color::ACCENT
                } else if resp.hovered() {
                    color::STROKE_STRONG
                } else {
                    color::STROKE
                };
                p.rect(
                    rect,
                    Rounding::same(control::THUMB_RADIUS),
                    color::PANEL_INSET,
                    Stroke::new(if is_active { 2.0 } else { 1.0 }, stroke_color),
                );
                if let Some(t) = &tex {
                    let mut mesh = egui::Mesh::with_texture(t.id());
                    let inner = rect.shrink(2.0);
                    mesh.add_rect_with_uv(
                        inner,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        Color32::WHITE,
                    );
                    p.add(egui::Shape::mesh(mesh));
                }
                // label badge
                let badge_h = 16.0;
                let badge = egui::Rect::from_min_max(
                    rect.left_bottom() + egui::vec2(4.0, -badge_h - 4.0),
                    rect.left_bottom() + egui::vec2(cell - 4.0, -4.0),
                );
                p.rect_filled(
                    badge,
                    Rounding::same(3.0),
                    Color32::from_rgba_premultiplied(0x06, 0x09, 0x0D, 0xC8),
                );
                p.text(
                    badge.left_center() + egui::vec2(6.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    truncate(&entry.label, 12),
                    egui::FontId::proportional(10.5),
                    color::TEXT,
                );
                // active check mark
                if is_active {
                    let center = rect.right_top() + egui::vec2(-10.0, 10.0);
                    p.circle_filled(center, 7.0, color::ACCENT);
                    p.text(
                        center,
                        egui::Align2::CENTER_CENTER,
                        "✓",
                        egui::FontId::proportional(11.0),
                        Color32::from_rgb(0x06, 0x18, 0x0F),
                    );
                }
                // delete on right-click / shift-click
                if resp.clicked() {
                    if ui.input(|i| i.modifiers.shift) {
                        *to_remove = Some(entry.path.clone());
                    } else {
                        *to_select = Some(entry.path.clone());
                    }
                }
                resp.context_menu(|ui| {
                    if ui.button("Remove from library").clicked() {
                        *to_remove = Some(entry.path.clone());
                        ui.close_menu();
                    }
                });
            }
            // Trailing import-tile (dashed) only on the last partially-filled row
            // — keeps the grid always feeling complete.
            if row.len() < 3 {
                let (rect, resp) =
                    ui.allocate_exact_size(egui::vec2(cell, cell), egui::Sense::click());
                let p = ui.painter();
                let stroke_color = if resp.hovered() {
                    color::STROKE_STRONG
                } else {
                    color::STROKE
                };
                // Dashed border drawn as short segments along each edge.
                p.rect_filled(
                    rect,
                    Rounding::same(control::THUMB_RADIUS),
                    color::PANEL_INSET,
                );
                draw_dashed_rect(p, rect, Stroke::new(1.0, stroke_color));
                p.text(
                    rect.center() - egui::vec2(0.0, 6.0),
                    egui::Align2::CENTER_CENTER,
                    "+",
                    egui::FontId::proportional(20.0),
                    color::TEXT_MUTED,
                );
                p.text(
                    rect.center() + egui::vec2(0.0, 12.0),
                    egui::Align2::CENTER_CENTER,
                    "Import",
                    egui::FontId::proportional(10.5),
                    color::TEXT_MUTED,
                );
                if resp.clicked() {
                    self.do_import();
                }
            }
        });
    }

    /// Small settings panel under the scene controls. Right now it only
    /// holds the Start-on-login toggle, but the section caption keeps it
    /// extensible for future per-user preferences (fps, …).
    fn sidebar_settings(&mut self, ui: &mut egui::Ui) {
        theme::section_caption(ui, "Settings");
        ui.add_space(space::SM);

        // Start-on-login toggle.
        if toggle_row(
            ui,
            self.cfg.start_on_login,
            "Start on login",
            "Run headless at login so apps see the cam",
        ) {
            self.cfg.start_on_login = !self.cfg.start_on_login;
            self.save_settings();
            // Match the on-disk autostart entry to the new preference.
            // Failure here is non-fatal — surface it to the user via the
            // header so they know the setting didn't take.
            let exe = std::env::current_exe().ok();
            let result = match (self.cfg.start_on_login, exe.as_deref()) {
                (true, Some(p)) => autostart::install(p),
                (true, None) => Err(anyhow!("could not resolve current binary path")),
                (false, _) => autostart::uninstall(),
            };
            if let Err(e) = result {
                log::warn!("autostart toggle: {e:#}");
                self.error = Some(format!("Autostart toggle failed: {e:#}"));
            }
        }

        ui.add_space(space::SM);

        // Show-preview toggle. Off → the preview pane shows a static
        // placeholder and the pipeline stops forwarding frames to the
        // GUI (saves a per-frame RGBA clone). The broadcast itself is
        // unaffected: consumers of /dev/video10 see the same picture.
        if toggle_row(
            ui,
            self.cfg.show_preview,
            "Show preview",
            "Render live frames in this window",
        ) {
            self.cfg.show_preview = !self.cfg.show_preview;
            self.save_settings();
            if let Some(tx) = &self.cmd_tx {
                let _ = tx.send(Command::SetPreviewEnabled(self.cfg.show_preview));
            }
            // Drop the cached texture so the placeholder renders cleanly
            // (otherwise the last-painted frame would linger until the
            // next pane resize).
            if !self.cfg.show_preview {
                self.preview_tex = None;
                self.last_preview_size = None;
            }
        }

        ui.add_space(space::SM);

        // Auto-frame toggle. On → the pipeline locks onto the silhouette
        // on the first detection and keeps the framing fixed there for
        // the rest of the session. Toggle off+on to re-acquire on a new
        // position. Works in all three bg modes: Blur and None do a
        // post-composite crop on the whole frame (wall zooms slightly
        // with you), Image keeps the bg image static and slides the
        // foreground over it.
        if toggle_row(
            ui,
            self.cfg.auto_frame,
            "Auto-frame",
            "Lock and centre on you",
        ) {
            self.cfg.auto_frame = !self.cfg.auto_frame;
            self.save_settings();
            if let Some(tx) = &self.cmd_tx {
                let _ = tx.send(Command::SetFraming(self.cfg.auto_frame));
            }
        }
    }

    fn primary_action(&mut self, ui: &mut egui::Ui) {
        let running = self.running();
        let label = if running {
            "Stop broadcasting"
        } else {
            "Start broadcasting"
        };
        let (fill, stroke_color, text_color) = if running {
            (color::DANGER_SOFT, color::DANGER, color::DANGER)
        } else {
            (color::ACCENT_SOFT, color::ACCENT, color::ACCENT)
        };
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), control::PRIMARY_HEIGHT),
            egui::Sense::click(),
        );
        let p = ui.painter();
        let bg = if resp.hovered() {
            fill.linear_multiply(1.4)
        } else {
            fill
        };
        p.rect(
            rect,
            Rounding::same(radius::MD),
            bg,
            Stroke::new(1.0, stroke_color.linear_multiply(0.5)),
        );
        // tally / start dot
        let dot = rect.left_center() + egui::vec2(16.0, 0.0);
        p.circle_filled(dot, 4.0, stroke_color);
        p.circle_stroke(
            dot,
            7.0,
            Stroke::new(1.0, stroke_color.linear_multiply(0.35)),
        );
        p.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(13.0),
            text_color,
        );
        if resp.clicked() {
            if running {
                self.stop_pipeline();
            } else {
                self.start_pipeline();
            }
        }
    }

    fn do_import(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp"])
            .pick_file()
        {
            match backgrounds::import(&p) {
                Ok(stored) => {
                    self.cfg.background_path = Some(stored);
                    self.cfg.mode = Mode::Replace;
                    self.refresh_library();
                    self.save_settings();
                    if self.running() {
                        self.update_background_live();
                    }
                }
                Err(e) => {
                    log::warn!("import: {e:#}");
                    self.error = Some(format!("Import failed: {e:#}"));
                }
            }
        }
    }
}
