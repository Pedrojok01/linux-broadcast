use anyhow::{anyhow, Result};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use eframe::egui::{self, Color32, Margin, Rounding, Stroke};
use lb_pipeline::{Background, Command, Pipeline, PipelineConfig, PreviewFrame};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::backgrounds::{self, LibraryEntry};
use crate::cameras::{enumerate, CameraEntry};
use crate::config::{Config, Mode, Model};
use crate::theme::{self, color, control, radius, space};
use crate::{MODEL_BINARY_ONNX, MODEL_MULTICLASS_ONNX, MODEL_RVM_ONNX};

/// Headless mode (no GUI) — same blocking behaviour as the v1 CLI binary.
pub fn run_headless() -> Result<()> {
    let cfg = Config::load();
    let pcfg = pipeline_config_from(&cfg, None);
    log::info!(
        "starting headless pipeline {} → {} ({}x{}@{}fps)",
        pcfg.source_device,
        pcfg.sink_device,
        pcfg.width,
        pcfg.height,
        pcfg.framerate,
    );
    let pipeline = Pipeline::start(pcfg, MODEL_BINARY_ONNX, MODEL_MULTICLASS_ONNX, MODEL_RVM_ONNX)?;
    pipeline.run_until_done()?;
    Ok(())
}

pub fn run_gui() -> Result<()> {
    let viewport = egui::ViewportBuilder::default()
        .with_inner_size([1000.0, 720.0])
        .with_min_inner_size([820.0, 560.0])
        .with_title("LinuxBroadcast")
        // app_id is required on Wayland for the compositor to match the
        // window to its desktop entry / icon. Must mirror the .desktop
        // file we ship at packaging time.
        .with_app_id("io.Pedrojok01.LinuxBroadcast")
        .with_icon(crate::icon::build());
    let opts = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native(
        "linux-broadcast",
        opts,
        Box::new(|cc| {
            theme::apply(&cc.egui_ctx);
            Ok(Box::new(App::new(cc)))
        }),
    )
    .map_err(|e| anyhow!("eframe: {e}"))
}

fn pipeline_config_from(
    cfg: &Config,
    preview_tx: Option<Sender<PreviewFrame>>,
) -> PipelineConfig {
    PipelineConfig {
        source_device: cfg.source_device.clone(),
        sink_device: cfg.sink_device.clone(),
        width: cfg.width,
        height: cfg.height,
        framerate: cfg.framerate,
        background: build_background(cfg),
        model: cfg.model.into_kind(),
        preview_tx,
    }
}

fn build_background(cfg: &Config) -> Background {
    match cfg.mode {
        Mode::None => Background::None,
        Mode::Blur => Background::Blur {
            strength: cfg.blur_strength,
        },
        Mode::Replace => match cfg.background_image_path() {
            Some(path) => match load_background_rgba(path) {
                Ok(bg) => bg,
                Err(e) => {
                    log::warn!("background image load failed ({e:#}); falling back to blur");
                    Background::Blur {
                        strength: cfg.blur_strength,
                    }
                }
            },
            None => Background::Blur {
                strength: cfg.blur_strength,
            },
        },
    }
}

fn load_background_rgba(path: &Path) -> Result<Background> {
    let img = image::open(path)?.to_rgba8();
    let (w, h) = (img.width(), img.height());
    Ok(Background::Image {
        rgba: img.into_raw(),
        width: w,
        height: h,
    })
}

struct App {
    cfg: Config,
    cameras: Vec<CameraEntry>,
    library: Vec<LibraryEntry>,
    thumbnails: HashMap<PathBuf, egui::TextureHandle>,
    pipeline: Option<Pipeline>,
    cmd_tx: Option<Sender<Command>>,
    preview_rx: Option<Receiver<PreviewFrame>>,
    preview_tex: Option<egui::TextureHandle>,
    last_preview_size: Option<[usize; 2]>,
    error: Option<String>,
}

impl App {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let cfg = Config::load();
        let cameras = enumerate(&cfg.sink_device);
        let library = backgrounds::list();
        Self {
            cfg,
            cameras,
            library,
            thumbnails: HashMap::new(),
            pipeline: None,
            cmd_tx: None,
            preview_rx: None,
            preview_tex: None,
            last_preview_size: None,
            error: None,
        }
    }

    fn save_settings(&self) {
        if let Err(e) = self.cfg.save() {
            log::warn!("config save failed: {e:#}");
        }
    }

    fn refresh_library(&mut self) {
        self.library = backgrounds::list();
        self.thumbnails
            .retain(|p, _| self.library.iter().any(|e| &e.path == p));
    }

    fn running(&self) -> bool {
        self.pipeline.is_some()
    }

    fn start_pipeline(&mut self) {
        if self.running() {
            return;
        }
        let (tx, rx) = crossbeam_channel::bounded::<PreviewFrame>(2);
        let pcfg = pipeline_config_from(&self.cfg, Some(tx));
        match Pipeline::start(pcfg, MODEL_BINARY_ONNX, MODEL_MULTICLASS_ONNX, MODEL_RVM_ONNX) {
            Ok(p) => {
                self.cmd_tx = Some(p.cmd_sender());
                self.pipeline = Some(p);
                self.preview_rx = Some(rx);
                self.preview_tex = None;
                self.last_preview_size = None;
                self.error = None;
            }
            Err(e) => {
                let msg = format!("Failed to start: {e:#}");
                log::error!("{msg}");
                self.error = Some(msg);
            }
        }
    }

    fn stop_pipeline(&mut self) {
        if let Some(p) = self.pipeline.take() {
            p.stop();
            drop(p);
        }
        self.cmd_tx = None;
        self.preview_rx = None;
        self.preview_tex = None;
        self.last_preview_size = None;
    }

    fn restart_pipeline(&mut self) {
        if self.running() {
            self.stop_pipeline();
            self.start_pipeline();
        }
    }

    fn update_background_live(&mut self) {
        if let Some(tx) = &self.cmd_tx {
            let _ = tx.send(Command::SetBackground(build_background(&self.cfg)));
        }
    }

    fn drain_preview(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.preview_rx else { return };
        let mut latest: Option<PreviewFrame> = None;
        loop {
            match rx.try_recv() {
                Ok(f) => latest = Some(f),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.preview_rx = None;
                    break;
                }
            }
        }
        if let Some(frame) = latest {
            let size = [frame.width as usize, frame.height as usize];
            let color = egui::ColorImage::from_rgba_unmultiplied(size, &frame.rgba);
            match &mut self.preview_tex {
                Some(tex) if self.last_preview_size == Some(size) => {
                    tex.set(color, egui::TextureOptions::LINEAR);
                }
                slot => {
                    *slot = Some(ctx.load_texture("preview", color, egui::TextureOptions::LINEAR));
                    self.last_preview_size = Some(size);
                }
            }
        }
    }

    fn thumbnail_for(&mut self, ctx: &egui::Context, path: &Path) -> Option<egui::TextureHandle> {
        if let Some(t) = self.thumbnails.get(path) {
            return Some(t.clone());
        }
        let img = image::open(path).ok()?.thumbnail(160, 160).to_rgba8();
        let size = [img.width() as usize, img.height() as usize];
        let color = egui::ColorImage::from_rgba_unmultiplied(size, &img.into_raw());
        let tex = ctx.load_texture(
            format!("thumb-{}", path.display()),
            color,
            egui::TextureOptions::LINEAR,
        );
        self.thumbnails.insert(path.to_path_buf(), tex.clone());
        Some(tex)
    }
}

// ---------------------------------------------------------------------------
//  egui::App impl — top-level layout
// ---------------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_preview(ctx);

        // Whole window fill (otherwise the area outside panels uses defaults).
        ctx.style_mut(|s| s.visuals.panel_fill = color::PANEL);

        self.header(ctx);
        self.footer(ctx);
        self.sidebar(ctx);
        self.preview_pane(ctx);

        if self.running() {
            ctx.request_repaint_after(Duration::from_millis(33));
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.stop_pipeline();
        self.save_settings();
    }
}

// ---------------------------------------------------------------------------
//  Layout sections
// ---------------------------------------------------------------------------

impl App {
    fn header(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("header")
            .exact_height(56.0)
            .frame(
                egui::Frame::none()
                    .fill(color::PANEL)
                    .stroke(Stroke::new(1.0, color::STROKE)),
            )
            .show(ctx, |ui| {
                ui.add_space(0.0);
                ui.with_layout(
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
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
                        p.circle_filled(
                            rect.right_top() + egui::vec2(-7.0, 7.0),
                            2.4,
                            color::ACCENT,
                        );

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

                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                ui.add_space(space::LG);
                                if let Some(err) = &self.error {
                                    let err = err.clone();
                                    ui.label(
                                        egui::RichText::new(err)
                                            .small()
                                            .color(color::DANGER),
                                    );
                                }
                            },
                        );
                    },
                );
            });
    }

    fn footer(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("footer")
            .exact_height(32.0)
            .frame(
                egui::Frame::none()
                    .fill(color::PANEL)
                    .stroke(Stroke::new(1.0, color::STROKE)),
            )
            .show(ctx, |ui| {
                ui.with_layout(
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        ui.add_space(space::LG);
                        if self.running() {
                            ui.label(egui::RichText::new("●").color(color::SUCCESS).small());
                            ui.add_space(space::XS);
                            ui.label(
                                egui::RichText::new("Running")
                                    .small()
                                    .color(color::SUCCESS),
                            );
                        } else {
                            ui.label(egui::RichText::new("●").color(color::TEXT_MUTED).small());
                            ui.add_space(space::XS);
                            ui.label(
                                egui::RichText::new("Idle")
                                    .small()
                                    .color(color::TEXT_MUTED),
                            );
                        }
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

                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                ui.add_space(space::LG);
                                ui.label(
                                    egui::RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                                        .monospace()
                                        .small()
                                        .color(color::TEXT_MUTED),
                                );
                            },
                        );
                    },
                );
            });
    }

    fn sidebar(&mut self, ctx: &egui::Context) {
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
                ui.spacing_mut().item_spacing.y = space::SECTION_GAP;
                ui.style_mut().spacing.item_spacing.y = space::SECTION_GAP;

                self.sidebar_camera(ui);
                self.sidebar_model(ui);
                self.sidebar_scene(ui);

                // Push primary action to the bottom.
                ui.with_layout(
                    egui::Layout::bottom_up(egui::Align::Center),
                    |ui| {
                        self.primary_action(ui);
                    },
                );
            });
    }

    fn preview_pane(&mut self, ctx: &egui::Context) {
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
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
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
                        },
                    );
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
                p.rect_stroke(rect, Rounding::same(radius::LG), Stroke::new(1.0, color::STROKE));

                if let Some(tex) = &self.preview_tex {
                    let [tw, th] = self.last_preview_size.unwrap_or([1280, 720]);
                    let aspect = tw as f32 / th as f32;
                    let mut w = rect.width();
                    let mut h = w / aspect;
                    if h > rect.height() {
                        h = rect.height();
                        w = h * aspect;
                    }
                    let centered = egui::Rect::from_center_size(
                        rect.center(),
                        egui::vec2(w, h),
                    );
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
                        ui.with_layout(
                            egui::Layout::left_to_right(egui::Align::Center),
                            |ui| {
                                floating_pill(
                                    ui,
                                    &format!("● Sending to {}", self.cfg.sink_device),
                                    color::DANGER,
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        let mirror_label = if self.cfg.mirror {
                                            "↔ Mirror ON"
                                        } else {
                                            "↔ Mirror"
                                        };
                                        if floating_pill_button(
                                            ui,
                                            mirror_label,
                                            self.cfg.mirror,
                                        )
                                        .clicked()
                                        {
                                            self.cfg.mirror = !self.cfg.mirror;
                                            self.save_settings();
                                        }
                                    },
                                );
                            },
                        );
                    });
                }
            });
    }
}

// ---------------------------------------------------------------------------
//  Sidebar sections
// ---------------------------------------------------------------------------

impl App {
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
        egui::popup_below_widget(ui, popup_id, &resp, egui::PopupCloseBehavior::CloseOnClick, |ui| {
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
        });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if ghost_button(ui, "⟳ Refresh").clicked() {
                self.cameras = enumerate(&self.cfg.sink_device);
            }
        });

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
                let (rect, resp) = ui.allocate_exact_size(
                    egui::vec2(cell_w, 30.0),
                    egui::Sense::click(),
                );
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
                ui.with_layout(
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "{}%",
                                (self.cfg.blur_strength * 100.0).round() as i32
                            ))
                            .monospace()
                            .small()
                            .color(color::TEXT),
                        );
                    },
                );
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
                // Dashed border emulated via 6 short strokes per side.
                p.rect_filled(
                    rect,
                    Rounding::same(control::THUMB_RADIUS),
                    color::PANEL_INSET,
                );
                draw_dashed_rect(
                    p,
                    rect,
                    control::THUMB_RADIUS,
                    Stroke::new(1.0, stroke_color),
                );
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
        p.circle_stroke(dot, 7.0, Stroke::new(1.0, stroke_color.linear_multiply(0.35)));
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

// ---------------------------------------------------------------------------
//  Small drawing helpers
// ---------------------------------------------------------------------------

fn pill(ui: &mut egui::Ui, text: &str, fg: Color32, bg_override: Option<Color32>) {
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

fn floating_pill(ui: &mut egui::Ui, text: &str, accent: Color32) {
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

fn floating_pill_button(ui: &mut egui::Ui, text: &str, active: bool) -> egui::Response {
    let font = egui::FontId::proportional(11.0);
    let galley = ui.fonts(|f| f.layout_no_wrap(text.to_string(), font.clone(), color::TEXT));
    let pad = egui::vec2(12.0, 7.0);
    let (rect, resp) = ui.allocate_exact_size(galley.size() + pad * 2.0, egui::Sense::click());
    let p = ui.painter();
    let stroke_color = if active {
        color::ACCENT
    } else if resp.hovered() {
        color::STROKE_STRONG
    } else {
        color::STROKE_STRONG.linear_multiply(0.7)
    };
    p.rect(
        rect,
        Rounding::same(999.0),
        Color32::from_rgba_premultiplied(0x08, 0x0A, 0x0E, 0xC0),
        Stroke::new(1.0, stroke_color),
    );
    p.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        text,
        font,
        if active { color::ACCENT } else { color::TEXT_WEAK },
    );
    resp
}

fn ghost_button(ui: &mut egui::Ui, text: &str) -> egui::Response {
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

fn sep(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(1.0, 14.0), egui::Sense::hover());
    ui.painter()
        .vline(rect.center().x, rect.y_range(), Stroke::new(1.0, color::STROKE_STRONG));
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max - 1).collect();
        out.push('…');
        out
    }
}

fn draw_dashed_rect(p: &egui::Painter, rect: egui::Rect, _radius: f32, stroke: Stroke) {
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
