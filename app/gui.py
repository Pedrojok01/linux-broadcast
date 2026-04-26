from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
    QComboBox,
    QFileDialog,
    QCheckBox,
    QRadioButton,
    QGroupBox,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QThread, Signal
import cv2
import numpy as np
import logging
import sys

from app.camera import Camera
from app.segment import create_segmenter
from app.vcam import VCam
from app.background import apply_background
from app.settings import load_settings, save_settings


class VideoThread(QThread):
    change_pixmap_signal = Signal(np.ndarray)

    def __init__(self, camera_id, model_path, background_path, virtual_device, blur=False):
        super().__init__()
        self._run_flag = True
        self.camera_id = camera_id
        self.model_path = model_path
        self.background_path = background_path
        self.virtual_device = virtual_device
        self.blur = blur

    def run(self):
        try:
            camera = Camera(camera_id=self.camera_id)
            segmenter = create_segmenter(model_path=self.model_path)
            background_image = cv2.imread(self.background_path) if not self.blur else None
            frame = camera.get_frame()
            if frame is None:
                logging.error("Could not get frame from camera.")
                return
            height, width, _ = frame.shape
            vcam = None
            if self.virtual_device:
                vcam = VCam(width=width, height=height, device=self.virtual_device)
        except Exception as e:
            logging.error(f"Error initializing video thread: {e}")
            return

        while self._run_flag:
            frame = camera.get_frame()
            if frame is not None:
                mask = segmenter.segment(frame)
                output_frame = apply_background(frame, mask, background_image, blur=self.blur)
                if vcam:
                    vcam.write_frame(output_frame)
                self.change_pixmap_signal.emit(output_frame)

        camera.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        if sys.platform == "win32":
            self.setWindowTitle("Virtual Background (Windows Preview)")
        else:
            self.setWindowTitle("Linux Virtual Background")

        self.settings = load_settings()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Model Selection
        model_groupbox = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        self.pytorch_radio = QRadioButton("MODNet (PyTorch)")
        self.onnx_radio = QRadioButton("MODNet (ONNX)")
        self.rvm_radio = QRadioButton("RVM (ONNX)")
        self.rmbg_radio = QRadioButton("RMBG-2.0 (ONNX)")
        model_layout.addWidget(self.pytorch_radio)
        model_layout.addWidget(self.onnx_radio)
        model_layout.addWidget(self.rvm_radio)
        model_layout.addWidget(self.rmbg_radio)
        model_groupbox.setLayout(model_layout)
        self.layout.addWidget(model_groupbox)

        model_setting = self.settings.get("model", "pytorch")
        if model_setting == "onnx":
            self.onnx_radio.setChecked(True)
        elif model_setting == "rvm":
            self.rvm_radio.setChecked(True)
        elif model_setting == "rmbg":
            self.rmbg_radio.setChecked(True)
        else:
            self.pytorch_radio.setChecked(True)

        self.pytorch_radio.toggled.connect(self.save_current_settings)
        self.onnx_radio.toggled.connect(self.save_current_settings)
        self.rvm_radio.toggled.connect(self.save_current_settings)
        self.rmbg_radio.toggled.connect(self.save_current_settings)

        # Camera selection
        self.camera_selector = QComboBox()
        try:
            self.available_cameras = Camera.list_cameras()
            if self.available_cameras:
                self.camera_selector.addItems([f"Camera {c}" for c in self.available_cameras])
                self.camera_selector.setCurrentIndex(self.settings.get("camera_id", 0))
            else:
                self.camera_selector.addItem("No cameras found")
                self.camera_selector.setEnabled(False)
        except Exception as e:
            logging.error(f"Error initializing camera selector: {e}")
            self.available_cameras = []

        self.camera_selector.currentIndexChanged.connect(self.save_current_settings)
        self.layout.addWidget(self.camera_selector)

        # Background selection
        self.background_button = QPushButton("Select Background Image")
        self.background_button.clicked.connect(self.select_background)
        self.layout.addWidget(self.background_button)
        self.background_path = self.settings.get(
            "background_path", "assets/default_backgrounds/office.jpg"
        )

        # Blur checkbox
        self.blur_checkbox = QCheckBox("Blur Background")
        self.blur_checkbox.setChecked(self.settings.get("blur", False))
        self.blur_checkbox.stateChanged.connect(self.toggle_blur)
        self.layout.addWidget(self.blur_checkbox)
        self.toggle_blur(self.blur_checkbox.checkState().value)  # Set initial state

        # Video display
        self.image_label = QLabel(self)
        self.image_label.resize(640, 480)
        self.layout.addWidget(self.image_label)

        # Start/Stop button
        self.button = QPushButton("Start", self)
        self.button.clicked.connect(self.toggle_video)
        self.layout.addWidget(self.button)

        self.video_thread = None

    def save_current_settings(self):
        if self.pytorch_radio.isChecked():
            model = "pytorch"
        elif self.onnx_radio.isChecked():
            model = "onnx"
        elif self.rvm_radio.isChecked():
            model = "rvm"
        else:
            model = "rmbg"
        self.settings["model"] = model
        if self.available_cameras:
            self.settings["camera_id"] = self.camera_selector.currentIndex()
        self.settings["background_path"] = self.background_path
        self.settings["blur"] = self.blur_checkbox.isChecked()
        save_settings(self.settings)

    def toggle_blur(self, state):
        is_blurred = state == Qt.CheckState.Checked.value
        self.background_button.setEnabled(not is_blurred)
        self.save_current_settings()

    def select_background(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Background Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_name:
            self.background_path = file_name
            self.save_current_settings()

    def toggle_video(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
            self.button.setText("Start")
        else:
            if not self.available_cameras:
                print("No cameras available to start.")
                return

            if self.pytorch_radio.isChecked():
                model_name = "modnet_webcam.pth"
            elif self.onnx_radio.isChecked():
                model_name = "modnet_webcam.onnx"
            elif self.rvm_radio.isChecked():
                model_name = "rvm_mobilenetv3_fp16.onnx"
            else:  # RMBG selected
                model_name = "rmbg_2_0_fp16.onnx"
            model_path = f"models/{model_name}"

            selected_camera = self.available_cameras[self.camera_selector.currentIndex()]

            # Platform-specific device: None on Windows, device path on Linux
            virtual_device = "/dev/video10" if sys.platform != "win32" else None

            self.video_thread = VideoThread(
                camera_id=selected_camera,
                model_path=model_path,
                background_path=self.background_path,
                virtual_device=virtual_device,
                blur=self.blur_checkbox.isChecked(),
            )
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.start()
            self.button.setText("Stop")

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()


def launch_gui():
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec()


if __name__ == "__main__":
    launch_gui()
