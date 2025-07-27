import sys
import numpy as np
import logging


class VCam:
    """
    A class to handle the virtual webcam.
    On Windows, it runs in a 'preview' mode where it does nothing.
    """

    def __init__(self, width, height, device, channels=3):
        self.width = width
        self.height = height
        self.device = device
        self.channels = channels
        self._camera = None

        if sys.platform == "win32":
            logging.info("Running on Windows. Virtual camera disabled (Preview Mode).")
            return

        try:
            import pyfakewebcam

            self._camera = pyfakewebcam.FakeWebcam(self.device, self.width, self.height)
        except ImportError:
            logging.error(
                "The 'pyfakewebcam' package is not installed. "
                "Please install it to use the virtual camera feature on Linux."
            )
        except Exception as e:
            logging.error(
                f"Could not create virtual webcam at {self.device}. "
                f"Please ensure v4l2loopback is installed and the device exists. Error: {e}"
            )

    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to the virtual camera.
        Does nothing if the virtual camera is not available.
        """
        if self._camera is None:
            return

        # pyfakewebcam expects RGB format
        frame_rgb = frame[:, :, ::-1]  # Convert BGR to RGB
        self._camera.schedule_frame(frame_rgb)

    def close(self):
        """
        Close the virtual camera.
        """
        if self._camera:
            # The underlying file is closed when the object is garbage collected.
            # This is a placeholder for any future cleanup.
            pass
