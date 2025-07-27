import cv2
import logging
import sys

logging.basicConfig(level=logging.INFO)


class Camera:
    """
    A class to handle camera operations, including capturing frames.
    """

    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open camera with id: {self.camera_id}")
        except cv2.error as e:
            logging.error(f"OpenCV error opening camera {self.camera_id}: {e}")
            raise
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while opening camera {self.camera_id}: {e}"
            )
            raise

    def get_frame(self):
        """
        Capture a single frame from the camera.

        Returns:
            A numpy.ndarray representing the captured frame, or None if an error occurs.
        """
        try:
            if not self.cap.isOpened():
                logging.warning(f"Camera {self.camera_id} is not open.")
                return None
            ret, frame = self.cap.read()
            if not ret:
                logging.warning(f"Could not read frame from camera {self.camera_id}.")
                return None
            return frame
        except cv2.error as e:
            logging.error(f"OpenCV error getting frame from camera {self.camera_id}: {e}")
            return None
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while getting frame from camera {self.camera_id}: {e}"
            )
            return None

    def release(self):
        """
        Release the camera resource.
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logging.info(f"Camera {self.camera_id} released.")

    @staticmethod
    def list_cameras():
        """
        List available cameras using a more robust, platform-specific method.

        Returns:
            A list of available camera indices.
        """
        if sys.platform == "win32":
            backend = cv2.CAP_MSMF
        else:
            backend = cv2.CAP_V4L2

        arr = []
        for index in range(10):  # Check up to 10 devices
            try:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    arr.append(index)
                    cap.release()
            except Exception as e:
                logging.debug(f"Could not probe camera at index {index}: {e}")
        return arr
