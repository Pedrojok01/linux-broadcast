import cv2
import logging

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
        List available cameras.

        Returns:
            A list of available camera indices.
        """
        index = 0
        arr = []
        while True:
            try:
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    break
                else:
                    arr.append(index)
                cap.release()
                index += 1
            except Exception as e:
                logging.error(f"Error while checking for camera at index {index}: {e}")
                break
        return arr
