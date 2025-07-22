import cv2
import numpy as np


def apply_background(frame: np.ndarray, mask: np.ndarray, background, blur: bool = False):
    """
    Apply the background to a frame using a mask.

    Args:
        frame (np.ndarray): The input frame.
        mask (np.ndarray): The foreground mask.
        background (np.ndarray): The background image. Can be None if blurring.
        blur (bool): Whether to apply a blur effect to the background.

    Returns:
        np.ndarray: The frame with the new background.
    """
    mask_3d = mask[:, :, np.newaxis]

    if blur:
        # Apply a Gaussian blur to the original frame to create a blurred background
        blurred_background = cv2.GaussianBlur(frame, (41, 41), 0)
        # Combine the foreground with the blurred background
        output = frame * mask_3d + blurred_background * (1 - mask_3d)
    else:
        if background is None:
            # If no background is provided and not blurring, return original frame
            return frame
        # Resize the background to match the frame size
        background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
        # Combine the foreground with the custom background
        output = frame * mask_3d + background * (1 - mask_3d)

    return output.astype(np.uint8)
