import os
import cv2
import numpy as np
from app.gui import launch_gui
from pathlib import Path

from scripts.download_model import download_models

os.path.join(
    os.getenv("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"), "bin"
)


def main():
    # --- Setup ---
    # Download the model if it doesn't exist
    download_models()

    # Create a default background if it does not exist
    DEFAULT_BACKGROUND_PATH = "assets/default_backgrounds/office.jpg"
    if not Path(DEFAULT_BACKGROUND_PATH).exists():
        print(
            f"Default background not found at {DEFAULT_BACKGROUND_PATH}. Creating a dummy background."
        )
        dummy_bg = np.zeros((720, 1280, 3), dtype=np.uint8)
        Path("assets/default_backgrounds").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(DEFAULT_BACKGROUND_PATH, dummy_bg)

    # Launch the GUI
    launch_gui()


if __name__ == "__main__":
    main()
