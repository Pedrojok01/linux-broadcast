import os
import urllib.request
from pathlib import Path


def reporthook(blocknum, blocksize, totalsize):
    """
    A simple reporthook for urllib.request.urlretrieve.
    """
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = f"\\r{percent:5.1f}% {readsofar:d} / {totalsize:d}"
        print(s, end="")
        if readsofar >= totalsize:
            print()
    else:  # total size is unknown
        print(f"read {readsofar:d}")


def download_file(url, file_path):
    """
    Downloads a file from a URL to a given path with progress.
    """
    if file_path.exists():
        print(f"File '{file_path}' already exists. Skipping download.")
        return

    print(f"File not found. Downloading from {url}...")

    # Create the parent directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, file_path, reporthook)
        print(f"File downloaded and saved to '{file_path}'")
    except Exception as e:
        print(f"Error downloading the file: {e}")
        # Clean up incomplete file if download failed
        if file_path.exists():
            os.remove(file_path)


def download_models():
    """
    Downloads the pre-trained PyTorch and ONNX models if they don't already exist.
    """
    # PyTorch Model
    pytorch_url = "https://github.com/clibdev/MODNet/releases/latest/download/modnet-webcam.pt"
    pytorch_path = Path("models/modnet_webcam.pth")
    download_file(pytorch_url, pytorch_path)

    # ONNX Model
    onnx_url = "https://github.com/clibdev/MODNet/releases/latest/download/modnet-webcam.onnx"
    onnx_path = Path("models/modnet_webcam.onnx")
    download_file(onnx_url, onnx_path)


if __name__ == "__main__":
    download_models()
