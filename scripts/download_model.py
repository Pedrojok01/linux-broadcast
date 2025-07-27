import os
import urllib.request
from pathlib import Path
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import shutil

load_dotenv()


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


def download_hf_file(repo_id, filename, file_path):
    """Download a gated file from Hugging Face Hub, using an optional HF_TOKEN env var."""
    if file_path.exists():
        print(f"File '{file_path}' already exists. Skipping download.")
        return

    print(f"File not found. Downloading {filename} from Hugging Face repo {repo_id}...")

    try:
        download_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=os.getenv("HF_TOKEN", ""),  # Default to empty string if not set
            local_dir=str(file_path.parent),
        )
        # Copy/rename to the exact destination expected by the app (e.g. models/rmbg_2_0_fp16.onnx)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(download_path, file_path)
        print(f"File downloaded and saved to '{file_path}'")
    except Exception as e:
        print(f"Error downloading the file from Hugging Face Hub: {e}")
        if file_path.exists():
            os.remove(file_path)


def download_models():
    """
    Downloads the pre-trained models if they don't already exist.
    """

    # RVM (Robust Video Matting) ONNX Model
    rvm_onnx_url = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp16.onnx"
    rvm_onnx_path = Path("models/rvm_mobilenetv3_fp16.onnx")
    download_file(rvm_onnx_url, rvm_onnx_path)

    # RMBG (Remove Background) 2.0 ONNX Model (gated on Hugging Face)
    rmbg_repo = "briaai/RMBG-2.0"
    rmbg_filename = "onnx/model_fp16.onnx"
    rmbg_onnx_path = Path("models/rmbg_2_0_fp16.onnx")
    download_hf_file(rmbg_repo, rmbg_filename, rmbg_onnx_path)


if __name__ == "__main__":
    download_models()
