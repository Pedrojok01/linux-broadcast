import unittest
import numpy as np

from app.segment import create_segmenter, PyTorchSegmenter, ONNXSegmenter
from scripts.download_model import download_models


class TestSegmenter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Download the models before running the tests.
        """
        download_models()
        cls.pytorch_model_path = "models/modnet_webcam.pth"
        cls.onnx_model_path = "models/modnet_webcam.onnx"
        cls.dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_create_segmenter(self):
        """
        Test the create_segmenter factory function.
        """
        pytorch_segmenter = create_segmenter(self.pytorch_model_path)
        self.assertIsInstance(pytorch_segmenter, PyTorchSegmenter)

        onnx_segmenter = create_segmenter(self.onnx_model_path)
        self.assertIsInstance(onnx_segmenter, ONNXSegmenter)

        with self.assertRaises(ValueError):
            create_segmenter("invalid_model.ext")

    def test_pytorch_segmenter(self):
        """
        Test the PyTorchSegmenter class.
        """
        segmenter = PyTorchSegmenter(self.pytorch_model_path)
        mask = segmenter.segment(self.dummy_image)
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.shape, (480, 640))

    def test_onnx_segmenter(self):
        """
        Test the ONNXSegmenter class.
        """
        segmenter = ONNXSegmenter(self.onnx_model_path)
        mask = segmenter.segment(self.dummy_image)
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.shape, (480, 640))


if __name__ == "__main__":
    unittest.main()
