import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import logging
import onnxruntime
import numpy as np
import cv2


class IBNorm(nn.Module):
    """Combine Instance Norm and Batch Norm into one module"""

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        self.in_channels = in_channels
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.inn = nn.InstanceNorm2d(self.in_channels)

    def forward(self, x):
        x_bn = self.bn(x)
        x_in = self.inn(x)
        return (x_bn + x_in) / 2


class Conv2dIBNormRelu(nn.Module):
    """
    Convolution + IBNorm + Relu
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        with_ibn=True,
        with_relu=True,
    ):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if with_ibn:
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# -----------------------------------------------------------------------------------------------------------
# -------------------------------------------------  MOD-Net  -----------------------------------------------
# -----------------------------------------------------------------------------------------------------------


class MODNet(nn.Module):
    """
    A Trimap-Free Portrait Matting Solution in Real Time
    This is the official implementation of MODNet.
    Original repository: https://github.com/ZHKKKe/MODNet
    """

    def __init__(
        self, in_channels=3, hr_channels=32, backbone_arch="mobilenetv2", backbone_pretrained=True
    ):
        super(MODNet, self).__init__()

        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_arch = backbone_arch
        self.backbone_pretrained = backbone_pretrained

        self.backbone = MobileNetV2(
            in_channels=self.in_channels, pretrained=self.backbone_pretrained
        )

        self.lr_branch = LRBranch(self.backbone.channels)
        self.hr_branch = HRBranch(
            self.hr_channels, self.backbone.channels[0], self.backbone.channels[1]
        )
        self.f_branch = FusionBranch(self.hr_channels, self.lr_branch.out_channels)

    def forward(self, img, inference):
        enc_feats = self.backbone.forward(img)
        lr_out = self.lr_branch.forward(enc_feats)
        hr_out = self.hr_branch.forward(img, enc_feats[0], enc_feats[1], enc_feats[2])
        f_out = self.f_branch.forward(hr_out, lr_out)

        if inference:
            return f_out
        else:
            return f_out, lr_out


class LRBranch(nn.Module):
    def __init__(self, enc_channels):
        super(LRBranch, self).__init__()

        self.enc_channels = enc_channels
        self.out_channels = 1

        self.convs = nn.Sequential(
            Conv2dIBNormRelu(self.enc_channels[4], 512, 3, stride=2, padding=1),
            Conv2dIBNormRelu(512, 512, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(512, self.out_channels)

    def forward(self, enc_feats):
        enc_out = self.convs(enc_feats[4])
        enc_out = torch.flatten(enc_out, 1)
        return self.fc(enc_out)


class HRBranch(nn.Module):
    def __init__(self, hr_channels, enc_channels_0, enc_channels_1):
        super(HRBranch, self).__init__()

        self.hr_channels = hr_channels
        self.enc_channels_0 = enc_channels_0
        self.enc_channels_1 = enc_channels_1

        self.to_hr = nn.Sequential(
            Conv2dIBNormRelu(3, self.hr_channels, 3, stride=2, padding=1),
            Conv2dIBNormRelu(self.hr_channels, self.hr_channels, 3, stride=1, padding=1),
        )
        self.conv_v = Conv2dIBNormRelu(
            self.hr_channels + self.enc_channels_1, self.hr_channels, 3, stride=1, padding=1
        )
        self.conv_h = Conv2dIBNormRelu(self.hr_channels, self.hr_channels, 3, stride=1, padding=1)
        self.se_block = SEBlock(self.hr_channels, self.hr_channels)

    def forward(self, img, enc_feat_0, enc_feat_1, enc_feat_2):
        hr = self.to_hr(img)
        hr = F.interpolate(hr, scale_factor=2, mode="bilinear", align_corners=False)
        hr = self.conv_v(torch.cat([hr, enc_feat_2], dim=1))
        hr = F.interpolate(hr, scale_factor=2, mode="bilinear", align_corners=False)
        hr = self.conv_h(torch.cat([hr, enc_feat_1], dim=1))
        hr = F.interpolate(hr, scale_factor=2, mode="bilinear", align_corners=False)
        hr = self.se_block(hr)
        return hr


class FusionBranch(nn.Module):
    def __init__(self, hr_channels, lr_out_channels):
        super(FusionBranch, self).__init__()

        self.conv1 = Conv2dIBNormRelu(hr_channels, 32, 3, padding=1)
        self.conv2 = Conv2dIBNormRelu(32, 1, 1, with_ibn=False, with_relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hr_out, lr_out):
        out = self.conv1(hr_out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return out


class MobileNetV2(nn.Module):
    """
    MODNet backbone, based on MobileNetV2
    """

    def __init__(self, in_channels=3, pretrained=True):
        super(MobileNetV2, self).__init__()
        from torchvision.models.mobilenet import mobilenet_v2

        self.model = mobilenet_v2(pretrained=pretrained)
        self.model.features[0][0] = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.channels = [16, 24, 32, 96, 1280]

    def forward(self, x):
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        f1 = x
        x = self.model.features[2](x)
        x = self.model.features[3](x)
        f2 = x
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        x = self.model.features[6](x)
        f3 = x
        x = self.model.features[7](x)
        x = self.model.features[8](x)
        x = self.model.features[9](x)
        x = self.model.features[10](x)
        x = self.model.features[11](x)
        x = self.model.features[12](x)
        x = self.model.features[13](x)
        f4 = x
        x = self.model.features[14](x)
        x = self.model.features[15](x)
        x = self.model.features[16](x)
        x = self.model.features[17](x)
        x = self.model.features[18](x)
        f5 = x

        return [f1, f2, f3, f4, f5]


class PyTorchSegmenter:
    """
    A wrapper class for the PyTorch MODNet model for easy loading and inference.
    """

    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = MODNet(backbone_arch="mobilenetv2")
            self.model = nn.DataParallel(self.model)
            # Load the model with strict=False to ignore mismatched layers
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device), strict=False
            )
            self.model.eval()
            self.model.to(self.device)
            self.transform = self._get_transform()
        except FileNotFoundError:
            logging.error(f"Model file not found at {model_path}. Please run the download script.")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading the PyTorch model: {e}")
            raise

    def _get_transform(self):
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def segment(self, image):
        """
        Segment the foreground from the background in an image.

        Args:
            image (numpy.ndarray): The input image in BGR format.

        Returns:
            numpy.ndarray: The foreground mask.
        """
        h, w, _ = image.shape
        # Convert BGR to RGB before transforming
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            matte = self.model(image_tensor, inference=True)

        matte = F.interpolate(matte, size=(h, w), mode="bilinear", align_corners=False)
        return matte.squeeze().cpu().numpy()


class ONNXSegmenter:
    """
    A wrapper class for the ONNX MODNet model for easy loading and inference.
    """

    def __init__(self, model_path: str):
        try:
            # Check for CUDA availability for ONNX Runtime
            if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                logging.info("ONNX: Using CUDAExecutionProvider.")
            else:
                providers = ["CPUExecutionProvider"]
                logging.info("ONNX: CUDA not available. Using CPUExecutionProvider.")

            self.session = onnxruntime.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.transform = self._get_transform()
        except FileNotFoundError:
            logging.error(f"Model file not found at {model_path}. Please run the download script.")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading the ONNX model: {e}")
            raise

    def _get_transform(self):
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def segment(self, image):
        """
        Segment the foreground from the background in an image.

        Args:
            image (numpy.ndarray): The input image in BGR format.

        Returns:
            numpy.ndarray: The foreground mask.
        """
        h, w, _ = image.shape
        # Convert BGR to RGB before transforming
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0).numpy()

        matte = self.session.run([self.output_name], {self.input_name: image_tensor})[0]

        matte_tensor = torch.from_numpy(matte)
        matte = F.interpolate(matte_tensor, size=(h, w), mode="bilinear", align_corners=False)
        return matte.squeeze().numpy()


class RVMSegmenter:
    """
    A wrapper class for the ONNX Robust Video Matting (RVM) model.
    It maintains recurrent states for temporal consistency.
    """

    def __init__(self, model_path: str):
        try:
            # Check for CUDA availability for ONNX Runtime
            if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                logging.info("RVM: Using CUDAExecutionProvider.")
            else:
                providers = ["CPUExecutionProvider"]
                logging.info("RVM: CUDA not available. Using CPUExecutionProvider.")

            self.session = onnxruntime.InferenceSession(model_path, providers=providers)
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]

            # RVM model has specific input names for recurrent states
            self.rec_in_names = sorted(
                [name for name in self.input_names if name.startswith("r") and name.endswith("i")]
            )
            self.rec_out_names = sorted(
                [name for name in self.output_names if name.startswith("r") and name.endswith("o")]
            )

            # Initialize recurrent states
            # RVM expects specific shapes for recurrent states, let's get them from the model
            self.rec = []
            for name in self.rec_in_names:
                for inp in self.session.get_inputs():
                    if inp.name == name:
                        self.rec.append(np.zeros(inp.shape, dtype=np.float32))
                        break

            self.downsample_ratio = np.array([0.25], dtype=np.float32)  # Hyperparameter

        except Exception as e:
            logging.error(f"An unexpected error occurred while loading the RVM model: {e}")
            raise

    def segment(self, image):
        """
        Segment the foreground from the background in an image using RVM.
        """
        h, w, _ = image.shape
        image_tensor = self._preprocess(image)

        inputs = {self.input_names[0]: image_tensor}
        for name, r_val in zip(self.rec_in_names, self.rec):
            inputs[name] = r_val
        if "downsample_ratio" in self.input_names:
            inputs["downsample_ratio"] = self.downsample_ratio

        outs = self.session.run(self.output_names, inputs)

        # The matte is usually the first output
        matte = outs[0]
        # Recurrent states are all outputs after the first two (fgr, pha)
        self.rec = outs[2:]  # Update recurrent states

        return self._postprocess(matte, h, w)

    def _preprocess(self, image):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize and add batch dimension
        image_rgb = image_rgb / 255.0
        image_norm = (image_rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return np.expand_dims(image_norm.transpose(2, 0, 1), 0).astype("float32")

    def _postprocess(self, matte, h, w):
        # Resize matte to original image size and remove batch dimension
        matte_tensor = torch.from_numpy(matte)
        matte = F.interpolate(matte_tensor, size=(h, w), mode="bilinear", align_corners=False)
        return matte.squeeze().numpy()


class RMBGSegmenter(ONNXSegmenter):
    """A wrapper for the RMBG (Remove Background) 2.0 ONNX model.

    The published ONNX weights expect a fixed 1024×1024 RGB input. We therefore
    resize each incoming frame to that resolution for inference and upscale the
    resulting matte back to the original frame size so the rest of the pipeline
    remains unchanged.
    """

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.input_size = (1024, 1024)  # (width, height)

    def segment(self, image):
        h, w, _ = image.shape
        # Convert BGR → RGB and resize for the model
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, self.input_size, interpolation=cv2.INTER_AREA)

        image_tensor = self.transform(image_resized).unsqueeze(0).numpy()
        matte = self.session.run([self.output_name], {self.input_name: image_tensor})[0]

        matte_tensor = torch.from_numpy(matte)
        matte = F.interpolate(matte_tensor, size=(h, w), mode="bilinear", align_corners=False)
        return matte.squeeze().numpy()


def create_segmenter(model_path: str):
    """
    Factory function to create the appropriate segmenter based on the model file extension or name.
    """
    if "rmbg" in model_path:
        return RMBGSegmenter(model_path)
    elif "rvm" in model_path:
        return RVMSegmenter(model_path)
    elif model_path.endswith(".pth"):
        return PyTorchSegmenter(model_path)
    elif model_path.endswith(".onnx"):
        return ONNXSegmenter(model_path)
    else:
        raise ValueError(f"Unsupported model file: {model_path}")
