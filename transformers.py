import sys

import numpy as np
from PIL import Image

from function import _compute_resized_output_size, get_dimensions, InterpolationMode, _interpolation_modes_from_int, \
    pil_modes_mapping


class Resize:
    """
    clone from torchvision
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True):
        self.size = size
        self.max_size = max_size

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def __call__(self, img):
        interpolation = self.interpolation
        size = self.size
        max_size = self.max_size
        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)
        elif not isinstance(interpolation, InterpolationMode):
            raise TypeError(
                "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
            )

        _, image_height, image_width = get_dimensions(img)
        if isinstance(size, int):
            size = [size]
        output_size = _compute_resized_output_size((image_height, image_width), size, max_size)

        if [image_height, image_width] == output_size:
            return img

        pil_interpolation = pil_modes_mapping[interpolation]
        return img.resize(tuple(size[::-1]), pil_interpolation)


class ToTensor:

    def __call__(self, image):
        image = np.array(image, dtype=np.float32)

        # 将像素值归一化到 [0, 1]
        image /= 255.0

        # 如果图像是灰度图，则扩展维度，使其成为 (C, H, W) 格式
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)

        # 转换为 (C, H, W) 格式
        image = np.transpose(image, (2, 0, 1))
        return image


class Normalize:
    def __init__(self, mean, std):
        """
        初始化Normalize对象。

        :param mean: 均值列表，长度应与图像通道数相同。
        :param std: 标准差列表，长度应与图像通道数相同。
        """
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image):
        """
        对图像进行归一化。

        :param image: 输入图像，形状为(C, H, W)，像素值范围通常在[0, 1]。
        :return: 归一化后的图像，形状为(C, H, W)。
        """
        # 检查输入图像是否符合要求
        if image.shape[0] != len(self.mean) or image.shape[0] != len(self.std):
            raise ValueError("The number of channels in the image must match the length of mean and std lists.")

        # 归一化： (image - mean) / std
        normalized_image = (image - self.mean[:, None, None]) / self.std[:, None, None]

        return normalized_image


class ToPILImage:
    """
    copy from torchvision
    """

    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic) -> Image:
        if not isinstance(pic, np.ndarray):
            raise TypeError(f"pic should be Tensor or ndarray. Got {type(pic)}.")

        if pic.ndim == 2:
            # if 2D image, add channel dimension (HWC)
            pic = np.expand_dims(pic, 2)
        if pic.ndim != 3:
            raise ValueError(f"pic should be 2/3 dimensional. Got {pic.ndim} dimensions.")

        if pic.shape[-1] > 4:
            raise ValueError(f"pic should not have > 4 channels. Got {pic.shape[-1]} channels.")

        npimg = pic

        mode = self.mode

        if np.issubdtype(npimg.dtype, np.floating) and mode != "F":
            npimg = (npimg * 255).astype(np.uint8)

        if npimg.shape[2] == 1:
            expected_mode = None
            npimg = npimg[:, :, 0]
            if npimg.dtype == np.uint8:
                expected_mode = "L"
            elif npimg.dtype == np.int16:
                expected_mode = "I;16" if sys.byteorder == "little" else "I;16B"
            elif npimg.dtype == np.int32:
                expected_mode = "I"
            elif npimg.dtype == np.float32:
                expected_mode = "F"
            if mode is not None and mode != expected_mode:
                raise ValueError(
                    f"Incorrect mode ({mode}) supplied for input type {np.dtype}. Should be {expected_mode}")
            mode = expected_mode

        elif npimg.shape[2] == 2:
            permitted_2_channel_modes = ["LA"]
            if mode is not None and mode not in permitted_2_channel_modes:
                raise ValueError(f"Only modes {permitted_2_channel_modes} are supported for 2D inputs")

            if mode is None and npimg.dtype == np.uint8:
                mode = "LA"

        elif npimg.shape[2] == 4:
            permitted_4_channel_modes = ["RGBA", "CMYK", "RGBX"]
            if mode is not None and mode not in permitted_4_channel_modes:
                raise ValueError(f"Only modes {permitted_4_channel_modes} are supported for 4D inputs")

            if mode is None and npimg.dtype == np.uint8:
                mode = "RGBA"
        else:
            permitted_3_channel_modes = ["RGB", "YCbCr", "HSV"]
            if mode is not None and mode not in permitted_3_channel_modes:
                raise ValueError(f"Only modes {permitted_3_channel_modes} are supported for 3D inputs")
            if mode is None and npimg.dtype == np.uint8:
                mode = "RGB"

        if mode is None:
            raise TypeError(f"Input type {npimg.dtype} is not supported")

        return Image.fromarray(npimg, mode=mode)


class Compose:

    def __init__(self, transformers):
        self.transformers = transformers

    def __call__(self, image):
        for trans in self.transformers:
            image = trans(image)
        return image
