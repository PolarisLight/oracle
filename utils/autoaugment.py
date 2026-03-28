import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.
        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"

# ====== 顶层定义可 pickle 的函数们 ======
def shear_x(img, magnitude, fillcolor=(128, 128, 128)):
    return img.transform(
        img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
        Image.BICUBIC, fillcolor=fillcolor
    )

def shear_y(img, magnitude, fillcolor=(128, 128, 128)):
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
        Image.BICUBIC, fillcolor=fillcolor
    )

def translate_x(img, magnitude, fillcolor=(128, 128, 128)):
    return img.transform(
        img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
        fillcolor=fillcolor
    )

def translate_y(img, magnitude, fillcolor=(128, 128, 128)):
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
        fillcolor=fillcolor
    )

def rotate_with_fill(img, magnitude):
    rot = img.convert("RGBA").rotate(magnitude)
    return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

def rotate(img, magnitude):
    return rotate_with_fill(img, magnitude)

def color(img, magnitude):
    return ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1]))

def posterize(img, magnitude):
    return ImageOps.posterize(img, magnitude)

def solarize(img, magnitude):
    return ImageOps.solarize(img, magnitude)

def contrast(img, magnitude):
    return ImageEnhance.Contrast(img).enhance(1 + magnitude * random.choice([-1, 1]))

def sharpness(img, magnitude):
    return ImageEnhance.Sharpness(img).enhance(1 + magnitude * random.choice([-1, 1]))

def brightness(img, magnitude):
    return ImageEnhance.Brightness(img).enhance(1 + magnitude * random.choice([-1, 1]))

def autocontrast(img, magnitude):
    return ImageOps.autocontrast(img)

def equalize(img, magnitude):
    return ImageOps.equalize(img)

def invert(img, magnitude):
    return ImageOps.invert(img)


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude *
                                         random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude *
                                         random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude *
                                         img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude *
                                         img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }
        func = {
            "shearX": shear_x,
            "shearY": shear_y,
            "translateX": translate_x,
            "translateY": translate_y,
            "rotate": rotate,
            "color": color,
            "posterize": posterize,
            "solarize": solarize,
            "contrast": contrast,
            "sharpness": sharpness,
            "brightness": brightness,
            "autocontrast": autocontrast,
            "equalize": equalize,
            "invert": invert
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


class FourierMix:
    def __init__(self, alpha=0.5, mode="low", p=0.5):
        """
        :param alpha: 频率混合比例 (0~1)，控制混合的强度
        :param mode: 选择混合的频率区域
                     "low" - 只交换低频部分（背景信息）
                     "high" - 只交换高频部分（边缘和纹理）
                     "full" - 全部频域混合
        :param p: 应用增强的概率 (0~1)
        """
        assert mode in ["low", "high", "full"], "mode 必须是 'low', 'high' 或 'full'"
        self.alpha = alpha
        self.mode = mode
        self.p = p  # 应用 FourierMix 的概率

    def __call__(self, img1, img2):
        """
        :param img1: PIL 图像
        :param img2: PIL 图像（用作增强的第二张图）
        :return: 频域混合后的 PIL 图像
        """
        if np.random.rand() > self.p:
            return img1  # 按概率决定是否使用 FourierMix

        # 转换为 numpy 数组
        img1 = np.array(img1)
        img2 = np.array(img2)

        # 处理 RGB 三通道
        img_mixed = self.fft_mix(img1, img2)

        # 转回 PIL
        import  torchvision.transforms as transforms
        return transforms.ToPILImage()(img_mixed)

    def fft_mix(self, img1, img2):
        """
        在频域进行混合
        :param img1: (H, W, 3) 第一张 RGB 图像
        :param img2: (H, W, 3) 第二张 RGB 图像
        :return: 频域混合后的 RGB 图像
        """
        img1, img2 = np.float32(img1), np.float32(img2)
        h, w, c = img1.shape
        mixed_img = np.zeros_like(img1)

        for i in range(c):  # 遍历 RGB 三个通道
            # 计算 FFT
            f1 = np.fft.fft2(img1[:, :, i])
            f2 = np.fft.fft2(img2[:, :, i])
            f1_shift, f2_shift = np.fft.fftshift(f1), np.fft.fftshift(f2)

            # 生成掩码
            mask = np.zeros((h, w))
            if self.mode == "low":
                mask[h//4:3*h//4, w//4:3*w//4] = 1  # 仅保留低频区域
            elif self.mode == "high":
                mask = 1 - mask  # 仅保留高频区域
            elif self.mode == "full":
                mask = np.ones((h, w))  # 交换所有频率成分

            # 进行频域混合
            mixed_fft = self.alpha * f1_shift + (1 - self.alpha) * f2_shift * mask

            # 逆 FFT
            f_ishift = np.fft.ifftshift(mixed_fft)
            img_mixed = np.fft.ifft2(f_ishift)
            mixed_img[:, :, i] = np.abs(img_mixed)  # 取绝对值，避免复数

        return np.clip(mixed_img, 0, 255).astype(np.uint8)

    def __repr__(self):
        return f"FourierMix(alpha={self.alpha}, mode={self.mode}, p={self.p})"