from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import os 
import sys
from utils.autoaugment import CIFAR10Policy, Cutout
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import os, math, random
from collections import defaultdict

data_transforms = {
    'base_train': transforms.Compose([
        # transforms.Resize((40,40)),
        # transforms.RandomCrop(32, padding=4),
        transforms.Resize((256,256)),
        transforms.RandomCrop(224, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

    ]),
    # augmentation adopted in balanced meta softmax & NCL
    'advanced_train': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        # CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),

    ]),
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),

    ])
}

def get_data_transforms(args):
    """
    根据 args.img_size 动态构造数据增强 pipeline
    """
    img_size = getattr(args.dataset, "imgsz", 224)   # 默认
    img_size = int(1.1*img_size)
    crop_size = getattr(args.dataset, "imgsz", 224)  # 默认与 img_size 相同

    data_transforms = {
        'base_train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(crop_size, padding=4),
            transforms.ToTensor(),
        ]),
        'advanced_train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
        ]),
        'test': transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
        ])
    }
    return data_transforms

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class OBC306(ImageFolder):
    """
    A custom dataset class for the OBC306 dataset, inheriting from ImageFolder.
    This class can be extended to include additional functionality specific to the OBC306 dataset.
    """
    def __init__(self, args, train=True,):
        train_transform = get_data_transforms(args)['advanced_train'] if train else get_data_transforms(args)['test']
        target_transform = None
        root = args.dataset.label_dir if hasattr(args.dataset, 'label_dir') else 'H:/DatasetD/OBC306'
        root = os.path.join(root, 'train' if train else 'test')
        super(OBC306, self).__init__(root, transform=train_transform, target_transform=target_transform)
        self.targets = to_categorical(self.targets,len(self.classes))
        self.train = train
        # Additional initialization code can be added here if needed
        # self.stat_cls_samples()


    # 统计类样本数
    def stat_cls_samples(self):
        cls_num_list = self.get_cls_num_list()
        for i, num in enumerate(cls_num_list):
            print(f"Class {i}: {num} samples")
        print(f'Average samples per class: {sum(cls_num_list) / len(cls_num_list):.2f}')

    def get_cls_num_list(self):
        """
        Get the number of samples for each class in the dataset.
        Returns:
            A list containing the number of samples for each class.
        """
        cls_num_list = [0] * len(self.classes)
        for _, label in self.samples:
            cls_num_list[label] += 1
        return cls_num_list
    def __getitem__(self, index):
        data = super().__getitem__(index)
        label_onehot = to_categorical(data[1], num_classes=len(self.classes))
        my_data_format = {'image': data[0], 'label': label_onehot}
        return my_data_format
    
    def __len__(self):
        return len(self.samples)

    def get_tail_classes(self, tail_num=100):
        self.tail_classes = []
        for i, num in enumerate(self.get_cls_num_list()):
            if num < tail_num:  # Assuming tail_num is the threshold for tail classes
                self.tail_classes.append(i)
        return self.tail_classes


class OBC306_Mixup(ImageFolder):
    """
    A custom dataset class for the OBC306 dataset, inheriting from ImageFolder.
    Now supports in-dataset Mixup (only when train=True).
    """
    def __init__(self, args, train=True, mixup_alpha=0.4, mixup_prob=0.5):
        train_transform = data_transforms['base_train'] if train else data_transforms['test']
        target_transform = None
        root = args.dataset.label_dir if hasattr(args.dataset, 'label_dir') else 'H:/DatasetD/OBC306'
        root = os.path.join(root, 'train' if train else 'test')
        super(OBC306_Mixup, self).__init__(root, transform=train_transform, target_transform=target_transform)

        self.num_classes = len(self.classes)
        self.targets = to_categorical(self.targets, self.num_classes)
        self.train = train

        # ---- Mixup 参数 ----
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
    def __len__(self):
        return len(self.samples)

    def get_cls_num_list(self):
        """
        Get the number of samples for each class in the dataset.
        Returns:
            A list containing the number of samples for each class.
        """
        cls_num_list = [0] * len(self.classes)
        for _, label in self.samples:
            cls_num_list[label] += 1
        return cls_num_list

    def get_tail_classes(self, tail_num=100):
        self.tail_classes = []
        for i, num in enumerate(self.get_cls_num_list()):
            if num < tail_num:  # Assuming tail_num is the threshold for tail classes
                self.tail_classes.append(i)
        return self.tail_classes

    # ------------------------------
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target = to_categorical(target, self.num_classes)

        # 仅训练时启用 Mixup
        if self.train and random.random() < self.mixup_prob:
            j = random.randrange(len(self.samples))
            img2, target2 = super().__getitem__(j)
            target2 = to_categorical(target2, self.num_classes)

            # Beta 分布采样 λ
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()

            # Mixup
            img = lam * img + (1 - lam) * img2
            target = lam * target + (1 - lam) * target2

        return {'image': img, 'label': target}

    def __len__(self):
        return len(self.samples)


# ---- CutMix Dataset ----
class OBC306_CutMix(ImageFolder):
    """
    A custom dataset class for the OBC306 dataset.
    Supports in-dataset CutMix (only when train=True).
    """
    def __init__(self, args, train=True, cutmix_alpha=1.0, cutmix_prob=0.5):
        train_transform = data_transforms['base_train'] if train else data_transforms['test']
        target_transform = None
        root = args.dataset.label_dir if hasattr(args.dataset, 'label_dir') else 'H:/DatasetD/OBC306'
        root = os.path.join(root, 'train' if train else 'test')
        super(OBC306_CutMix, self).__init__(root, transform=train_transform, target_transform=target_transform)

        self.num_classes = len(self.classes)
        self.targets = to_categorical(self.targets, self.num_classes)
        self.train = train

        # CutMix 参数
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob

    def __len__(self):
        return len(self.samples)

    def rand_bbox(self, size, lam):
        """生成CutMix区域"""
        W, H = size[1], size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # 随机中心点
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        return x1, y1, x2, y2

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target = to_categorical(target, self.num_classes)

        # 仅训练时启用 CutMix
        if self.train and random.random() < self.cutmix_prob:
            j = random.randrange(len(self.samples))
            img2, target2 = super().__getitem__(j)
            target2 = to_categorical(target2, self.num_classes)

            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            x1, y1, x2, y2 = self.rand_bbox(img.size(), lam)

            # CutMix 替换
            img[:, y1:y2, x1:x2] = img2[:, y1:y2, x1:x2]

            # 按区域面积修正 λ
            lam = 1 - ((x2 - x1) * (y2 - y1) / (img.size(-1) * img.size(-2)))
            target = lam * target + (1 - lam) * target2

        return {'image': img, 'label': target}

    def get_cls_num_list(self):
        cls_num_list = [0] * len(self.classes)
        for _, label in self.samples:
            cls_num_list[label] += 1
        return cls_num_list

    def get_tail_classes(self, tail_num=100):
        self.tail_classes = []
        for i, num in enumerate(self.get_cls_num_list()):
            if num < tail_num:
                self.tail_classes.append(i)
        return self.tail_classes

class OBC306_ReMix(ImageFolder):
    """
    A custom dataset class for the OBC306 dataset, inheriting from ImageFolder.
    Supports ReMix (rebalanced mixup): tail classes get higher weight in mixup.
    """
    def __init__(self, args, train=True, mixup_alpha=0.4, mixup_prob=0.5, tail_threshold=100):
        train_transform = data_transforms['base_train'] if train else data_transforms['test']
        target_transform = None
        root = args.dataset.label_dir if hasattr(args.dataset, 'label_dir') else 'H:/DatasetD/OBC306'
        root = os.path.join(root, 'train' if train else 'test')
        super(OBC306_ReMix, self).__init__(root, transform=train_transform, target_transform=target_transform)

        self.num_classes = len(self.classes)
        self.targets = to_categorical(self.targets, self.num_classes)
        self.train = train

        # ---- Mixup 参数 ----
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.tail_threshold = tail_threshold

        # 类别样本数 & 频率（在 ReMix 里用）
        self.cls_num_list = self.get_cls_num_list()
        self.class_freq = torch.tensor(self.cls_num_list, dtype=torch.float)
        self.class_freq = self.class_freq / self.class_freq.sum()

    def __len__(self):
        return len(self.samples)

    def get_cls_num_list(self):
        cls_num_list = [0] * len(self.classes)
        for _, label in self.samples:
            cls_num_list[label] += 1
        return cls_num_list

    def get_tail_classes(self, tail_num=100):
        return [i for i, num in enumerate(self.get_cls_num_list()) if num < tail_num]

    # ------------------------------
    def __getitem__(self, index):
        img, target_idx = super().__getitem__(index)
        target = to_categorical(target_idx, self.num_classes)

        if self.train and random.random() < self.mixup_prob:
            j = random.randrange(len(self.samples))
            img2, target_idx2 = super().__getitem__(j)
            target2 = to_categorical(target_idx2, self.num_classes)

            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()

            # ==== ReMix 调整 ====
            # 如果两个样本来自 head / tail，则提高 tail 的权重
            freq1 = self.class_freq[target_idx]
            freq2 = self.class_freq[target_idx2]

            # 权重反比于频率（样本越少，权重越大）
            w1 = 1.0 / (freq1 + 1e-6)
            w2 = 1.0 / (freq2 + 1e-6)

            # 用加权 λ 替换原始 λ
            lam = (lam * w1) / (lam * w1 + (1 - lam) * w2)
            lam = float(lam)

            # === Mixup ===
            img = lam * img + (1 - lam) * img2
            target = lam * target + (1 - lam) * target2

        return {'image': img, 'label': target}

class OBC306_MorphoMix(ImageFolder):
    """
    A custom dataset class for the OBC306 dataset, inheriting from ImageFolder.
    Now supports in-dataset MorphoMix (class-internal mix + optional inter-class dual perturbation).
    """
    def __init__(self, args, train=True,
                 preset='standard',      # 'light' | 'standard' | 'strong'
                 use_inter=True,         # 是否启用类间对偶扰动
                 onehot=True):           # 是否以 soft one-hot 返回（便于和KD/CE兼容）
        train_transform = data_transforms['base_train'] if train else data_transforms['test']
        target_transform = None
        root = args.dataset.label_dir if hasattr(args.dataset, 'label_dir') else 'H:/DatasetD/OBC306'
        root = os.path.join(root, 'train' if train else 'test')
        super(OBC306_MorphoMix, self).__init__(root, transform=train_transform, target_transform=target_transform)

        self.train = train
        self.num_classes = len(self.classes)
        self.onehot = onehot
        self.tail_classes = []

        # ---- MorphoMix 配置（少参数预设） ----
        self.morphomix_enabled = train  # 只在训练集默认启用；验证/测试自动关闭
        self.use_inter = use_inter
        cfgs = {
            'light':   dict(p_intra=0.20, p_inter=0.10, alpha=0.50, morph_p=0.50, blur_p=0.20, cut_p=0.20, edge_crop=1),
            'standard':dict(p_intra=0.30, p_inter=0.20, alpha=0.40, morph_p=0.70, blur_p=0.30, cut_p=0.30, edge_crop=1),
            'strong':  dict(p_intra=0.40, p_inter=0.30, alpha=0.30, morph_p=0.80, blur_p=0.40, cut_p=0.40, edge_crop=2),
        }
        self.mm_cfg = cfgs.get(preset, cfgs['standard'])
        if not self.use_inter:
            self.mm_cfg['p_inter'] = 0.0

        # ---- 为类内/类间配对预建索引表（只做一次）----
        self.cls2idx = defaultdict(list)
        for i, (_, y) in enumerate(self.samples):
            self.cls2idx[int(y)].append(i)
        for c in range(self.num_classes):
            if len(self.cls2idx[c]) == 0:
                self.cls2idx[c] = [0]  # 占位，防越界

    # 可选：运行时开关
    def enable_morphomix(self, enabled=True):
        self.morphomix_enabled = bool(enabled)

    # -------------------------
    # 形态/几何基础算子（轻量实现）
    @staticmethod
    def _erode(x, k=1):
        # x: [C,H,W], 近似腐蚀（细化笔画）
        xpad = F.pad(x.unsqueeze(0), (k,k,k,k), mode='replicate')
        return (-F.max_pool2d(-xpad, kernel_size=2*k+1, stride=1)).squeeze(0)

    @staticmethod
    def _dilate(x, k=1):
        # x: [C,H,W], 近似膨胀（加粗笔画）
        xpad = F.pad(x.unsqueeze(0), (k,k,k,k), mode='replicate')
        return (F.max_pool2d(xpad, kernel_size=2*k+1, stride=1)).squeeze(0)

    @staticmethod
    def _gblur(x, k=3, sigma=0.8):
        # x: [C,H,W] 轻高斯模糊
        dev = x.device
        ax = torch.arange(k, device=dev) - (k-1)/2
        k1 = torch.exp(-0.5*(ax/sigma)**2); k1 = k1/k1.sum()
        kx = k1.view(1,1,1,k); ky = k1.view(1,1,k,1)
        C = x.size(0)
        x = F.conv2d(x.unsqueeze(0), kx.expand(C,1,1,k), padding=(0,k//2), groups=C)
        x = F.conv2d(x, ky.expand(C,1,k,1), padding=(k//2,0), groups=C).squeeze(0)
        return x

    @staticmethod
    def _thin_cutout(x, num=2):
        # x: [C,H,W] 细裂遮挡
        C,H,W = x.shape
        m = torch.ones((H,W), dtype=torch.bool, device=x.device)
        for _ in range(num):
            length = random.randint(max(1,H//8), max(2,H//4))
            width  = random.randint(1, 3)
            y0 = random.randint(0, H-1); x0 = random.randint(0, W-1)
            th = random.uniform(0, 2*math.pi)
            dy, dx = math.sin(th), math.cos(th)
            for t in range(length):
                yy = int(y0 + dy*t); xx = int(x0 + dx*t)
                if 0<=yy<H and 0<=xx<W:
                    yy0 = max(0, yy-width//2); yy1 = min(H, yy+(width+1)//2)
                    xx0 = max(0, xx-width//2); xx1 = min(W, xx+(width+1)//2)
                    m[yy0:yy1, xx0:xx1] = False
        return x * m

    @staticmethod
    def _affine(x, max_deg=5, max_shift=3, min_scale=0.97, max_scale=1.03):
        # x: [C,H,W] 轻仿射
        C,H,W = x.shape
        ang = random.uniform(-max_deg, max_deg) * math.pi/180
        s   = random.uniform(min_scale, max_scale)
        tx  = random.uniform(-max_shift, max_shift)*2.0/W
        ty  = random.uniform(-max_shift, max_shift)*2.0/H
        a = s*math.cos(ang); b = -s*math.sin(ang)
        c = s*math.sin(ang); d =  s*math.cos(ang)
        theta = torch.tensor([[a,b,tx],[c,d,ty]], device=x.device, dtype=x.dtype).unsqueeze(0)
        grid  = F.affine_grid(theta, size=(1,C,H,W), align_corners=False)
        return F.grid_sample(x.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze(0)

    @staticmethod
    def _edge_crop(x, px=1):
        if px <= 0: return x
        C,H,W = x.shape
        x = x[:, px:H-px, px:W-px]
        return F.pad(x, (px,px,px,px), mode='replicate')

    # ---- 采样同类/异类索引 ----
    def _sample_same_class(self, y, self_idx):
        pool = self.cls2idx[int(y)]
        if len(pool) <= 1: return self_idx
        j = random.choice(pool)
        return j if j != self_idx else random.choice(pool)

    def _sample_diff_class(self, y):
        # 随机找一个异类，最多重试若干次
        for _ in range(5):
            c = random.randrange(self.num_classes)
            if c != int(y) and len(self.cls2idx[c]) > 0:
                return random.choice(self.cls2idx[c])
        # 兜底：任取一个非空异类
        non_empty = [c for c in range(self.num_classes) if len(self.cls2idx[c])>0 and c!=int(y)]
        c = random.choice(non_empty) if non_empty else int(y)
        return random.choice(self.cls2idx[c])

    # -------------------------
    def __getitem__(self, index):
        # 调用父类：得到 transform 后的 tensor 和 int label
        img, y = super().__getitem__(index)   # img: [C,H,W] in [0,1], y:int
        assert torch.is_tensor(img) and img.dtype.is_floating_point
        y = int(y)
        img = img.clone()

        # 默认 one-hot / hard label 输出
        label_out = to_categorical(y, self.num_classes) if self.onehot else torch.tensor(y, dtype=torch.long)

        # 仅训练时做 MorphoMix；验证/测试直接返回
        if not (self.train and self.morphomix_enabled) and y in getattr(self, 'tail_classes', []):
            return {'image': img, 'label': label_out}

        cfg = self.mm_cfg

        # 轻边裁：模拟残片缺边
        img = self._edge_crop(img, cfg['edge_crop'])

        # ---- 类内 Mix（按概率触发；同类标签不变/one-hot 不变）
        if random.random() < cfg['p_intra']:
            j = self._sample_same_class(y, index)
            img_j, _ = super().__getitem__(j)
            img_j = img_j.to(img.dtype)
            img_j = self._edge_crop(img_j, cfg['edge_crop'])
            mix_rate = float(torch.distributions.Beta(cfg['alpha'], cfg['alpha']).sample(()).item())
            img = mix_rate * img + (1.0 - mix_rate) * img_j

        # ---- 类间对偶扰动 + 软标签（按概率触发）
        if self.use_inter and (random.random() < cfg['p_inter']):
            k = self._sample_diff_class(y)
            img_k, yk = super().__getitem__(k)
            yk = int(yk)
            img_a = img.clone()
            img_b = img_k.to(img.dtype).clone()

            # a：形态扰动（erode/dilate/blur/细裂）
            if random.random() < cfg['morph_p']:
                img_a = (self._erode(img_a, 1) if random.random() < 0.5 else self._dilate(img_a, 1))
            if random.random() < cfg['blur_p']:
                img_a = self._gblur(img_a, 3, 0.8)
            if random.random() < cfg['cut_p']:
                img_a = self._thin_cutout(img_a, num=2)

            # b：轻仿射
            img_b = self._edge_crop(img_b, cfg['edge_crop'])
            img_b = self._affine(img_b)

            # 混合 + 软标签
            mix_rate = float(torch.distributions.Beta(cfg['alpha'], cfg['alpha']).sample(()).item())
            img = mix_rate * img_a + (1.0 - mix_rate) * img_b
            if self.onehot:
                ya = to_categorical(y,  self.num_classes)
                yb = to_categorical(yk, self.num_classes)
                label_out = mix_rate * ya + (1.0 - mix_rate) * yb
            else:
                # 强制硬标签的话，保持 y 不变；更建议 onehot=True
                pass

        img = img.clamp(0, 1)
        return {'image': img, 'label': label_out}

    # ------- 你原有的方法保持不变 -------
    def stat_cls_samples(self):
        cls_num_list = self.get_cls_num_list()
        for i, num in enumerate(cls_num_list):
            print(f"Class {i}: {num} samples")
        print(f'Average samples per class: {sum(cls_num_list) / len(cls_num_list):.2f}')

    def get_cls_num_list(self):
        cls_num_list = [0] * len(self.classes)
        for _, label in self.samples:
            cls_num_list[label] += 1
        return cls_num_list

    def __len__(self):
        return len(self.samples)

    def get_tail_classes(self, tail_num=100):
        self.tail_classes = []
        for i, num in enumerate(self.get_cls_num_list()):
            if num < tail_num:
                self.tail_classes.append(i)
        return self.tail_classes

class OBC306_freq_adaptive(ImageFolder):
    """
    Custom dataset class for OBC306, with in-dataset MorphoMix augmentation.
    Now supports frequency-adaptive augmentation (rarer classes -> stronger augmentation).
    """

    def __init__(self, args, train=True,
                 preset='standard',
                 use_inter=True,
                 onehot=True,
                 adapt_k=100):  # 控制增强强度，值越大 → 尾类增强更强
        train_transform = data_transforms['advanced_train'] if train else data_transforms['test']
        target_transform = None
        root = args.dataset.label_dir if hasattr(args.dataset, 'label_dir') else 'H:/DatasetD/OBC306'
        root = os.path.join(root, 'train' if train else 'test')
        super(OBC306_freq_adaptive, self).__init__(root, transform=train_transform, target_transform=target_transform)

        self.train = train
        self.num_classes = len(self.classes)
        self.onehot = onehot
        self.adapt_k = adapt_k

        # ---- MorphoMix 配置 ----
        self.morphomix_enabled = train
        self.use_inter = use_inter
        cfgs = {
            'light':   dict(p_intra=0.20, p_inter=0.10, alpha=0.50, morph_p=0.50, blur_p=0.20, cut_p=0.20, edge_crop=1),
            'standard':dict(p_intra=0.30, p_inter=0.20, alpha=0.40, morph_p=0.70, blur_p=0.30, cut_p=0.30, edge_crop=1),
            'strong':  dict(p_intra=0.40, p_inter=0.30, alpha=0.30, morph_p=0.80, blur_p=0.40, cut_p=0.40, edge_crop=2),
        }
        self.mm_cfg = cfgs.get(preset, cfgs['standard'])
        if not self.use_inter:
            self.mm_cfg['p_inter'] = 0.0

        # ---- 类频统计 ----
        self.cls2idx = defaultdict(list)
        for i, (_, y) in enumerate(self.samples):
            self.cls2idx[int(y)].append(i)
        self.cls_freq = {c: len(self.cls2idx[c]) for c in range(self.num_classes)}
        self.adapt_p = {c: min(0.5, float(self.adapt_k) / max(1, self.cls_freq[c])) for c in range(self.num_classes)}
        # 小样本类 → 概率高，大样本类 → 概率低

    # 可选：运行时开关
    def enable_morphomix(self, enabled=True):
        self.morphomix_enabled = bool(enabled)

    # -------------------------
    # 基础形态变换算子
    @staticmethod
    def _erode(x, k=1):
        xpad = F.pad(x.unsqueeze(0), (k,k,k,k), mode='replicate')
        return (-F.max_pool2d(-xpad, kernel_size=2*k+1, stride=1)).squeeze(0)

    @staticmethod
    def _dilate(x, k=1):
        xpad = F.pad(x.unsqueeze(0), (k,k,k,k), mode='replicate')
        return (F.max_pool2d(xpad, kernel_size=2*k+1, stride=1)).squeeze(0)

    @staticmethod
    def _gblur(x, k=3, sigma=0.8):
        dev = x.device
        ax = torch.arange(k, device=dev) - (k-1)/2
        k1 = torch.exp(-0.5*(ax/sigma)**2); k1 = k1/k1.sum()
        kx = k1.view(1,1,1,k); ky = k1.view(1,1,k,1)
        C = x.size(0)
        x = F.conv2d(x.unsqueeze(0), kx.expand(C,1,1,k), padding=(0,k//2), groups=C)
        x = F.conv2d(x, ky.expand(C,1,k,1), padding=(k//2,0), groups=C).squeeze(0)
        return x

    @staticmethod
    def _thin_cutout(x, num=2):
        C,H,W = x.shape
        m = torch.ones((H,W), dtype=torch.bool, device=x.device)
        for _ in range(num):
            length = random.randint(max(1,H//8), max(2,H//4))
            width  = random.randint(1, 3)
            y0 = random.randint(0, H-1); x0 = random.randint(0, W-1)
            th = random.uniform(0, 2*math.pi)
            dy, dx = math.sin(th), math.cos(th)
            for t in range(length):
                yy = int(y0 + dy*t); xx = int(x0 + dx*t)
                if 0<=yy<H and 0<=xx<W:
                    yy0 = max(0, yy-width//2); yy1 = min(H, yy+(width+1)//2)
                    xx0 = max(0, xx-width//2); xx1 = min(W, xx+(width+1)//2)
                    m[yy0:yy1, xx0:xx1] = False
        return x * m

    @staticmethod
    def _affine(x, max_deg=5, max_shift=3, min_scale=0.97, max_scale=1.03):
        C,H,W = x.shape
        ang = random.uniform(-max_deg, max_deg) * math.pi/180
        s   = random.uniform(min_scale, max_scale)
        tx  = random.uniform(-max_shift, max_shift)*2.0/W
        ty  = random.uniform(-max_shift, max_shift)*2.0/H
        a = s*math.cos(ang); b = -s*math.sin(ang)
        c = s*math.sin(ang); d =  s*math.cos(ang)
        theta = torch.tensor([[a,b,tx],[c,d,ty]], device=x.device, dtype=x.dtype).unsqueeze(0)
        grid  = F.affine_grid(theta, size=(1,C,H,W), align_corners=False)
        return F.grid_sample(x.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze(0)

    @staticmethod
    def _edge_crop(x, px=1):
        if px <= 0: return x
        C,H,W = x.shape
        x = x[:, px:H-px, px:W-px]
        return F.pad(x, (px,px,px,px), mode='replicate')

    # ---- 采样同类/异类 ----
    def _sample_same_class(self, y, self_idx):
        pool = self.cls2idx[int(y)]
        if len(pool) <= 1: return self_idx
        j = random.choice(pool)
        return j if j != self_idx else random.choice(pool)

    def _sample_diff_class(self, y):
        for _ in range(5):
            c = random.randrange(self.num_classes)
            if c != int(y) and len(self.cls2idx[c]) > 0:
                return random.choice(self.cls2idx[c])
        non_empty = [c for c in range(self.num_classes) if len(self.cls2idx[c])>0 and c!=int(y)]
        c = random.choice(non_empty) if non_empty else int(y)
        return random.choice(self.cls2idx[c])

    # -------------------------
    def __getitem__(self, index):
        img, y = super().__getitem__(index)
        y = int(y)
        img = img.clone()

        label_out = to_categorical(y, self.num_classes) if self.onehot else torch.tensor(y, dtype=torch.long)

        if not (self.train and self.morphomix_enabled):
            return {'image': img, 'label': label_out}

        cfg = self.mm_cfg
        # ---- 根据类别频率决定是否增强 ----
        if random.random() > self.adapt_p[y]:
            return {'image': img, 'label': label_out}

        # 轻边裁
        img = self._edge_crop(img, cfg['edge_crop'])

        # ---- 类内增强 ----
        if random.random() < cfg['p_intra']:
            j = self._sample_same_class(y, index)
            img_j, _ = super().__getitem__(j)
            img_j = self._edge_crop(img_j.to(img.dtype), cfg['edge_crop'])
            mix_rate = float(torch.distributions.Beta(cfg['alpha'], cfg['alpha']).sample(()).item())
            img = mix_rate * img + (1.0 - mix_rate) * img_j

        # ---- 类间增强 ----
        if self.use_inter and (random.random() < cfg['p_inter']):
            k = self._sample_diff_class(y)
            img_k, yk = super().__getitem__(k); yk = int(yk)
            img_a = img.clone()
            img_b = self._edge_crop(img_k.to(img.dtype).clone(), cfg['edge_crop'])
            img_b = self._affine(img_b)

            if random.random() < cfg['morph_p']:
                img_a = (self._erode(img_a,1) if random.random()<0.5 else self._dilate(img_a,1))
            if random.random() < cfg['blur_p']:
                img_a = self._gblur(img_a, 3, 0.8)
            if random.random() < cfg['cut_p']:
                img_a = self._thin_cutout(img_a, num=2)

            mix_rate = float(torch.distributions.Beta(cfg['alpha'], cfg['alpha']).sample(()).item())
            img = mix_rate * img_a + (1.0 - mix_rate) * img_b
            if self.onehot:
                ya = to_categorical(y,  self.num_classes)
                yb = to_categorical(yk, self.num_classes)
                label_out = mix_rate * ya + (1.0 - mix_rate) * yb

        img = img.clamp(0, 1)
        return {'image': img, 'label': label_out}

    # ------- 原有工具函数 -------
    def stat_cls_samples(self):
        cls_num_list = self.get_cls_num_list()
        for i, num in enumerate(cls_num_list):
            print(f"Class {i}: {num} samples")
        print(f'Average samples per class: {sum(cls_num_list) / len(cls_num_list):.2f}')

    def get_cls_num_list(self):
        cls_num_list = [0] * len(self.classes)
        for _, label in self.samples:
            cls_num_list[label] += 1
        return cls_num_list

    def __len__(self):
        return len(self.samples)

    def get_tail_classes(self, tail_num=100):
        return [i for i,num in enumerate(self.get_cls_num_list()) if num < tail_num]




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OBC306 Dataset Example")
    # parser.add_argument('--root', type=str, required=True, help='Root directory of the dataset')
    args = parser.parse_args()

    # root = "H:\DatasetD\OBC306"
    # dataset = OBC306(root=root, transform=None)
    # print(f"Number of classes: {len(dataset.classes)}")
    # print(f"Class names: {dataset.classes}")
    # print(f"Number of samples per class: {dataset.get_cls_num_list()}")
    # print(f"Total number of samples: {len(dataset)}")
    # print(f"Tail classes: {dataset.tail_classes}")

    # 调用函数
    data_dir = 'H:\DatasetD\OBC306'  # 数据集根目录
    output_dir = 'H:\DatasetD\OBC306_filtered'  # 输出目录
    create_data_split(data_dir, output_dir, train_ratio=0.7, min_samples=2)