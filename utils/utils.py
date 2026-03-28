import functools
from loguru import logger
import random
import numpy as np
import torch
import os
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
def core_module(cls):
    orig_init = cls.__init__

    @functools.wraps(cls.__init__)
    def new_init(self, *args):
        logger.bind(params=True).info(f"{'='*30}")
        logger.bind(params=True).info(f"Instantiating core module: {cls.__name__}")
        if hasattr(self, 'get_core_params'):
            core_params = self.get_core_params()
            logger.bind(params=True).info("Core hyperparameters and recommended tuning ranges:")
            for param, tuning_range in core_params.items():
                logger.bind(params=True).info(f"  {param} (Recommended range: {tuning_range})")
        logger.bind(params=True).info(f"{'='*30}")
        orig_init(self, *args)
    
    cls.__init__ = new_init
    return cls

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)


def compute_class_accuracies(test_logits, test_targets):
    """
    计算测试集上每个类别的准确度
    :param test_logits: 模型输出的 logits (shape: [batch_size, num_classes])
    :param test_targets: 真实标签 (shape: [batch_size])
    :return: 每个类别的准确度 (shape: [num_classes])
    """
    
    # 通过 argmax 获取预测的类别
    predicted_labels = torch.argmax(test_logits, dim=1)

    # 获取类别数量（根据 logits 的维度）
    num_classes = test_logits.size(1)

    # 用于存储每个类别的准确度
    class_accuracies = torch.zeros(num_classes)

    # 遍历每个类别，计算该类别的准确度
    for class_idx in range(num_classes):
        # 获取当前类别的所有样本
        class_mask = (test_targets == class_idx)

        # 获取当前类别预测为该类别的样本数
        correct_predictions = (predicted_labels[class_mask] == class_idx).sum().item()

        # 计算当前类别的准确度
        total_class_samples = class_mask.sum().item()

        if total_class_samples > 0:
            class_accuracies[class_idx] = correct_predictions / total_class_samples
        else:
            class_accuracies[class_idx] = 0.0  # 如果某类别在测试集中没有样本，准确度为0

    return class_accuracies