import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitAdjustmentLoss(nn.Module):
    def __init__(self, class_freq, tau=1.0, reduction='mean'):
        """
        Logit Adjustment Loss
        Args:
            class_freq (list or np.ndarray or tensor): 每个类别的样本数/频率
            tau (float): 温度系数
            reduction (str): 'mean' | 'sum' | 'none'
        """
        super().__init__()
        freq = torch.tensor(class_freq, dtype=torch.float)
        freq = freq / freq.sum()  # 转为分布
        self.log_prior = tau * torch.log(freq + 1e-12)  # 避免log(0)
        self.reduction = reduction

    def forward(self, logits, targets,s):
        """
        Args:
            logits: [B, C] 模型输出
            targets: [B] 或 one-hot
        """
        device = logits.device
        log_prior = self.log_prior.to(device)

        # 调整 logit
        adj_logits = logits + log_prior*s

        # 交叉熵
        if targets.ndim == 2:  # one-hot
            targets = targets.argmax(dim=1)
        loss = F.cross_entropy(adj_logits, targets, reduction=self.reduction)
        return loss
