import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicClassWeightLoss(nn.Module):
    def __init__(self, num_classes, alpha=1.0, beta=0.1, epsilon=0.2):
        """
        初始化类
        :param num_classes: 类别总数
        :param alpha: 权重调整的超参数，控制相对优势的影响
        :param beta: KL散度正则化的系数
        :param epsilon: 用于PPO的裁剪参数
        """
        super(DynamicClassWeightLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.device =  ('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化类别权重
        self.class_weights = torch.ones(num_classes).to(self.device)

    def calculate_relative_advantage(self, class_losses):
        """
        计算类别的相对优势
        :param class_losses: 各类别的损失
        :return: 每个类别的相对优势
        """
        mean_loss = class_losses.mean()
        std_loss = class_losses.std()
        return (class_losses - mean_loss) / std_loss

    def update_class_weights(self, relative_advantages):
        """
        根据相对优势更新类别权重
        :param relative_advantages: 每个类别的相对优势
        :return: 更新后的类别权重
        """
        with torch.no_grad():
            self.class_weights = torch.exp(self.alpha * relative_advantages)
            self.class_weights = self.class_weights / self.class_weights.sum()  # 保证总和为1

    def forward(self, outputs, targets):
        """
        计算动态加权的交叉熵损失
        :param outputs: 模型输出的logits
        :param targets: 真实标签
        :return: 加权交叉熵损失
        """
        batch_size = outputs.size(0)
        if targets.shape[1]>1:
            targets = torch.argmax(targets,dim=1)
        
        # 计算每个类别的交叉熵损失
        class_losses = F.cross_entropy(outputs, targets, reduction='none')  # 每个样本的交叉熵损失
        
        # 获取one-hot编码的目标标签
        one_hot_targets = F.one_hot(targets, num_classes=self.num_classes).float()

        # 计算每个类别的损失，通过矩阵相乘而不是循环
        # class_losses.shape = (batch_size,)
        # one_hot_targets.shape = (batch_size, num_classes)
        # 这里我们通过矩阵相乘来计算每个类别的损失
        class_losses_per_class = (class_losses.unsqueeze(1) * one_hot_targets).sum(dim=0) / batch_size

        # 计算相对优势并更新类别权重
        relative_advantages = self.calculate_relative_advantage(class_losses_per_class)
        self.update_class_weights(relative_advantages)

        # 计算加权交叉熵损失
        weighted_loss = F.cross_entropy(outputs, targets, weight=self.class_weights)

        # 可选的KL散度正则化（如果需要）
        kl_loss = self.beta * torch.sum(self.class_weights * torch.log(self.class_weights))

        return weighted_loss + kl_loss


import torch
import torch.nn as nn
from collections import deque, defaultdict
from typing import Optional  # ← 兼容 <3.10

class GroupRelativeLossMem(nn.Module):
    """
    长尾视觉识别用：带历史记忆的 Group-Relative Loss
    * 支持 index-label 与 one-/soft-hot
    * 无需保证 “同类 ≥2 张”
    """

    def __init__(
        self,
        base_loss_fn: Optional[nn.Module] = None,   # ← 改
        *,
        lambda_: float = 2.0,
        eps: float = 0.3,
        memory_size: int = 20,
        ema_momentum: float = 0.9,
        class_weights: Optional[torch.Tensor] = None,   # ← 改
        reduction: str = "mean",
    ):
        super().__init__()
        self.base_loss_fn = (
            base_loss_fn
            if base_loss_fn is not None
            else nn.CrossEntropyLoss(reduction="none")
        )
        self.lambda_ = float(lambda_)
        self.eps = float(eps)
        self.memory_size = memory_size
        self.m = ema_momentum
        self.reduction = reduction

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None  # type: ignore

        self.mem_queue: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=memory_size)
        )
        self.register_buffer("ema_loss", torch.zeros(0))

    # ---------- helpers ----------
    def _ensure_ema_size(self, cls: int, device):
        if self.ema_loss.numel() <= cls:
            pad = cls + 1 - self.ema_loss.numel()
            self.ema_loss = torch.cat(
                [self.ema_loss, torch.zeros(pad, device=device)]
            )

    def _update_history(self, cls: int, value: float, device):
        self._ensure_ema_size(cls, device)
        self.ema_loss[cls] = self.m * self.ema_loss[cls] + (1 - self.m) * value
        self.mem_queue[cls].append(value)

    def _get_baseline(self, cls: int, global_mean: torch.Tensor):
        q = self.mem_queue[cls]
        if len(q) >= 2:
            return sum(q) / len(q)                     # 队列均值
        elif self.ema_loss.numel() > cls and self.ema_loss[cls] > 0:
            return self.ema_loss[cls].item()             # EMA
        else:
            return global_mean.item()                    # 全局回退

    # ---------- forward ----------
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [B, C]
        targets : [B]  或 [B, C]  (one-hot / soft-label)
        """
        base_loss = self.base_loss_fn(logits, targets)      # 必须 [B]

        # 取分组标签
        if targets.dim() == 1:
            cls_index = targets
        elif targets.dim() == 2:
            cls_index = targets.argmax(dim=1)
        else:
            raise ValueError("targets 的维度应为 1 或 2 (one-hot)。")

        if self.class_weights is not None:
            base_loss = base_loss * self.class_weights[cls_index]

        device = base_loss.device
        global_mean = base_loss.mean().detach()
        baselines = torch.empty_like(base_loss)

        # 遍历 batch 中出现的类别
        for cls in torch.unique(cls_index):
            cls = int(cls)
            idx = (cls_index == cls).nonzero(as_tuple=False).squeeze(1)
            cls_batch_mean = base_loss[idx].mean().item()

            
            # 取 baseline
            baselines[idx] = self._get_baseline(cls, global_mean)

            # 更新历史
            self._update_history(cls, cls_batch_mean, device)


        # 优势 & 缩放
        advantage = baselines - base_loss
        scale = (1 + self.lambda_ * advantage).clamp_(1 - self.eps, 1 + self.eps)

        loss = scale * base_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

# class GIBALoss(nn.Module):
#     def __init__(self, class_freq: List[float], tail_classes: List[int], 
#                  group_size: int = 3, sigma: float = 0.1, 
#                  alpha: float = 1.0, beta: float = 1.0, 
#                  lambda_tbsr: float = 0.1):
#         super(GIBALoss, self).__init__()
#         self.class_freq = torch.tensor(class_freq, dtype=torch.float32)
#         self.tail_classes = tail_classes
#         self.group_size = group_size
#         self.sigma = sigma
#         self.alpha = alpha
#         self.beta = beta
#         self.lambda_tbsr = lambda_tbsr

#     def compute_boundary_distances(self, group_pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         batch_size = targets.size(0)
#         distances = torch.zeros(self.group_size, batch_size).to(targets.device)
        
#         for j in range(self.group_size):
#             true_logits = group_pred[j, torch.arange(batch_size), targets]  
#             other_logits = group_pred[j].clone()
#             other_logits[torch.arange(batch_size), targets] = float('-inf')  
#             max_other_logits = torch.max(other_logits, dim=-1)[0]  
#             distances[j] = true_logits - max_other_logits  
        
#         return distances

#     def compute_weights(self, distances: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         distance_mean = distances.mean(dim=0)  
#         # Sigmoid-like transformation for difficulty (high for negative means)
#         difficulty = 1.0 / (1.0 + torch.exp(distance_mean))  
#         class_weights = 1.0 / torch.log(1 + self.class_freq).to(targets.device)
#         freq_weights = class_weights[targets]  
#         weights = self.alpha * difficulty + self.beta * freq_weights  
#         weights = torch.clamp(weights, min=0.0)  
#         return weights / weights.sum()  

#     def compute_tbsr_loss(self, distances: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         is_tail = torch.tensor([1 if t.item() in self.tail_classes else 0 for t in targets]).to(targets.device)
#         distance_var = torch.var(distances, dim=0)  
#         tbsr_loss = torch.mean(is_tail * distance_var)
#         return tbsr_loss

#     def forward(self, pred: torch.Tensor, feature: torch.Tensor, fc: nn.Module, targets: torch.Tensor) -> torch.Tensor:
#         device = pred.device
#         batch_size = pred.size(0)
#         targets = torch.argmax(targets, dim=1) if targets.dim() > 1 else targets.squeeze()
#         # Generate group predictions
#         group_pred = torch.zeros(self.group_size, batch_size, pred.size(-1)).to(device)
#         for j in range(self.group_size):
#             noise = torch.randn_like(feature) * self.sigma
#             perturbed_feature = feature + noise  
#             group_pred[j] = fc(perturbed_feature)  

#         # Compute boundary distances
#         distances = self.compute_boundary_distances(group_pred, targets)  

#         # Compute weights
#         weights = self.compute_weights(distances, targets)

#         # Compute TBSR loss
#         tbsr_loss = self.compute_tbsr_loss(distances, targets)

#         # Compute total loss
#         class_loss = F.cross_entropy(pred, targets, reduction='none')  
#         weighted_class_loss = (weights * class_loss).mean()
#         total_loss = weighted_class_loss + self.lambda_tbsr * tbsr_loss

#         return total_loss
    

class GIBALossV2(nn.Module):
    """
    修复点：
    - 用 logit margin（不走 softmax）
    - 样本权重对 CE 停止梯度；用 sum/sum 归一
    - tail-only 的方差/均值 margin 正则
    - 频率权重用 Effective Number；整体权重均值归一
    - 噪声按特征统计自适应
    - targets 兼容 one-hot / int
    - 全量向量化 & 设备安全
    """
    def __init__(
        self,
        class_freq,                # List[float] or 1D tensor length = C
        tail_classes,              # List[int]
        group_size: int = 4,
        sigma: float = 0.05,
        alpha: float = 1.0,        # 难度项指数
        beta: float = 1.0,         # 频率项指数
        lambda_tbsr: float = 0.2,
        mu_margin: float = 0.5,    # 均值 margin 惩罚权重
        target_margin: float = 0.1,# tail 样本期望 margin
        k_diff: float = 2.0,       # 难度项 sigmoid 斜率
        logit_temp: float = 1.0,   # margin 温度（除数）
        beta_eff: float = 0.999,   # Effective Number 超参
        stopgrad_weights: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        class_freq = torch.as_tensor(class_freq, dtype=torch.float32)
        num_classes = class_freq.numel()

        # Effective Number -> 频率权重（越少越大）
        eff_num = (1.0 - torch.pow(beta_eff, class_freq.clamp(min=1.0)))
        freq_w = (1.0 - beta_eff) / eff_num
        freq_w = freq_w / (freq_w.mean() + 1e-8)

        self.register_buffer("freq_w_per_class", freq_w)  # [C]
        tail_mask = torch.zeros(num_classes, dtype=torch.bool)
        if len(tail_classes) > 0:
            tail_mask[torch.as_tensor(tail_classes, dtype=torch.long)] = True
        self.register_buffer("tail_mask", tail_mask)

        self.group_size = group_size
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.lambda_tbsr = lambda_tbsr
        self.mu_margin = mu_margin
        self.target_margin = target_margin
        self.k_diff = k_diff
        self.logit_temp = logit_temp
        self.stopgrad_weights = stopgrad_weights
        self.eps = eps

    @staticmethod
    def _ensure_int_targets(targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 2:
            return targets.argmax(dim=1)
        return targets

    def _vectorized_group_logits(self, feature: torch.Tensor, fc: nn.Module):
        """
        对特征做自适应尺度噪声，再过同一个 fc。
        feature: [B, D]，返回 group_logits: [G, B, C]
        """
        B, D = feature.shape
        # 自适应噪声尺度：按特征的列标准差（更稳定）
        col_std = feature.detach().float().std(dim=0, keepdim=True).clamp(min=1e-3)  # [1, D]
        noise = torch.randn(self.group_size, B, D, device=feature.device, dtype=feature.dtype)
        noise = noise * (self.sigma * col_std)  # [G, B, D]
        perturbed = feature.unsqueeze(0) + noise  # [G, B, D]
        # 共享 fc 计算
        G = self.group_size
        logits = fc(perturbed.reshape(G * B, D)).reshape(G, B, -1)  # [G, B, C]
        return logits

    def _logit_margins(self, group_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        在 logit 空间计算 margin：z_y - max_{k!=y} z_k，向量化实现
        group_logits: [G, B, C]; targets: [B]
        返回 distances: [G, B]
        """
        G, B, C = group_logits.shape
        z = group_logits / max(self.logit_temp, 1e-6)
        # true logits
        idx = targets.view(1, B, 1).expand(G, B, 1)
        z_true = z.gather(dim=2, index=idx).squeeze(2)  # [G, B]

        # competitor logits：mask 掉 true，再取 max
        onehot = F.one_hot(targets, num_classes=C).to(torch.bool)      # [B, C]
        mask = ~onehot                                               # [B, C]
        mask = mask.unsqueeze(0).expand(G, B, C)                     # [G, B, C]
        z_masked = z.masked_fill(~mask, float('-inf'))               # [G, B, C]
        z_others, _ = z_masked.max(dim=2)                            # [G, B]

        margins = z_true - z_others                                  # [G, B]
        return margins

    def _sample_weights(self, margins_mean: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        margins_mean: [B]
        返回样本权重，均值归一到 1.0
        """
        # 难度：sigmoid(-k * margin)，margin 小/负 -> 难度大
        difficulty = torch.sigmoid(-self.k_diff * margins_mean)  # [B]
        # 频率：Effective Number per class
        freq_w = self.freq_w_per_class[targets]                  # [B]
        # 乘法融合 + 幂指数
        w = (difficulty.clamp(min=0., max=1.) ** self.alpha) * (freq_w ** self.beta)  # [B]
        # 均值归一，稳定梯度尺度
        w = w / (w.mean().detach() + self.eps)
        return w

    def _tbsr_loss(self, margins: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        margins: [G, B]（logit margin）
        只在 tail 样本上：方差 + 均值 margin 约束
        """
        B = targets.size(0)
        is_tail = self.tail_mask[targets]  # [B], bool
        if not is_tail.any():
            return margins.new_tensor(0.0)

        marg_tail = margins[:, is_tail]                 # [G, B_tail]
        var_term = marg_tail.var(dim=0, unbiased=False) # [B_tail]
        mean_term = marg_tail.mean(dim=0)               # [B_tail]

        # 方差稳定 + 平均 margin 推离边界
        loss_var = var_term.mean()
        loss_mean_margin = F.relu(self.target_margin - mean_term).mean()

        return loss_var + self.mu_margin * loss_mean_margin

    def forward(self, pred: torch.Tensor, feature: torch.Tensor, fc: nn.Module, targets: torch.Tensor) -> torch.Tensor:
        """
        pred: [B, C]，主干前向得到的 logits（用于 CE）
        feature: [B, D]，进入 fc 前的特征
        fc: 线性分类头
        targets: [B] 或 [B, C]
        """
        y = self._ensure_int_targets(targets).to(pred.device)   # [B]
        B = y.size(0)

        # 1) 组扰动 logits（向量化）
        group_logits = self._vectorized_group_logits(feature, fc)  # [G, B, C]

        # 2) logit margin（向量化）
        margins = self._logit_margins(group_logits, y)             # [G, B]
        margins_mean = margins.mean(dim=0)                          # [B]

        # 3) 样本权重
        weights = self._sample_weights(margins_mean, y)            # [B]
        if self.stopgrad_weights:
            weights = weights.detach()

        # 4) TBSR
        tbsr = self._tbsr_loss(margins, y)

        # 5) 加权 CE（sum/sum）
        ce = F.cross_entropy(pred, y, reduction='none')            # [B]
        ce = (weights * ce).sum() / (weights.sum() + self.eps)

        total = ce + self.lambda_tbsr * tbsr

        if torch.isnan(total):
            print("[GIBALossV2] NaN! check margins_mean:", margins_mean[:8].detach().cpu())
        return total

class GIBALoss_best(nn.Module):
    def __init__(self, class_freq: List[float], tail_classes: List[int], 
                 group_size: int = 3, sigma: float = 0.1, 
                 alpha: float = 1.0, beta: float = 1.0, 
                 lambda_tbsr: float = 0.1, temp: float = 1.0):
        """
        GIBALoss class for long-tail visual recognition, inspired by GRPO.
        
        Args:
            class_freq (List[float]): Frequency of each class.
            tail_classes (List[int]): Indices of tail classes.
            group_size (int): Number of perturbed samples (G).
            sigma (float): Perturbation strength for feature noise.
            alpha (float): Weight for difficulty term (based on boundary distance mean).
            beta (float): Weight for frequency term.
            lambda_tbsr (float): Weight for TBSR regularization.
            temp (float): Temperature for softmax in boundary distance calculation.
        """
        super(GIBALoss_best, self).__init__()
        self.class_freq = torch.tensor(class_freq, dtype=torch.float32)
        self.tail_classes = tail_classes
        self.group_size = group_size
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.lambda_tbsr = lambda_tbsr
        self.temp = temp
    
    def compute_boundary_distances(self, group_pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B = targets.size(0)
        distances = torch.zeros(self.group_size, B, device=targets.device)
        for j in range(self.group_size):
            logits = group_pred[j] / self.temp  # [B, C]
            true_logits = logits[torch.arange(B), targets]
            logits_others = logits.clone()
            logits_others[torch.arange(B), targets] = float('-inf')
            max_other_logits = logits_others.max(dim=-1)[0]
            distances[j] = true_logits - max_other_logits
        return distances

    def compute_weights(self, distances: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute sample weights based on boundary distance mean and class frequency"""
        distance_mean = distances.mean(dim=0)  # [batch_size]
        difficulty = torch.exp(-distance_mean)  # [batch_size], positive and larger for smaller distances
        class_weights = 1.0 / torch.log(1 + self.class_freq).to(targets.device)
        freq_weights = class_weights[targets]  # [batch_size]
        weights = self.alpha * difficulty + self.beta * freq_weights  # [batch_size]
        weights = torch.clamp(weights, min=0.0, max=10.0)  # Clamp to reasonable range
        return weights / weights.max()  # Normalize weights

    def compute_tbsr_loss(self, distances: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """自适应版 Tail Boundary Stability Regularization"""
        device = targets.device
        is_tail = torch.tensor([1 if t.item() in self.tail_classes else 0 for t in targets], device=device).bool()
        if not is_tail.any():
            return distances.new_tensor(0.0)

        # Tail 样本的 margin
        tail_distances = distances[:, is_tail]  # [G, B_tail]
        var_term = torch.var(tail_distances, dim=0, unbiased=False).mean()

        # 均值 & 方差（当前 batch）
        mean_tail_batch = tail_distances.mean(dim=0).mean()
        std_tail_batch = tail_distances.mean(dim=0).std(unbiased=False) + 1e-8

        # 初始化 EMA buffer（只初始化一次）
        if not hasattr(self, "_ema_tail_mean"):
            self._ema_tail_mean = mean_tail_batch.detach()
            self._ema_tail_std = std_tail_batch.detach()

        # 更新 EMA（动量 0.9，可调）
        self._ema_tail_mean = 0.9 * self._ema_tail_mean + 0.1 * mean_tail_batch.detach()
        self._ema_tail_std = 0.9 * self._ema_tail_std + 0.1 * std_tail_batch.detach()

        # 自适应 target margin
        margin_floor = 0.05  # 最低要求
        target_margin = max(margin_floor, float(self._ema_tail_mean + 0.5 * self._ema_tail_std))

        # 均值项：越低于 target_margin，惩罚越大
        mean_term = F.relu(target_margin - tail_distances.mean(dim=0)).mean()

        return var_term + 0.5 * mean_term  # 0.5 为均值项系数，可调


    def forward(self, pred, feature, fc, targets):
        device = pred.device
        B = pred.size(0)
        batch_size = B
        targets = targets.argmax(1) if targets.dim() == 2 else targets

        # 归一化特征再加大尺度噪声
        normed_feat = feature / (feature.norm(dim=1, keepdim=True) + 1e-8)
        group_pred = torch.zeros(self.group_size, B, pred.size(-1), device=device)

        # 改成自适应 sigma
        with torch.no_grad():
            # 基础 margin（用 pred 不加噪声计算）
            z_true = pred[torch.arange(batch_size), targets]
            z_other = pred.clone()
            z_other[torch.arange(batch_size), targets] = float('-inf')
            z_max = z_other.max(dim=1)[0]
            base_margin = z_true - z_max  # [B]

            # 负 margin / 小 margin → 噪声更大
            diff_factor = torch.sigmoid((0.1 - base_margin) * 5)  # 0.1 是目标，5 是斜率
            tail_boost = torch.tensor([1.0 if t.item() in self.tail_classes else 0.0 for t in targets],
                                    device=device)
            tail_boost = 1.0 + 0.5 * tail_boost  # 尾类额外放大 50%

            sigma_i = self.sigma * (1.0 + diff_factor) * tail_boost
            sigma_i = sigma_i.view(-1, 1)  # [B,1] 方便广播

        # 生成组预测
        for j in range(self.group_size):
            noise = torch.randn_like(feature) * sigma_i
            perturbed_feature = feature + noise
            group_pred[j] = fc(perturbed_feature)

    
        # margin（logit差）
        distances = self.compute_boundary_distances(group_pred, targets)

        # difficulty 用 1/(1+margin)
        distance_mean = distances.mean(dim=0)
        difficulty = 1.0 / (1.0 + distance_mean.abs())
        freq_weights = (1.0 / torch.log(1 + self.class_freq.to(device)))[targets]
        weights = self.alpha * difficulty + self.beta * freq_weights

        # TBSR loss
        tbsr_loss = self.compute_tbsr_loss(distances, targets)

        class_loss = F.cross_entropy(pred, targets, reduction='none')
        weighted_class_loss = (weights.detach() * class_loss).mean()
        return weighted_class_loss + self.lambda_tbsr * tbsr_loss
    
class GIBALoss(nn.Module):
    def __init__(self, class_freq: List[float], tail_classes: List[int], 
                 group_size: int = 5, sigma: float = 0.1, 
                 alpha: float = 1.0, beta: float = 1.0, 
                 lambda_tbsr: float = 0.1, temp: float = 1.0):
        """
        GIBALoss class for long-tail visual recognition, inspired by GRPO.
        
        Args:
            class_freq (List[float]): Frequency of each class.
            tail_classes (List[int]): Indices of tail classes.
            group_size (int): Number of perturbed samples (G).
            sigma (float): Perturbation strength for feature noise.
            alpha (float): Weight for difficulty term (based on boundary distance mean).
            beta (float): Weight for frequency term.
            lambda_tbsr (float): Weight for TBSR regularization.
            temp (float): Temperature for softmax in boundary distance calculation.
        """
        super(GIBALoss, self).__init__()
        self.class_freq = torch.tensor(class_freq, dtype=torch.float32)
        self.tail_classes = tail_classes
        self.group_size = group_size
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.lambda_tbsr = lambda_tbsr
        self.temp = temp

    def _set_tail_classes(self, tail_classes: List[int]):
        self.tail_classes = tail_classes

    def compute_boundary_distances(self, group_pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B = targets.size(0)
        distances = torch.zeros(self.group_size, B, device=targets.device)
        for j in range(self.group_size):
            logits = group_pred[j] / self.temp  # [B, C]
            true_logits = logits[torch.arange(B), targets]
            logits_others = logits.clone()
            logits_others[torch.arange(B), targets] = float('-inf')
            max_other_logits = logits_others.max(dim=-1)[0]
            distances[j] = true_logits - max_other_logits
        return distances

    def compute_weights(self, distances: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute sample weights based on boundary distance mean and class frequency"""
        distance_mean = distances.mean(dim=0)  # [batch_size]
        difficulty = torch.exp(-distance_mean)  # [batch_size], positive and larger for smaller distances
        class_weights = 1.0 / torch.log(1 + self.class_freq).to(targets.device)
        freq_weights = class_weights[targets]  # [batch_size]
        weights = self.alpha * difficulty + self.beta * freq_weights  # [batch_size]
        weights = torch.clamp(weights, min=0.0, max=10.0)  # Clamp to reasonable range
        return weights / weights.max()  # Normalize weights

    def compute_tbsr_loss(self, distances: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """自适应版 Tail Boundary Stability Regularization"""
        device = targets.device
        is_tail = torch.tensor([1 if t.item() in self.tail_classes else 0 for t in targets], device=device).bool()
        if not is_tail.any():
            return distances.new_tensor(0.0)

        # Tail 样本的 margin
        tail_distances = distances[:, is_tail]  # [G, B_tail]
        var_term = torch.var(tail_distances, dim=0, unbiased=False).mean()

        # 均值 & 方差（当前 batch）
        mean_tail_batch = tail_distances.mean(dim=0).mean()
        std_tail_batch = tail_distances.mean(dim=0).std(unbiased=False) + 1e-8

        # 初始化 EMA buffer（只初始化一次）
        if not hasattr(self, "_ema_tail_mean"):
            self._ema_tail_mean = mean_tail_batch.detach()
            self._ema_tail_std = std_tail_batch.detach()

        # 更新 EMA（动量 0.9，可调）
        self._ema_tail_mean = 0.9 * self._ema_tail_mean + 0.1 * mean_tail_batch.detach()
        self._ema_tail_std = 0.9 * self._ema_tail_std + 0.1 * std_tail_batch.detach()

        # 自适应 target margin
        margin_floor = 0.05  # 最低要求
        target_margin = max(margin_floor, float(self._ema_tail_mean + 0.5 * self._ema_tail_std))

        # 均值项：越低于 target_margin，惩罚越大
        mean_term = F.relu(target_margin - tail_distances.mean(dim=0)).mean()

        return var_term + 0.5 * mean_term  # 0.5 为均值项系数，可调

    def compute_directional_adv(self, pred, targets, feature, fc, sigma_i, base_margin, K=3, rho=0.4):
        """
        方向自适应扰动（Top-K 决策边界法向 + 负边样本自适应强度）
        pred: [B, C]
        targets: [B]
        feature: [B, D] or [B, D, H, W]
        fc: 分类头 (nn.Linear)
        sigma_i: [B, 1] 自适应扰动幅度
        base_margin: [B]
        K: Top-K 竞争者数量
        rho: 全局系数
        """
        device = pred.device
        B, C = pred.size()

        # 选 Top-K 竞争者（排除真类）
        comp = pred.clone()
        comp[torch.arange(B, device=device), targets] = -float('inf')
        K = min(K, C - 1)
        topk_idx = comp.topk(K, dim=1).indices  # [B, K]

        adv = torch.zeros_like(feature)
        W = getattr(fc, "weight", None)
        if W is None or W.dim() != 2:
            return adv  # 无法取权重时，回退为零方向

        # 匹配 [C, D] / [D, C] 两种权重排布
        if W.size(0) == C:    # [C, D]
            w_true = W[targets]                  # [B, D]
            w_comp = W[topk_idx]                  # [B, K, D]
            w_comp_mean = w_comp.mean(dim=1)      # [B, D]
        elif W.size(1) == C:  # [D, C]
            w_true = W[:, targets].T                                   # [B, D]
            w_comp = W[:, topk_idx]                                    # [D, B, K]
            w_comp = w_comp.permute(1, 2, 0)                           # [B, K, D]
            w_comp_mean = w_comp.mean(dim=1)                           # [B, D]
        else:
            return adv  # 权重形状不匹配，安全回退

        # 方向向量
        nvec = w_true - w_comp_mean
        nvec = nvec / (nvec.norm(dim=1, keepdim=True) + 1e-8)

        # 负边/小边样本自适应步长
        gamma = 5.0 / (base_margin.std().clamp(min=1e-3))
        h_neg = torch.sigmoid(gamma * (-base_margin)).view(B, 1)

        adv2d = (rho * sigma_i * h_neg) * nvec  # [B, D]
        if feature.dim() == 4:
            adv = adv2d.unsqueeze(-1).unsqueeze(-1)
        else:
            adv = adv2d
        return adv


    def forward(self, pred, feature, fc, targets, epoch=None):
        device = pred.device
        B = pred.size(0)
        batch_size = B
        targets = targets.argmax(1) if targets.dim() == 2 else targets

        # 归一化特征再加大尺度噪声
        normed_feat = feature / (feature.norm(dim=1, keepdim=True) + 1e-8)
        group_pred = torch.zeros(self.group_size, B, pred.size(-1), device=device)

        # 改成自适应 sigma
        with torch.no_grad():
            # 基础 margin（用 pred 不加噪声计算）
            z_true = pred[torch.arange(batch_size), targets]
            z_other = pred.clone()
            z_other[torch.arange(batch_size), targets] = float('-inf')
            z_max = z_other.max(dim=1)[0]
            base_margin = z_true - z_max  # [B]

            # 负 margin / 小 margin → 噪声更大
            diff_factor = torch.sigmoid((0.1 - base_margin) * 5)  # 0.1 是目标，5 是斜率
            tail_boost = torch.tensor([1.0 if t.item() in self.tail_classes else 0.0 for t in targets],
                                    device=device)
            tail_boost = 1.0 + 0.5 * tail_boost  # 尾类额外放大 50%

            sigma_i = self.sigma * (1.0 + diff_factor) * tail_boost
            sigma_i = sigma_i.view(-1, 1)  # [B,1] 方便广播

        # 生成组预测
        for j in range(self.group_size):
            noise = torch.randn_like(feature) * sigma_i
            perturbed_feature = normed_feat + noise
            group_pred[j] = fc(perturbed_feature)

    
        # margin（logit差）
        distances = self.compute_boundary_distances(group_pred, targets)

        # # difficulty 用 1/(1+margin)
        # distance_mean = distances.mean(dim=0)
        # difficulty = 1.0 / (1.0 + distance_mean.abs())
        # ===[改动F2-开始]  动态温度控制 margin 分散度 =================
        if epoch is not None:
            T_start = 1.5   # 初始温度（分布收缩）
            T_end   = 0.5   # 最终温度（分布拉开）
            progress = min(1.0, epoch / 200)
            T = T_start - (T_start - T_end) * progress

            distance_mean = distances.mean(dim=0) / T  # 温度缩放
        else:
            distance_mean = distances.mean(dim=0)
        # 标准化 + Sigmoid 难度映射
        mu = distance_mean.detach().mean()
        std = distance_mean.detach().std().clamp(min=1e-3)
        z = (distance_mean - mu) / std
        k = 2.5 / std
        difficulty = torch.sigmoid(-k * z)
        # ===[改动F2-结束]================================================
        freq_weights = (1.0 / torch.log(1 + self.class_freq.to(device)))[targets]
        weights = self.alpha * difficulty + self.beta * freq_weights

        # TBSR loss
        tbsr_loss = self.compute_tbsr_loss(distances, targets)

        # ===================[改动#1-开始] Balanced Softmax 替换 CE ====================
        # 渐进先验强度 τ_bs：前期小，后期 → 1（线性；要更平滑可改余弦）
        progress = min(1.0, float(epoch) / max(1, 200))  # 0 → 1
        tau_bs = progress                      # 从 0 → 1；若想更温和用 tau_bs = 0.5* (1 - math.cos(math.pi*progress))

        # 先验：log(n_c)
        log_n = torch.log(self.class_freq.to(pred.device) + 1e-12)   # [C]

        # BSL logits（只作用于分类主损；不影响你用于 TBSR 的组扰动）
        bs_logits = pred + tau_bs * log_n

        # 用 BSL 的逐样本损失替代 CE
        class_loss = F.cross_entropy(bs_logits, targets, reduction='none')  # [B]
        # ===================[改动#1-结束]================================================

        # class_loss = F.cross_entropy(pred, targets, reduction='none')
        weighted_class_loss = (weights.detach() * class_loss).mean()
        return weighted_class_loss + self.lambda_tbsr * tbsr_loss
    
    def forward(self, pred, feature, fc, targets, epoch=None):
        device = pred.device
        B = pred.size(0)
        batch_size = B
        targets = targets.argmax(1) if targets.dim() == 2 else targets

        # 归一化特征
        normed_feat = feature / (feature.norm(dim=1, keepdim=True) + 1e-8)
        group_pred = torch.zeros(self.group_size, B, pred.size(-1), device=device)

        # --------- 自适应 sigma（R1：目标翻转概率模型） ----------
        with torch.no_grad():
            # 基础 margin（干净预测）
            z_true = pred[torch.arange(batch_size), targets]
            z_other = pred.clone()
            z_other[torch.arange(batch_size), targets] = float('-inf')
            z_max, k_star = z_other.max(dim=1)             # 最强竞争类 index
            base_margin = z_true - z_max                   # [B]

            # 计算 ||W_y - W_k*||
            C = pred.size(1)
            W = getattr(fc, "weight", None)
            if (W is not None) and (W.dim() == 2):
                if W.size(0) == C:        # [C, D]
                    w_true = W[targets]               # [B, D]
                    w_comp = W[k_star]                # [B, D]
                elif W.size(1) == C:      # [D, C]
                    w_true = W[:, targets].T          # [B, D]
                    w_comp = W[:, k_star].T           # [B, D]
                else:
                    # 兜底：无法取到权重，退化为单位范数
                    w_true = w_comp = None
                if w_true is not None:
                    g_norm = (w_true - w_comp).norm(dim=1).clamp(min=1e-6)  # [B]
                else:
                    g_norm = torch.ones(B, device=device)
            else:
                g_norm = torch.ones(B, device=device)

            # 目标翻转概率 p0（唯一可解释超参，建议 0.3；也可 0.25/0.35 试）
            p0 = 0.30
            inv_phi = torch.sqrt(torch.tensor(2.0, device=device)) * torch.erfinv(2*torch.tensor(p0, device=device)-1.0)
            inv_phi = inv_phi.abs().clamp(min=1e-6)  # |Φ^{-1}(p0)|

            # 正 margin 用解析式；负/极小 margin 用批中位作下限，避免全零扰动
            pos = base_margin > 1e-6
            sigma_i = torch.empty(B, device=device)
            sigma_i[pos] = base_margin[pos] / (g_norm[pos] * inv_phi + 1e-8)

            if pos.any():
                sigma_floor = sigma_i[pos].median()
            else:
                sigma_floor = torch.tensor(0.0, device=device)
            sigma_i[~pos] = sigma_floor

            sigma_i = sigma_i.view(-1, 1)  # [B,1]
        # ------------------------------------------------------

        # 生成组预测（其余不动）
        for j in range(self.group_size):
            noise = torch.randn_like(feature) * sigma_i
            perturbed_feature = normed_feat + noise
            group_pred[j] = fc(perturbed_feature)

        # 组内 margin
        distances = self.compute_boundary_distances(group_pred, targets)

        # --------- difficulty（R2：经验翻转概率，无温度/无k/无批标准化） ----------
        m = distances.mean(dim=0)                            # [B]
        s = distances.std(dim=0, unbiased=False).clamp(min=1e-6)
        # difficulty ≈ P(margin < 0) = Φ( -m / s )
        difficulty = 0.5 * (1.0 + torch.erf((-m / (s + 1e-8)) / 1.41421356237))
        # ------------------------------------------------------

        # 频率项与加权（保持你的写法）
        freq_weights = (1.0 / torch.log(1 + self.class_freq.to(device)))[targets]
        weights = self.alpha * difficulty + self.beta * freq_weights

        # TBSR（原实现不动）
        tbsr_loss = self.compute_tbsr_loss(distances, targets)

        # 分类主损（你现在用了 BSL，就保持不动；若用 LA，同理换成 LA logits）
        progress = 0.0 if epoch is None else min(1.0, float(epoch) / max(1, 200))
        tau_bs = progress
        log_n = torch.log(self.class_freq.to(pred.device) + 1e-12)
        bs_logits = pred + tau_bs * log_n
        class_loss = F.cross_entropy(bs_logits, targets, reduction='none')

        weighted_class_loss = (weights.detach() * class_loss).mean()
        return weighted_class_loss + self.lambda_tbsr * tbsr_loss
