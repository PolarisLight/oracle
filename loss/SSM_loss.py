import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaSSMLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.1, label_dis=None):
        """
        SSM-inspired loss with fixed state transition matrix.
        :param label_dis: list of class sample numbers, used to build frequency-based A matrix.
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha

        # 权重初始化为全1
        self.register_buffer("weights", torch.ones(num_classes))

        # 构建状态转移矩阵 A 和输入矩阵 B
        if label_dis is not None:
            freq = torch.tensor(label_dis, dtype=torch.float32)
            freq = freq / freq.sum()
            A = torch.diag(freq)  # 类别频率对角矩阵
        else:
            A = torch.eye(num_classes)
        A = (1 - alpha) * A + alpha * torch.ones_like(A) / num_classes  # 平滑处理
        B = torch.eye(num_classes) *alpha

        self.register_buffer("A", A)
        self.register_buffer("B", B)

    def compute_avg_loss_per_class(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = F.cross_entropy(logits, targets, reduction='none')  # [B]
        total_loss = torch.zeros(self.num_classes, device=logits.device)
        count = torch.zeros(self.num_classes, device=logits.device)
        total_loss = total_loss.scatter_add(0, targets, losses)
        count = count.scatter_add(0, targets, torch.ones_like(losses))
        avg_loss = total_loss / (count + 1e-6)
        return avg_loss

    def forward(self, logits, targets):
        if targets.dim() > 1:
            targets = torch.argmax(targets, dim=1)
        else:
            targets = targets.squeeze()

        ce = F.cross_entropy(logits, targets, reduction='none')  # [B]

        # --- Step 1: 计算各类 avg loss
        avg_loss_per_class = self.compute_avg_loss_per_class(logits, targets)

        # --- Step 2: 权重演化更新 w[t] = A @ w[t-1] + B @ avg_loss[t]
        with torch.no_grad():
            evolved_weights = self.A @ self.weights + self.B @ avg_loss_per_class.detach()
            self.weights.copy_(evolved_weights.clamp(min=0.1, max=10.0))  # 防止过大或为0

        # --- Step 3: 对每个样本应用对应类的权重
        # 归一化处理，防止权重爆炸或梯度失衡
        normalized_weights = (self.weights - self.weights.min()) / (self.weights.max() - self.weights.min() + 1e-6)
        sample_weights = 0.1 + 0.9 * normalized_weights[targets]  # 保持权重 > 0

        # 计算加权损失
        weighted_loss = ce * sample_weights
        return weighted_loss.mean()
    

class RebalancedCELoss(nn.Module):
    """
    使用类别频率进行加权的交叉熵损失（Rebalanced Method）
    w_c ∝ 1 / freq_c
    """
    def __init__(self, label_distribution: torch.Tensor, reduction: str = 'mean'):
        """
        Args:
            label_distribution: Tensor of shape [num_classes] 表示每个类别的样本频数或比例
        """
        super().__init__()
        self.reduction = reduction

        # 反频率作为权重（避免除零）
        weights = 1.0 / (label_distribution + 1e-6)
        weights = weights / weights.sum() * len(weights)  # 归一化，使均值为1
        self.register_buffer('class_weights', weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C], targets: [B]
        """
        loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction=self.reduction)
        return loss
    

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    Args:
        gamma: focusing parameter γ > 0 (default=2)
        alpha: class weight (default=None); can be scalar or tensor [C]
        reduction: mean, sum, or none
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is not None:
            if isinstance(alpha, (list, torch.Tensor)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = torch.tensor([alpha], dtype=torch.float32)
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape [B, C]
            targets: LongTensor of shape [B] (class indices)
        """
        if targets.dim() > 1:
            targets = torch.argmax(targets, dim=1)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [B]
        probs = F.softmax(logits, dim=1)  # [B, C]
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # p_t for true class

        focal_factor = (1 - pt) ** self.gamma  # [B]
        loss = focal_factor * ce_loss  # [B]

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # [B]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # [B]