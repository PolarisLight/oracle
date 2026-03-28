import torch
import torch.nn as nn


class SSMLogitAdjustment(nn.Module):
    def __init__(self, num_classes):
        """
        使用 SSM 进行 logits 调整
        :param num_classes: 类别数量，LTVR任务中需要处理的类别数
        """
        super(SSMLogitAdjustment, self).__init__()

        # 设置类别数量和初始损失权重
        self.num_classes = num_classes
        self.A = nn.Parameter(torch.zeros(num_classes, num_classes))  # 状态转移矩阵 A
        self.B = nn.Parameter(torch.randn(num_classes, 1))  # 控制矩阵 B

        # 存储每个epoch结束时的accuracies
        self.accuracies = torch.zeros(num_classes).cuda()  # 初始化为0

    def update_accuracies(self, accuracies):
        """
        更新准确度（每个 epoch 结束时调用）
        :param accuracies: 当前类别的准确度
        """
        self.accuracies = accuracies.cuda()  # 存储当前准确度

    
    def forward(self, logits, target_labels=None, ssm_weight=1.0):
        """
        使用 SSM 学习到的矩阵调整 logits
        :param logits: 模型输出的 logit (shape: [batch_size, num_classes])
        :param target_labels: 真实标签，用于计算损失和计算准确度（在训练时）
        :return: 调整后的 logits
        """
        batch_size = logits.size(0)

        # 如果是测试阶段，没有target_labels，使用模型预测的准确度来更新
        if target_labels is not None:
            if target_labels.dim() == 1:
                target_labels = target_labels.unsqueeze(1)
            if target_labels.size(1) > 1:
                target_labels = torch.argmax(target_labels, dim=1)
            # 训练阶段，通过 target_labels 计算每个类别的准确度
            batch_accuracies = self.accuracies[target_labels]  # 从 accuracies 中获取每个样本对应类别的准确度
        else:
            # 测试阶段，使用模型预测结果来动态计算准确度
            predicted_labels = torch.argmax(logits, dim=1)
            batch_accuracies = torch.zeros_like(self.accuracies[predicted_labels])  # 从 accuracies 中获取每个样本对应类别的准确度

        # 使用 SSM 进行 logit 调整
        # 1. \( A \) 和 logits 的矩阵乘法
        A_logits = torch.matmul(logits, self.A)  # [batch_size, num_classes] @ [num_classes, num_classes] => [batch_size, num_classes]

        # 2. \( B \) 和 accuracies 的调整：每个样本的准确度应该与 B 矩阵相乘
        B_accuracies = torch.matmul(batch_accuracies.unsqueeze(1), self.B.t())

        # 3. 最终调整 logits
        adjusted_logits = logits + (A_logits + B_accuracies) * ssm_weight  # 调整后的 logits

        return adjusted_logits
    
import torch
import torch.nn as nn
import torch.nn.functional as F

# class MambaLoss(nn.Module):
#     """
#     类似于 Mamba 状态演化的 loss 设计：
#     每个类别的权重 w_c 随训练动态调整：
#         w_c[t] = A * w_c[t-1] + B * avg_loss_c[t]
#     当前版本使用固定 A=1-alpha，B=alpha，表示指数平滑
#     """
#     def __init__(self, num_classes, alpha=0.1):
#         super().__init__()
#         self.num_classes = num_classes
#         self.alpha = alpha  # 控制更新速率
#         self.register_buffer("weights", torch.ones(num_classes))  # 初始权重为1

#     def compute_avg_loss_per_class(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         """
#         向量化计算每类的平均 loss
#         logits: [B, C], targets: [B]
#         return: [C] 各类平均 loss
#         """
#         losses = F.cross_entropy(logits, targets, reduction='none')  # [B]
#         total_loss = torch.zeros(self.num_classes, device=logits.device)
#         count = torch.zeros(self.num_classes, device=logits.device)
#         total_loss = total_loss.scatter_add(0, targets, losses)
#         count = count.scatter_add(0, targets, torch.ones_like(losses))
#         avg_loss = total_loss / (count + 1e-6)
#         avg_loss = torch.clamp(avg_loss, min=0.0, max=10.0)
#         if torch.isnan(avg_loss).any() or torch.isinf(avg_loss).any():
#             print("⚠️ avg_loss_per_class 出现异常！")
#             avg_loss = torch.nan_to_num(avg_loss, nan=1.0, posinf=1.0, neginf=1.0)
#         return avg_loss

#     def forward(self, logits, targets):
#         if targets.dim() == 1:
#             targets = targets.unsqueeze(1)
#         if targets.size(1) > 1:
#             targets = torch.argmax(targets, dim=1)
#         # Step 1: 样本级 CrossEntropy
#         ce = F.cross_entropy(logits, targets, reduction='none')  # [B]

#         # Step 2: 类别级平均 loss（向量化）
#         avg_loss_per_class = self.compute_avg_loss_per_class(logits, targets)

#         # Step 3: 权重更新（指数滑动平均）
#         A, B = 1 - self.alpha, self.alpha

#         mean_loss = avg_loss_per_class.mean()
#         loss_delta = avg_loss_per_class - mean_loss
#         self.weights = A * self.weights + B * loss_delta.detach()

#         # Step 4: 每个样本应用其类别对应的权重
#         sample_weights = self.weights[targets]  # [B]
#         weighted_loss = ce * sample_weights
#         return weighted_loss.mean()

import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaLoss(nn.Module):
    """
    改进版 Mamba 状态演化式 Loss：
    动态类别权重：w_c[t] = A * w_c[t-1] + B * (avg_loss_c[t] - mean_loss)
    - 加入归一化和限制，防止发散或负值
    """
    def __init__(self, num_classes, alpha=0.1, clamp_min=0.5, clamp_max=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # 更新速率
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # 初始化权重为 1（均衡）
        self.register_buffer("weights", torch.ones(num_classes))

    def compute_avg_loss_per_class(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        向量化计算每类的平均 loss（训练 batch）
        """
        losses = F.cross_entropy(logits, targets, reduction='none')  # [B]
        total_loss = torch.zeros(self.num_classes, device=logits.device)
        count = torch.zeros(self.num_classes, device=logits.device)

        total_loss = total_loss.scatter_add(0, targets, losses)
        count = count.scatter_add(0, targets, torch.ones_like(losses))
        avg_loss = total_loss / (count + 1e-6)

        avg_loss = torch.clamp(avg_loss, min=0.0, max=10.0)  # 避免异常值
        if torch.isnan(avg_loss).any() or torch.isinf(avg_loss).any():
            print("⚠️ avg_loss_per_class 出现异常！")
            avg_loss = torch.nan_to_num(avg_loss, nan=1.0, posinf=1.0, neginf=1.0)
        return avg_loss

    def forward(self, logits, targets):
        # Ensure target shape is [B]
        if targets.dim() > 1:
            targets = torch.argmax(targets, dim=1)

        # Step 1: 样本级 cross entropy
        ce = F.cross_entropy(logits, targets, reduction='none')  # [B]

        # Step 2: 类别级平均 loss
        avg_loss_per_class = self.compute_avg_loss_per_class(logits, targets)  # [C]

        # Step 3: 权重更新（delta 形式，中心化）
        A, B = 1 - self.alpha, self.alpha
        mean_loss = avg_loss_per_class.mean()
        loss_delta = avg_loss_per_class - mean_loss
        new_weights = A * self.weights + B * loss_delta.detach()

        # Step 4: Normalize（中心化并进行 clamp 限幅）
        new_weights = torch.clamp(new_weights, min=self.clamp_min, max=self.clamp_max)
        new_weights = new_weights / new_weights.mean().detach()  # 保持均值为 1，防止 loss scale 被破坏
        self.weights = new_weights.detach()  # 不传播梯度

        # Step 5: 按样本权重计算损失
        sample_weights = self.weights[targets]  # [B]
        weighted_loss = ce * sample_weights
        return weighted_loss.mean()


import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaSSMLoss(nn.Module):
    """
    基于 SSM（状态空间模型）的类别权重演化 Loss：
    - A 矩阵按类别频率定义（尾类记忆更多）
    - B 矩阵固定为单位矩阵（新 loss 直接加入状态）
    - w[t+1] = A @ w[t] + B @ avg_loss[t] = A @ w[t] + avg_loss[t]
    """
    def __init__(self, num_classes, alpha=0.1, label_dis=None):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.register_buffer("weights", torch.ones(num_classes))
        label_dis = torch.tensor(label_dis) if label_dis is not None else None
        # 构造 A（按类别频率）和 B（单位矩阵）
        if label_dis is not None:
            self._init_A_from_freq(label_dis)
        else:
            # 默认 A = (1 - alpha) * I（所有类均等），B = I
            I = torch.eye(num_classes)
            self.register_buffer("A", (1 - alpha) * I)

        self.register_buffer("B", torch.eye(num_classes))  # 固定为单位矩阵

    def _init_A_from_freq(self, label_dis: torch.Tensor):
        freq = label_dis.float() / label_dis.sum()      # 归一化频率
        inv_freq = 1.0 / (freq + 1e-6)                   # 倒数
        inv_freq = inv_freq / inv_freq.max()            # 归一化
        A_diag = (1 - self.alpha) * inv_freq            # 尾类保留历史多
        A = torch.diag(A_diag)
        self.register_buffer("A", A)

    def compute_avg_loss_per_class(self, logits, targets):
        """
        向量化计算每类平均损失
        """
        losses = F.cross_entropy(logits, targets, reduction='none')  # [B]
        total_loss = torch.zeros(self.num_classes, device=logits.device)
        count = torch.zeros(self.num_classes, device=logits.device)
        total_loss = total_loss.scatter_add(0, targets, losses)
        count = count.scatter_add(0, targets, torch.ones_like(losses))
        avg_loss = total_loss / (count + 1e-6)
        return avg_loss.clamp(min=0.0, max=10.0)

    def forward(self, logits, targets):
        if targets.dim() > 1:
            targets = torch.argmax(targets, dim=1)

        ce = F.cross_entropy(logits, targets, reduction='none')  # [B]

        # 更新权重：w[t+1] = A @ w[t] + B @ avg_loss[t]
        avg_loss = self.compute_avg_loss_per_class(logits, targets).detach()
        self.weights = self.A @ self.weights + self.B @ avg_loss  # B=I 时可直接加

        # 应用权重到每个样本
        sample_weights = self.weights[targets]  # [B]
        weighted_loss = ce * sample_weights
        return weighted_loss.mean()
