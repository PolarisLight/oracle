import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegativeMining_Proto(nn.Module):
    def __init__(self, num_classes, feature_dim, momentum=0.9, temperature=0.07, k=3, device='cuda'):

        super(HardNegativeMining_Proto, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.temperature = temperature
        self.k = k
        self.device = device

        self.prototypes = nn.Parameter(
            torch.zeros(num_classes, feature_dim, device=device), requires_grad=False
        )

        self.register_buffer("confusion_matrix", torch.zeros(num_classes, num_classes, device=device))

    def update_prototypes(self, features, labels):

        if labels.dim() == 2 and labels.size(1) > 1:
            labels = torch.argmax(labels, dim=1)

        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = labels == label
            feature_mean = features[mask].mean(dim=0).detach()
            self.prototypes.data[label] = (
                self.momentum * self.prototypes.data[label] +
                (1 - self.momentum) * feature_mean
            )

    def update_confusion_matrix(self, predicted, labels):

        if labels.dim() == 2 and labels.size(1) > 1:
            labels = torch.argmax(labels, dim=1)

        predicted = predicted.long()
        labels = labels.long()
        indices = torch.stack([labels, predicted], dim=0)
        updates = torch.ones_like(labels, dtype=torch.float)
        self.confusion_matrix.index_put_(tuple(indices), updates, accumulate=True)

    def apply_epoch_momentum(self, momentum=0.9):

        self.confusion_matrix *= 1 - momentum

    def get_hard_negative_classes(self, labels):

        hard_negatives = []
        for label in labels:
            class_confusion = self.confusion_matrix[label] 
            _, hard_classes = torch.topk(class_confusion, k=self.k, largest=True)
            hard_negatives.append(hard_classes)
        return torch.stack(hard_negatives)

    def compute_contrastive_loss(self, features, labels):

        if labels.dim() == 2 and labels.size(1) > 1:
            labels = torch.argmax(labels, dim=1)

        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)

        pos_sim = F.cosine_similarity(features, prototypes[labels], dim=1)

        hard_negative_classes = self.get_hard_negative_classes(labels)  # (batch_size, k)
        hard_negative_prototypes = prototypes[hard_negative_classes]  # (batch_size, k, feature_dim)

        neg_sim = torch.matmul(
            features.unsqueeze(1),  # (batch_size, 1, feature_dim)
            hard_negative_prototypes.transpose(1, 2)  # (batch_size, feature_dim, k)
        ).squeeze(1) / self.temperature  # (batch_size, k)

        weighted_neg_sim = torch.exp(neg_sim).mean(dim=1)

        loss = -torch.log(torch.exp(pos_sim / self.temperature) / (torch.exp(pos_sim / self.temperature) + weighted_neg_sim))
        return loss.mean()

    def reset_confusion_matrix(self):

        self.confusion_matrix = torch.zeros_like(self.confusion_matrix)


class HardNegativeMining_Proto_Enhanced(nn.Module):
    def __init__(self, num_classes, feature_dim, momentum=0.9, temperature=0.07, k=3, device='cuda'):
        super(HardNegativeMining_Proto_Enhanced, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.temperature = temperature
        self.k = k
        self.device = device

        self.prototypes = nn.Parameter(
            torch.zeros(num_classes, feature_dim, device=device), requires_grad=False
        )
        self.register_buffer("confusion_matrix", torch.zeros(num_classes, num_classes, device=device))

    # ✅ 原型更新
    def update_prototypes(self, features, labels):
        if labels.dim() == 2 and labels.size(1) > 1:
            labels = torch.argmax(labels, dim=1)
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = labels == label
            feature_mean = features[mask].mean(dim=0).detach()
            self.prototypes.data[label] = (
                self.momentum * self.prototypes.data[label] +
                (1 - self.momentum) * feature_mean
            )

    # ✅ 混淆矩阵更新
    def update_confusion_matrix(self, predicted, labels):
        if labels.dim() == 2 and labels.size(1) > 1:
            labels = torch.argmax(labels, dim=1)
        if predicted.dim() == 2 and predicted.size(1) > 1:
            predicted = torch.argmax(predicted, dim=1)
        predicted = predicted.long()
        labels = labels.long()
        indices = torch.stack([labels, predicted], dim=0)
        updates = torch.ones_like(labels, dtype=torch.float)
        self.confusion_matrix.index_put_(tuple(indices), updates, accumulate=True)

    def apply_epoch_momentum(self, momentum=0.9):
        self.confusion_matrix *= 1 - momentum

    def reset_confusion_matrix(self):
        self.confusion_matrix = torch.zeros_like(self.confusion_matrix)

    # ✅ 获取 hard negative 类别索引
    def get_hard_negative_classes(self, labels):
        hard_negatives = []
        for label in labels:
            class_confusion = self.confusion_matrix[label]
            _, hard_classes = torch.topk(class_confusion, k=self.k, largest=True)
            hard_negatives.append(hard_classes)
        return torch.stack(hard_negatives)

    # ✅ 原有的 hard negative 对比学习损失
    def compute_contrastive_loss(self, features, labels):
        if labels.dim() == 2:
            labels = torch.argmax(labels, dim=1)
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)

        pos_sim = F.cosine_similarity(features, prototypes[labels], dim=1)

        hard_negative_classes = self.get_hard_negative_classes(labels)
        hard_negative_prototypes = prototypes[hard_negative_classes]

        neg_sim = torch.matmul(
            features.unsqueeze(1),
            hard_negative_prototypes.transpose(1, 2)
        ).squeeze(1) / self.temperature

        weighted_neg_sim = torch.exp(neg_sim).mean(dim=1)

        loss = -torch.log(torch.exp(pos_sim / self.temperature) /
                          (torch.exp(pos_sim / self.temperature) + weighted_neg_sim))
        return loss.mean()

    # ✅ [新增] 原型对比学习损失（Prototype Contrastive）
    def compute_proto_contrastive_loss(self, features, labels):
        if labels.dim() == 2:
            labels = torch.argmax(labels, dim=1)
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)

        logits = torch.matmul(features, prototypes.T) / self.temperature
        loss = F.cross_entropy(logits, labels)
        return loss

    # ✅ [新增] logits 修正（基于原型距离）
    def correct_logits(self, logits, features, labels, lambda_coef=0.1):
        if labels.dim() == 2:
            labels = torch.argmax(labels, dim=1)
        for i in range(features.size(0)):
            proto = self.prototypes[labels[i]]
            distance = torch.norm(features[i] - proto.detach(), p=2)
            logits[i] -= lambda_coef * distance
        return logits

    # ✅ [新增] 原型插值特征增强
    def prototype_interpolate(self, features, labels, alpha=0.5):
        if labels.dim() == 2:
            labels = torch.argmax(labels, dim=1)
        new_features = features.clone()
        for i in range(features.size(0)):
            proto = self.prototypes[labels[i]]
            new_features[i] = alpha * features[i] + (1 - alpha) * proto.detach()
        return new_features
    
    def compute_dynamic_proto_contrastive_loss(self, features, labels, epoch, warmup=5):
        """
        前 warmup epoch 使用所有类，之后只用 hardest negatives
        """
        if epoch < warmup:
            return self.compute_proto_contrastive_loss(features, labels)  # 全类对比
        else:
            return self.compute_contrastive_loss(features, labels)  # HMM-based