import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import sys
from datasets.OBC306 import OBC306
from config.config_cifar_oracle import config
from config.config_cifar_base import config as base_config
from argparse import Namespace, ArgumentParser
from utils.utils import set_seed
import tqdm

class WDClassifier(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, gamma=0.5):
        super().__init__(in_features, out_features, bias)
        self.gamma = gamma

    def forward(self, x):
        # 权重范数缩放
        weight = self.weight
        norm = weight.norm(p=2, dim=1, keepdim=True)  # [C,1]
        scaled_weight = weight / (norm.pow(self.gamma) + 1e-6)
        return F.linear(x, scaled_weight, self.bias)

# -----------------------
# 1. Backbone + Classifier 定义
# -----------------------
class ResNet34(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        backbone = torchvision.models.resnet34(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # 去掉 fc
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.fc_rt = nn.Linear(backbone.fc.in_features, num_classes)
        # self.fc = WDClassifier(backbone.fc.in_features, num_classes, bias=True, gamma=0.5)
        # self.fc_rt = WDClassifier(backbone.fc.in_features, num_classes, bias=True, gamma=0.5)

    def forward(self, x,rt=False):
        feat = self.backbone(x).flatten(1)
        if not rt:
            logits = self.fc(feat)
        else:
            logits = self.fc_rt(feat)
        return logits, feat


class ResNet32(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        # ResNet32 for CIFAR
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.fc_rt = nn.Linear(64, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = []
        layers.append(self._make_basic_block(self.inplanes, planes, stride))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(self._make_basic_block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def _make_basic_block(self, inplanes, planes, stride=1):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )
    def forward(self, x,rt=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        if not rt:
            logits = self.fc(feat)
        else:
            logits = self.fc_rt(feat)
        return logits, feat

class ResNet50(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        backbone = torchvision.models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # 去掉 fc
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.fc_rt = nn.Linear(backbone.fc.in_features, num_classes)
        # self.fc = WDClassifier(backbone.fc.in_features, num_classes, bias=True, gamma=0.5)
        # self.fc_rt = WDClassifier(backbone.fc.in_features, num_classes, bias=True, gamma=0.5)

    def forward(self, x,rt=False):
        feat = self.backbone(x).flatten(1)
        if not rt:
            logits = self.fc(feat)
        else:
            logits = self.fc_rt(feat)
        return logits, feat

# -----------------------
# 2. 训练工具函数
# -----------------------
def train_epoch(model, dataloader, criterion, optimizer, device,epoch=0):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data in tqdm.tqdm(dataloader):
        imgs, labels = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        logits, _ = model(imgs)
        loss = criterion(logits, labels,epoch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels.argmax(1)).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total
class_freq = []
from metrics.metrics import accuracy_score,compute_hmt_acc
@torch.no_grad()
def evaluate(model, dataloader, criterion, device,hmt=None):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds = []
    all_labels = []
    for data in dataloader:
        imgs, labels = data['image'].to(device), data['label'].to(device)
        logits, _ = model(imgs)
        loss = criterion(logits, labels)
        all_preds.append(logits)
        all_labels.append(labels)

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels.argmax(1)).sum().item()
        total += labels.size(0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    if hmt is not None:
        hmt_acc = hmt(all_preds, all_labels)
        print(f"H/M/T Acc: {hmt_acc}")
    return total_loss / total, correct / total


# -----------------------
# 3. 生成均衡采样子集
# -----------------------
def make_balanced_subset(dataset, num_samples_per_class=100):
    targets = np.array(dataset.targets)
    indices = []
    for c in np.unique(targets):
        cls_idx = np.where(targets == c)[0]
        chosen = np.random.choice(cls_idx, size=min(num_samples_per_class, len(cls_idx)), replace=False)
        indices.extend(chosen)
    return Subset(dataset, indices)

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

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C] 模型输出
            targets: [B] 或 one-hot
        """
        device = logits.device
        log_prior = self.log_prior.to(device)

        # 调整 logit
        adj_logits = logits + log_prior

        # 交叉熵
        if targets.ndim == 2:  # one-hot
            targets = targets.argmax(dim=1)
        loss = F.cross_entropy(adj_logits, targets, reduction=self.reduction)
        return loss
    
class ProgressLogitAdjustmentLoss(nn.Module):
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

    def forward(self, logits, targets, epoch):
        """
        Args:
            logits: [B, C] 模型输出
            targets: [B] 或 one-hot
        """
        device = logits.device
        log_prior = self.log_prior.to(device)

        # 调整 logit
        adj_logits = logits + log_prior * (0.5 + epoch / 200)

        # 交叉熵
        if targets.ndim == 2:  # one-hot
            targets = targets.argmax(dim=1)
        loss = F.cross_entropy(adj_logits, targets, reduction=self.reduction)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Focal Loss for classification
        Args:
            alpha (float or list): 类别平衡因子。如果是 float，则对所有类别相同；如果是 list/ndarray，则针对每个类别。
            gamma (float): 难样本聚焦参数，gamma=2 最常用
            reduction (str): 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, (list, tuple, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        else:
            self.alpha = alpha

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C] 模型输出 (未经过 softmax)
            targets: [B] 或 one-hot
        """
        if targets.ndim == 2:  # one-hot → index
            targets = targets.argmax(dim=1)

        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = softmax(logits)[y]

        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha.to(logits.device)[targets]
        else:
            alpha_t = self.alpha

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ClassBalancedLoss(nn.Module):
    def __init__(self, class_freq, beta=0.9999, gamma=0.0, reduction='mean'):
        """
        Class-Balanced Loss (可与 CE/Focal 结合)
        Args:
            class_freq: list[int] 每个类别的样本数
            beta: 衰减系数，接近1时更强调少数类
            gamma: focal loss 的 gamma (如果=0就是普通CB-CE)
            reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        effective_num = 1.0 - np.power(beta, class_freq)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * len(class_freq)  # normalize
        self.weights = torch.tensor(weights, dtype=torch.float)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C]
            targets: [B] 或 one-hot
        """
        if targets.ndim == 2:  # one-hot → index
            targets = targets.argmax(dim=1)

        ce_loss = F.cross_entropy(logits, targets, weight=self.weights.to(logits.device), reduction='none')
        pt = torch.exp(-ce_loss)  # p_t
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        if target.ndim == 2:  # one-hot
            target = target.argmax(dim=1)
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

class KPSLoss(nn.Module):
    r"""Implement of KPS Loss :
    Args:
    """

    def __init__(self, cls_num_list, max_m=0.5, weighted=False, weight= None, s=30):
        super(KPSLoss, self).__init__()
        assert s > 0

        s_list = torch.cuda.FloatTensor(cls_num_list)
        s_list = s_list*(50/s_list.min())
        s_list = torch.log(s_list) #torch.log(s_list) #s_list**(1/4) #torch.log(s_list) #s_list**(1/4)#s_list = torch.log(s_list)**2  #s_list**(1/5)
        s_list = s_list*(1/s_list.min()) #s+ s_list #
        self.s_list = s_list
        self.s = s
        
        m_list =  torch.flip(self.s_list, dims=[0])
        m_list = m_list * (max_m / m_list.max())
        self.m_list = m_list
                
        self.weighted = weighted
        self.weight = weight
        

    def forward(self, input, label):
        if label.ndim == 2:  # one-hot
            label = label.argmax(dim=1)
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = input*self.s_list
        phi = cosine - self.m_list
        # --------------------------- convert label to one-hot ---------------------------
        index = torch.zeros_like(input, dtype=torch.uint8)
        index.scatter_(1, label.data.view(-1, 1), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        #output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output = torch.where(index, phi, cosine)
        
        if self.weighted == False:
            output *= self.s
        elif self.weighted == True:
            index_float = index.type(torch.cuda.FloatTensor)
            batch_s = torch.flip(self.s_list, dims=[0])*self.s
            batch_s = torch.clamp(batch_s, self.s, 50)    #s过大不好。          
            batch_s = torch.matmul(batch_s[None, :], index_float.transpose(0,1)) 
            batch_s = batch_s.view((-1, 1))           
            output *= batch_s
        else:
            output *= self.s
        return F.cross_entropy(output, label, weight= self.weight)


# -----------------------
# 4. 主流程 (Decoupling)
# -----------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    opt = ArgumentParser()
    opt.add_argument('--task', type=str, default='oracle-20k', help='task name')
    opt.add_argument('--model', type=str, default='ResNet_MoE', help='model name')
    opt.add_argument('--dataset', type=str, default='OBC306_CutMix', help='dataset name')
    opt.add_argument('--seed', type=int, default=123, help='random seed')
    opt.add_argument('--save_log', type=bool, default=0, help='save log')
    opt = opt.parse_args()
    set_seed(opt.seed)
    args = config(opt.task, opt.model, opt.dataset, opt.save_log)
    args.base_args = base_config(opt.task, opt.model, opt.dataset, opt.save_log)
    # args.train.use_wandb = False  # Disable Weights & Biases logging

    train_set = OBC306(args=args, train=True)
    test_set = OBC306(args=args, train=False)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

    # ---- 第一阶段：训练 Backbone + Classifier ----
    model = ResNet50(num_classes=args.model.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.train.lr, weight_decay=1e-4)
    from torch.optim import lr_scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 
                                                        T_max=180, 
                                                        eta_min=0, last_epoch=-1)
    from metrics.metrics import HMTAccuracy
    hmt = HMTAccuracy(class_freq=train_set.get_cls_num_list())
    from loss.moe_loss import soft_entropy
    # criterion = LogitAdjustmentLoss(class_freq=train_set.get_cls_num_list(),)
    # criterion = FocalLoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = ClassBalancedLoss(class_freq=train_set.get_cls_num_list())
    # criterion = LDAMLoss(cls_num_list=train_set.get_cls_num_list())
    # criterion = KPSLoss(cls_num_list=train_set.get_cls_num_list(), max_m=0.5, weighted=True, s=30)
    criterion = ProgressLogitAdjustmentLoss(class_freq=train_set.get_cls_num_list(),)
    eval_criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(200):  # 可调整
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, device,epoch)
        scheduler.step()
        if (epoch+1) % 1 == 0:
            val_loss, val_acc = evaluate(model, test_loader, eval_criterion, device,hmt)
            if val_acc > best_acc:
                best_acc = val_acc
            print(f"[Stage1][Epoch {epoch+1}] Train Acc: {acc:.4f} | Test Acc: {val_acc:.4f} | Best Test Acc: {best_acc:.4f}")

    # ---- 第二阶段：冻结 Backbone，只训练 Classifier ----
    for param in model.backbone.parameters():
        param.requires_grad = False

    balanced_train_set = make_balanced_subset(train_set, num_samples_per_class=200)
    balanced_loader = DataLoader(balanced_train_set, batch_size=128, shuffle=True, num_workers=0)

    optimizer = optim.AdamW(model.fc_rt.parameters(), lr=args.train.lr, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0, last_epoch=-1)

    for epoch in range(0):  # 短一点即可
        loss, acc = train_epoch(model, balanced_loader, criterion, optimizer, device)
        scheduler.step()
        if (epoch+1) % 5 == 0:
            val_loss, val_acc = evaluate(model, test_loader, eval_criterion, device)
            if val_acc > best_acc:
                best_acc = val_acc
            print(f"[Stage2][Epoch {epoch+1}] Train Acc: {acc:.4f} | Test Acc: {val_acc:.4f} | Best Test Acc: {best_acc:.4f}")

    print(f"[Stage2] Best Test Acc: {best_acc:.4f}")
    # save model
    torch.save(model.state_dict(), f"{args.train.save_dir}/best_model.pth")

if __name__ == "__main__":
    main()
