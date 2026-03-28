from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from datasets.CIFAR_LT import IMBALANCECIFAR100

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型（ResNet-18）
model = models.resnet34(pretrained=False, num_classes=100).to(device)

# 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

from config.config_cifar_moe import Arguments
args = Arguments()
args.dataset = Arguments()
args.dataset.data_dir = 'H:\DatasetD\cifar'
args.dataset.imb_factor = 0.01

# 数据加载
train_loader = torch.utils.data.DataLoader(IMBALANCECIFAR100(args, train=True), batch_size=128, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(IMBALANCECIFAR100(args, train=False), batch_size=128, shuffle=False, num_workers=0)

# 🟢 **Step 3: 在训练过程中监视 Logits**
logits_tracking = {i: [] for i in range(100)}  # 存储每个类别的 logits 变化情况

@torch.no_grad()
def track_logits(model, dataloader):
    """
    记录所有类别的 Logits 分布
    """
    model.eval()
    temp_logits = {i: [] for i in range(100)}

    for data in dataloader:
        images, labels = data['image'].to(device), data['label'].to(device)
        # 🟢 **转换 One-Hot Labels 为整数类别索引**
        labels = torch.argmax(labels, dim=1)  # 从 (batch_size, num_classes) → (batch_size,)

        logits = model(images)  # 获取 logits（不经过 softmax）

        labels_list = labels.tolist()  # 确保转换成 Python 列表

        for i, label in enumerate(labels_list):
            logit_values = logits[i].detach().cpu().numpy()  # 确保转换到 NumPy
            logit_mean = float(np.mean(logit_values))  # 确保转换为 float
            temp_logits[label].append(logit_mean)  # 记录平均 logits 值


    # 计算每个类别的 logits 均值
    avg_logits = {cls: np.mean(temp_logits[cls], axis=0) if len(temp_logits[cls]) > 0 else 0 for cls in temp_logits}
    var_logits = {cls: np.var(temp_logits[cls], axis=0) if len(temp_logits[cls]) > 0 else 0 for cls in temp_logits}

    return avg_logits, var_logits

# 训练循环
with tqdm.tqdm(total=200) as pbar:
    for epoch in range(200):
        model.train()
        total_loss = 0
        for data in train_loader:
            images, labels = data['image'].to(device), data['label'].to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 记录 logits 变化
        avg_logits, var_logits = track_logits(model, train_loader)
        for cls in range(100):
            logits_tracking[cls].append((np.mean(avg_logits[cls]), np.mean(var_logits[cls])))

        pbar.set_description(f"Epoch [{epoch+1}/{200}], Loss: {total_loss:.4f}")
        pbar.update(1)
        scheduler.step()

# 🟢 **Step 5: 可视化 Logits 分布**
cls_indices = np.arange(100)
head_cls = cls_indices[:100 // 3]  # 头部类别
tail_cls = cls_indices[-100 // 3:]  # 尾部类别

# 计算最终的 logits 平均值和方差
final_avg_logits = [np.mean([l[0] for l in logits_tracking[cls]]) for cls in cls_indices]
final_var_logits = [np.mean([l[1] for l in logits_tracking[cls]]) for cls in cls_indices]

plt.figure(figsize=(12, 6))
plt.plot(cls_indices, final_avg_logits, label="Avg Logits", marker="o")
plt.plot(cls_indices, final_var_logits, label="Logits Variance", marker="o")
plt.axvline(x=head_cls[-1], linestyle="--", color="r", label="Head-Tail Boundary")
plt.xlabel("Class Index (Sorted by Frequency)")
plt.ylabel("Logit Value")
plt.title("Logits Distribution in Long-Tailed Classification")
plt.legend()
plt.show()