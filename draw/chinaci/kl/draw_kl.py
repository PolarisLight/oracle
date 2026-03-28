import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.append("../..")  # 添加上级目录到路径中

base_path = "Log/oracle-20k"
save_path = "draw/kl"


def compute_kl_divergence(logits, targets):
    # 对logits进行softmax，获得每个类的概率分布
    probs = F.softmax(logits, dim=2)
    if targets.ndim == 2:  # one-hot
        targets = targets.argmax(dim=1)  # [6935]
    # 初始化KL散度矩阵
    num_classes = probs.shape[2]
    num_experts = probs.shape[1]
    kl_divergence = torch.zeros(num_classes, num_experts, num_experts).cuda()

    # 根据标签过滤每个类的样本
    for class_idx in range(num_classes):
        # 获取当前类的样本
        class_mask = (targets == class_idx)
        class_probs = probs[class_mask, :, class_idx]  # 当前类的专家概率分布

        # 对每对专家之间计算KL散度
        for i in range(3):
            for j in range(i+1, 3):
                P = class_probs[:, i]  # 专家i的概率分布
                Q = class_probs[:, j]  # 专家j的概率分布
                # 计算KL散度
                kl = F.kl_div(P.log(), Q, reduction='batchmean', log_target=False)
                kl_divergence[class_idx, i, j] = kl
                kl_divergence[class_idx, j, i] = kl  # KL散度是对称的

    # 计算每个类的平均KL散度（每对专家之间）
    avg_kl_per_class = kl_divergence.sum(dim=(1, 2))  # 对所有专家之间的KL散度求平均

    return avg_kl_per_class.cpu().numpy()

plt.rcParams['font.sans-serif'] = ['Times New Roman']
# set text size
plt.rcParams.update({'font.size': 16})
# 载入数据
type_c = 'IACL'
plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'figure.titlesize': 20
})


data_path = f'{base_path}/DCL/predictions.pth'

data_pal = torch.load(data_path)

data_path = f'{base_path}/CE/predictions.pth'
data_ce = torch.load(data_path)

# 提取logits和targets
logits_pal = data_pal['predictions']  # shape: [bs, 3, 100]
targets_pal = torch.tensor(data_pal['labels']).cuda()  # shape: [bs]

logits_ce = data_ce['predictions']  # shape: [bs, 3, 100]
targets_ce = torch.tensor(data_ce['labels']).cuda()  # shape: [bs]


# 初始化一个存储每个专家在每个类上的准确率的矩阵
num_classes = logits_pal.shape[2]  # 类的数量
num_experts = logits_pal.shape[1]  # 专家的数量
accuracy_per_class = torch.zeros(num_classes, num_experts).cuda()  # [100, 3] 每个类每个专家的准确率

avg_kl_per_class_pal = compute_kl_divergence(logits_pal, targets_pal)
avg_kl_per_class_ce = compute_kl_divergence(logits_ce, targets_ce)

avg_kl_per_class_pal = np.abs(-np.log(avg_kl_per_class_pal + 1e-6))
avg_kl_per_class_ce = np.abs(-np.log(avg_kl_per_class_ce + 1e-6))
import pandas as pd
def fill_missing(arr):
    arr = pd.Series(arr)
    arr = arr.replace([np.inf, -np.inf], np.nan)  # 去掉 inf
    arr = arr.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    return arr.values
avg_kl_per_class_pal = fill_missing(avg_kl_per_class_pal)
avg_kl_per_class_ce = fill_missing(avg_kl_per_class_ce)
#设置图像大小
fig, ax1 = plt.subplots(figsize=(8, 6))

# 绘制 KL 散度（每个类的平均 KL 散度）
ax1.stackplot(
    np.arange(num_classes),
    avg_kl_per_class_pal,
    avg_kl_per_class_ce,
    labels=['AWM', 'CE'],
    alpha=0.5
)

# 添加标题、标签
ax1.set_xlabel('Class Index', fontdict={'size': 24})
ax1.set_ylabel('log(KL Divergence)', fontdict={'size': 24})

# 设置 x 轴刻度：显示 10 个标签
ax1.set_xticks(np.arange(0, num_classes, num_classes // 10))
# 自动缩放 y 轴
# ax1.set_ylim(0, 5)

# 显示图例
ax1.legend(loc='upper center', ncol=2)

# 显示图形
plt.tight_layout()
plt.savefig(f'{save_path}/KL_divergence.png', dpi=500, bbox_inches='tight', pad_inches=0.1)
# plt.show()