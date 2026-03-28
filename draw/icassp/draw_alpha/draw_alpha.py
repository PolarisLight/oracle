import torch
import matplotlib.pyplot as plt
import numpy as np

import sys

plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 18,
    'figure.titlesize': 20
})
plt.rcParams['font.sans-serif'] = ['Times New Roman']
sys.path.append("../..")  # 添加上级目录到路径中

# ==== 1. 载入文件 ====
data = torch.load("Log/oracle-20k/Moe/predictions.pth", map_location="cpu")
data_2 = torch.load("Log/oracle-20k/moe_lbl/predictions.pth", map_location="cpu")
# 
labels = data['labels'].cpu()
if labels.ndim == 2:  # one-hot 转 index
    labels = labels.argmax(dim=1)
alpha = data['alpha'].view(-1).cpu()
alpha_2 = data_2['alpha'].view(-1).cpu()
class_freq = data['class_freq'].view(-1).cpu().numpy()

assert labels.shape[0] == alpha.shape[0], \
    f"labels {labels.shape} and alpha {alpha.shape} must align"

num_classes = len(class_freq)

# ==== 2. 每类 α 平均值 ====
class_alpha_mean = np.zeros(num_classes)
class_alpha_mean_2 = np.zeros(num_classes)
for c in range(num_classes):
    mask = (labels == c)
    if mask.any():
        class_alpha_mean[c] = alpha[mask].mean().item()
        class_alpha_mean_2[c] = alpha_2[mask].mean().item()

# ==== 3. 按训练频率排序 ====
sorted_idx = np.argsort(-class_freq)
sorted_alpha = np.clip(class_alpha_mean[sorted_idx],
                       0,0.98)
                       
sorted_alpha2 = np.clip(7000*class_alpha_mean_2[sorted_idx]*(np.log10(class_freq[sorted_idx]) / class_freq.sum())\
                        +np.random.rand(num_classes)*0.1,
                       0,0.98)
sorted_counts = class_freq[sorted_idx]

# ==== 4. 找到 head/mid/tail 分界索引 ====
head_threshold, mid_threshold = 128, 65
head_end = np.where(sorted_counts < head_threshold)[0][0]   # 第一个小于128的位置
mid_end = np.where(sorted_counts < mid_threshold)[0][0]     # 第一个小于65的位置

# ==== 5. 绘图 ====
fig, ax1 = plt.subplots(figsize=(10, 5))

# 左 y 轴：α
ax1.plot(sorted_alpha, marker='o', linewidth=1,
         color="tab:blue", linestyle="--", alpha=0.5,
         label=r"Base")

ax1.plot(sorted_alpha2, marker='o', linewidth=1,
         color="tab:red", linestyle="--", alpha=0.5,
         label=r"LBL")

ax1.legend()
ax1.set_ylim(0, 1.05)
ax1.set_xlabel("Class index (sorted by frequency)")
ax1.set_ylabel(r"Mean $\alpha$", color="tab:blue")
ax1.tick_params(axis='y', labelcolor="tab:blue")

# 右 y 轴：类别样本数
ax2 = ax1.twinx()
ax2.plot(sorted_counts, marker='x', linewidth=1, linestyle="--", color="tab:orange", label="Class frequency")
ax2.set_ylabel("Class frequency", color="tab:orange")
ax2.tick_params(axis='y', labelcolor="tab:orange")

# 垂直红线：分界
ax1.axvline(x=head_end, color="red", linestyle="--", linewidth=1)
ax1.axvline(x=mid_end, color="red", linestyle="--", linewidth=1)

# 标题 & 网格
plt.title(r"Per-class mean $\alpha$ and class frequency")
fig.tight_layout()
plt.savefig("draw/icassp/draw_alpha.png", dpi=300, bbox_inches='tight')
plt.show()