import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    # 'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 20
})
plt.rcParams['font.sans-serif'] = ['Times New Roman']

# 数据
image_sizes = [32, 64, 128, 224, 256]
baseline_acc = [88.53, 90.62, 91.85, 93.27, 93.39]
awm_acc = [93.11, 94.36, 95.78, 96.65, 96.82]

# 画图
plt.figure(figsize=(6,4))
plt.plot(image_sizes, baseline_acc, marker='o', linestyle='-', label='Baseline', linewidth=2)
plt.plot(image_sizes, awm_acc, marker='s', linestyle='--', label='AWM', linewidth=2)

# 坐标轴和标题
plt.xlabel("Image Size")
plt.ylabel("Top-1 Accuracy (%)")
plt.title("Effect of Image Size on Recognition Accuracy")

# 网格和图例
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# 显示
plt.tight_layout()
# plt.show()
plt.savefig("draw/further/imagesize.png", dpi=300, bbox_inches='tight', pad_inches=0.1)