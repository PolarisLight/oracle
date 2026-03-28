import os
import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.append("../../..")  # 添加上级目录到路径中

plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'figure.titlesize': 20
})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


def count_dataset_samples(root_dir, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    """统计数据集每个类的样本数"""
    class_counts = {}
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            count = sum(1 for f in os.listdir(class_path) if f.lower().endswith(exts))
            class_counts[class_name] = count
    return class_counts


if __name__ == "__main__":
    datasets = {
        "Oracle-20K": r"H:\DatasetD\oracles20k",
        
        "HUST-OBC": r"H:\DatasetD\HUST-OBC\processed",
        "OBC306": r"H:\DatasetD\OBC306_total\train",
    }

    plt.figure(figsize=(8, 6))

    for dataset_name, dataset_dir in datasets.items():
        if not dataset_dir or not os.path.exists(dataset_dir):
            print(f"跳过 {dataset_name}，路径无效：{dataset_dir}")
            continue

        counts = count_dataset_samples(dataset_dir)
        sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

        # 输出最大值和最小值
        max_class = max(sorted_counts, key=sorted_counts.get)
        min_class = min(sorted_counts, key=sorted_counts.get)
        print(f"[{dataset_name}] 最大类: {max_class} -> {sorted_counts[max_class]}")
        print(f"[{dataset_name}] 最小类: {min_class} -> {sorted_counts[min_class]}")

        # 转换为占比
        class_values = np.array(list(sorted_counts.values()), dtype=float)
        class_values = class_values / class_values.max()

        # x轴归一化到 [0,1]
        x = np.linspace(0, 1, len(class_values))

        plt.plot(x, class_values, label=dataset_name, linewidth=2)

    plt.xlabel("Sorted Class Index")
    plt.ylabel("Samples Number (Normalized)")
    plt.title("Normalized Class Distribution of OBCR Datasets")
    plt.legend()
    plt.tight_layout()
    plt.savefig("draw/icassp/draw_freq/draw_freq.png", dpi=300,
                bbox_inches='tight', pad_inches=0.1)
    # plt.show()
