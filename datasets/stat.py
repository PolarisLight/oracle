import os
import matplotlib.pyplot as plt
import sys

sys.path.append("..")  # 添加上级目录到路径中

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20
})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


def count_dataset_samples(root_dir, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    """
    统计数据集每个类的样本数
    :param root_dir: 数据集根目录，每个子文件夹对应一个类
    :param exts: 可识别的图像后缀
    :return: {类别名: 样本数}
    """
    class_counts = {}
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            count = sum(1 for f in os.listdir(class_path) if f.lower().endswith(exts))
            class_counts[class_name] = count
    return class_counts


if __name__ == "__main__":
    # 这里维护你的数据集字典
    datasets = {
        "Oracle-20K": "H:\DatasetD\oracles20k",
        "OBC306": "H:\DatasetD\OBC306_total\\train",
        "HUST-OBC": "H:\DatasetD\HUST-OBC\processed",   # 空置
    }
    color_palette = ['#004343','#008585','#74a892',]
    for i, (dataset_name, dataset_dir) in enumerate(datasets.items()):
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

        # 画直方图（横坐标用数字索引）
        class_values = list(sorted_counts.values())
        x = list(range(len(class_values)))  # 0 ~ class_num-1

        plt.figure(figsize=(8, 6))
        plt.bar(x, class_values,color=color_palette[i])

        # 设置x轴刻度：只显示5个（包含首尾）
        num_ticks = 5
        tick_positions = [int(i) for i in 
                          list(range(0, len(x), max(1, len(x)//(num_ticks-1))))]
        if tick_positions[-1] != len(x)-1:
            tick_positions.append(len(x)-1)

        plt.xticks(tick_positions, [str(i) for i in tick_positions])

        plt.xlabel("Class Index")
        plt.ylabel("Sample Number")
        plt.title(f"Class Distribution of {dataset_name} Dataset")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"draw/icassp/{dataset_name}_class_distribution.png")