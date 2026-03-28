import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torch
import seaborn as sns
import sys
sys.path.append("../..")  # 添加上级目录到路径中

base_dir = "Log/oracle-20k"
save_path = "draw/tsne"

plt.rcParams['font.sans-serif'] = ['Times New Roman']


methods = ['CE','AWM']

def draw_tsne(data,method="CE"):
    features = data['features']
    labels = data['labels']
    num_class = labels.shape[1]
    labels = labels.argmax(dim=1)  # [6935]
    # 等间距选取十个类
    selected_classes = range(0,num_class)
    # features = torch.mean(features, dim=1)  # 如果需要对特征进行平均，可以取消注释
    features = features[:,0,:]
    features = features.cpu().numpy()
    labels = labels.cpu().numpy()
    selected_samples = np.isin(labels, selected_classes)
    selected_features = features[selected_samples]
    selected_labels = labels[selected_samples]
    if method == 'AWM':
        perplexity = 30
    else:
        perplexity = 30  # 设置 t-SNE 的 perplexity 参数
    tsne = TSNE(n_components=2, random_state=42,perplexity=perplexity)
    
    feature_tsne = tsne.fit_transform(selected_features)

    # 绘制 t-SNE 图
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("bright", n_colors=len(selected_classes))
    # scatter = plt.scatter(feature_tsne[:, 0], feature_tsne[:, 1], c=selected_labels, cmap='tab10', alpha=0.7)
    sns.scatterplot(
        x=feature_tsne[:, 0],
        y=feature_tsne[:, 1],
        hue=selected_labels,
        # style=selected_labels,
        legend=False,
        palette=palette,
        ax=ax
        # c=['blue' if label == pred else 'red' for label, pred in zip(labels, preds)]
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # 显示图形
    plt.tight_layout()
    plt.savefig(f'{save_path}/tsne_{method}-1.png', dpi=500, bbox_inches='tight', pad_inches=0.1)
    # plt.show()


for method in methods:
    
    data = torch.load(f'{base_dir}\{method}\predictions.pth')
    draw_tsne(data,method)