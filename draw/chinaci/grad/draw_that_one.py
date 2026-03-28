import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import sys
sys.path.append("../..")  # 添加上级目录到路径中

base_path = "Log/oracle-20k"
save_path = "draw/grad"

# 读取 CSV 文件路径（请根据实际路径修改）
ce_grad_path = "ce_grad.csv"
ce_loss_path = "ce_exp_loss.csv"
col_grad_path = "col_grad.csv"
col_loss_path = "col_exp_loss.csv"

# 如果文件在 base_path 目录下，请使用以下代码
ce_grad_path = f"{base_path}/CE/val_grad.csv"
ce_loss_path = f"{base_path}/CE/val_exp_loss.csv"
col_grad_path = f"{base_path}/DCL/val_grad.csv"
col_loss_path = f"{base_path}/DCL/val_exp_loss.csv"

def smooth_nan(df: pd.DataFrame):
    """
    检查并修复 DataFrame 中的 NaN 值。
    采用线性插值方法（前后插值），如果在边界仍有 NaN，则用最近的非 NaN 值填充。

    Args:
        df (pd.DataFrame): 输入的 DataFrame，行=epoch，列=experts

    Returns:
        pd.DataFrame: 修复后的 DataFrame
    """
    # 线性插值
    df_fixed = df.interpolate(method='linear', limit_direction='both', axis=0)
    # 再次检查是否还存在 NaN，如果有就用 0 或者均值填补（保险）
    if df_fixed.isna().sum().sum() > 0:
        df_fixed = df_fixed.fillna(method='bfill').fillna(method='ffill')
    return df_fixed


# 读取前180个epoch的数据，跳过第一列epoch索引
ce_grad = pd.read_csv(ce_grad_path).iloc[:, :]
ce_loss = pd.read_csv(ce_loss_path).iloc[:, :]
col_grad = pd.read_csv(col_grad_path).iloc[:, :]
col_loss = pd.read_csv(col_loss_path).iloc[:, :]

# ==== 新增：修复 NaN ====
ce_grad = smooth_nan(ce_grad)
ce_loss = smooth_nan(ce_loss)
col_grad = smooth_nan(col_grad)
col_loss = smooth_nan(col_loss)
# 计算每个专家的梯度占比与loss占比
ce_grad_share = ce_grad.div(ce_grad.sum(axis=1), axis=0)
ce_loss_share = ce_loss.div(ce_loss.sum(axis=1), axis=0)
col_grad_share = col_grad.div(col_grad.sum(axis=1), axis=0)
col_loss_share = col_loss.div(col_loss.sum(axis=1), axis=0)


plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 20,
    'figure.titlesize': 20
})
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# 可视化
def plot_correlation(loss_data, grad_data, title):
    plt.figure(figsize=(8, 6))
    for i in range(3):
        x = loss_data.iloc[:, i]
        y = grad_data.iloc[:, i]
        slope, intercept, r, p, _ = linregress(x, y)
        plt.scatter(x, y, label=f"Expert {i+1} (r={r:.2f})", alpha=0.6)
        plt.plot(np.sort(x), slope * np.sort(x) + intercept, linestyle="--")
    plt.title(f"{title}: Loss Share vs Gradient Share")
    plt.xlabel("Loss Share",fontdict={'size':24})
    plt.ylabel("Gradient Share",fontdict={'size':24})
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}/{title}_correlation.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    #plt.show()

# 分别绘图
plot_correlation(ce_loss_share, ce_grad_share, "CE")
plot_correlation(col_loss_share, col_grad_share, "DCL")