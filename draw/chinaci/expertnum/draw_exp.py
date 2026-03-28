import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'figure.titlesize': 20
})
plt.rcParams['font.sans-serif'] = ['Times New Roman']

data_x = [1, 2, 3, 4, 5, 6, 7]
data_y = [91.01,92.14,93.11,93.14,93.15,93.03,92.99]

# draw the line and scatter plot
plt.figure(figsize=(12, 6))
plt.plot(data_x, data_y, marker='o', linestyle='-', color='#F58231', linewidth=2, markersize=8)
plt.xlabel('Expert Number')
plt.ylabel('ACC')
plt.title('Expert Number vs ACC')
plt.xticks(np.arange(1, 8, 1))
# plt.xlim(0, 1.0)
plt.grid(True)
base_dir = "draw/expertnum"
save_path = "draw/expertnum"
plt.savefig(f"{save_path}/expertnum_vs_ACC.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()