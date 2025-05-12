import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({'font.size': 14, 'font.family': 'SimHei'})

def show_number(ax, data, cmap, vmin, vmax, fmt="{:.1f}", force_last_col_black=False):
    norm = plt.Normalize(vmin, vmax)
    for (i, j), val in np.ndenumerate(data):
        color_value = norm(val)
        # 卷积核最后一列黑色字体
        if force_last_col_black and j == data.shape[1] - 1:
            text_color = "black"
        else:
            text_color = "white" if color_value < 0.6 else "black"
        ax.text(j, i, fmt.format(val), ha='center', va='center', color=text_color, fontsize=12, weight='bold')

input_matrix = np.array([
    [0.8, -0.3,  0.3,  0.7,  0.5],
    [0.7,  0.3,  0.7,  0.6,  0.5],
    [-0.4, -0.5, -0.6,  0.7,  0.5],
    [-0.1,  0.0, -0.3,  0.3,  0.4],
    [0.3,   0.3, 0.3, -0.3, -0.7]
])
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# 计算卷积（valid模式）
feature_map = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        feature_map[i, j] = np.sum(input_matrix[i:i+3, j:j+3] * kernel)

fig, axs = plt.subplots(1, 3, figsize=(13, 4), gridspec_kw={'width_ratios': [1, 0.75, 1]})

# 1. 输入矩阵
im0 = axs[0].imshow(input_matrix, cmap='Blues', vmin=-1, vmax=1)
axs[0].set_title("输入特征图", pad=10)
show_number(axs[0], input_matrix, 'Blues', -1, 1)
rect = Rectangle((-0.5, -0.5), 3, 3, linewidth=2, edgecolor='red', facecolor='none')
axs[0].add_patch(rect)
axs[0].set_xticks(range(input_matrix.shape[1]))
axs[0].set_yticks(range(input_matrix.shape[0]))

# 2. 卷积核
im1 = axs[1].imshow(kernel, cmap='Greens', vmin=-1, vmax=1)
axs[1].set_title("卷积核", pad=10)
show_number(axs[1], kernel, 'Greens', -1, 1, fmt="{:d}", force_last_col_black=True)
axs[1].set_xticks(range(kernel.shape[1]))
axs[1].set_yticks(range(kernel.shape[0]))

# 3. 输出特征图
max_val = np.max(np.abs(feature_map))
im2 = axs[2].imshow(feature_map, cmap='Reds', vmin=-max_val, vmax=max_val)
axs[2].set_title("输出特征图", pad=10)
show_number(axs[2], feature_map, 'Reds', -max_val, max_val)
axs[2].set_xticks(range(feature_map.shape[1]))
axs[2].set_yticks(range(feature_map.shape[0]))

# 精简边框
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 箭头设置
arrowprops = dict(arrowstyle="->", lw=2, color='black')
# 箭头1：输入特征图 -> 卷积核
axs[0].annotate(
    "",
    xy=(1.45, 0.5), xycoords='axes fraction',
    xytext=(1.05, 0.5), textcoords='axes fraction',
    arrowprops=arrowprops,
)
axs[0].text(1.25, 0.54, "卷积操作", ha='center', va='bottom', fontsize=15, color='black', transform=axs[0].transAxes)

# 箭头2：卷积核 -> 输出特征图
axs[1].annotate(
    "",
    xy=(1.45, 0.5), xycoords='axes fraction',
    xytext=(1.05, 0.5), textcoords='axes fraction',
    arrowprops=arrowprops,
)
axs[1].text(1.25, 0.54, "结果", ha="center", va="bottom", fontsize=15, color="black", transform=axs[1].transAxes)

plt.tight_layout()
plt.subplots_adjust(wspace=0.45)
plt.savefig('conv_workflow_zh_arrow_clear.png', dpi=300, bbox_inches='tight')
plt.show()
