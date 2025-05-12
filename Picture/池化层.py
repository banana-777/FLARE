import numpy as np
import matplotlib.pyplot as plt

# 防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def draw_matrix(ax, data, title="", highlight=None, cmap="Blues"):
    im = ax.imshow(data, cmap=cmap, vmin=np.min(data), vmax=np.max(data))
    # 颜色自适应
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if np.abs(data[i, j]) >= (np.max(data) + np.min(data))/2 else "black"
            ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center", color=color, fontsize=13, fontweight='bold')
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_title(title, fontsize=16, pad=15)
    ax.tick_params(length=0)
    # 高亮
    if highlight is not None:
        x, y, w, h = highlight
        rect = plt.Rectangle((y-0.5, x-0.5), h, w, edgecolor='red', facecolor='none', lw=2)
        ax.add_patch(rect)
    return im

fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'width_ratios':[1,0.45,1]})

# ===== 输入特征图 =====
input_map = np.array([
    [0.8, -0.3, 0.3, 0.7, 0.5],
    [0.7,  0.3, 0.7, 0.6, 0.5],
    [-0.4,-0.5,-0.6, 0.7, 0.5],
    [-0.1,0.0, -0.3, 0.0, 0.4],
    [0.3, 0.3,-0.3,-0.7, 0.1]
])

draw_matrix(axes[0], input_map, "输入特征图", highlight=[0,0,2,2])

# ===== 池化操作说明（中间） =====
axes[1].axis('off')
# 箭头
axes[1].annotate('', xy=(0.88,0.55), xytext=(0.12,0.55),
                 arrowprops=dict(arrowstyle="simple", fc='black', ec='black',lw=2))
# 操作说明分上下两行，排版更清楚
axes[1].text(0.5, 0.65, "池化操作", fontsize=14, ha='center', va='center')
axes[1].text(0.5, 0.5, "2×2最大池化", fontsize=13, ha='center', va='center')
axes[1].plot([0.12,0.88], [0.37,0.37], lw=2, color="black") # 短线
axes[1].set_xlim(0,1)
axes[1].set_ylim(0,1)

# ===== 输出特征图 =====
# 按2x2最大池化（stride=2）计算3x3输出
output_map = np.array([
    [np.max(input_map[0:2,0:2]),    np.max(input_map[0:2,2:4]),    np.max(input_map[0:2,4:5])],
    [np.max(input_map[2:4,0:2]),    np.max(input_map[2:4,2:4]),    np.max(input_map[2:4,4:5])],
    [np.max(input_map[4:5,0:2]),    np.max(input_map[4:5,2:4]),    np.max(input_map[4:5,4:5])]
])
draw_matrix(axes[2], output_map, "输出特征图", cmap="Reds")

plt.subplots_adjust(wspace=0.22)
plt.show()
