import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

x = np.linspace(-6, 6, 400)
relu = np.maximum(0, x)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
softmax_x = np.linspace(-6, 6, 400)
sm_y1 = [softmax([i, 0, 0])[0] for i in softmax_x]

fig, axes = plt.subplots(2,2, figsize=(10,8))

titles = [
    ("ReLU",       r"$y = \max(0,\, x)$"),
    ("Sigmoid",    r"$y = \frac{1}{1 + e^{-x}}$"),
    ("Tanh",       r"$y = \tanh(x)$"),
    ("Softmax",    r"$y_i = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$")
]

# 1. ReLU
axes[0,0].plot(x, relu, color='#e41a1c', lw=2)
axes[0,0].set_xlim([-6,6]); axes[0,0].set_ylim([-1,6.2])
axes[0,0].grid(alpha=0.3)
axes[0,0].axvline(0,ls=":",c="#222")
axes[0,0].axhline(0,ls=":",c="#222")

# 2. Sigmoid
axes[0,1].plot(x, sigmoid, color='#377eb8', lw=2)
axes[0,1].set_xlim([-6,6]); axes[0,1].set_ylim([-0.1,1.1])
axes[0,1].grid(alpha=0.3)
axes[0,1].axvline(0,ls=":",c="#222")
axes[0,1].axhline(0,ls=":",c="#222")

# 3. Tanh
axes[1,0].plot(x, tanh, color='#4daf4a', lw=2)
axes[1,0].set_xlim([-6,6]); axes[1,0].set_ylim([-1.1,1.1])
axes[1,0].grid(alpha=0.3)
axes[1,0].axvline(0,ls=":",c="#222")
axes[1,0].axhline(0,ls=":",c="#222")

# 4. Softmax（一维第1分量）
axes[1,1].plot(softmax_x, sm_y1, color='#984ea3', lw=2)
axes[1,1].set_xlim([-6,6]); axes[1,1].set_ylim([0,1.1])
axes[1,1].grid(alpha=0.3)
axes[1,1].axvline(0,ls=":",c="#222")
axes[1,1].axhline(0,ls=":",c="#222")

# 标题（左对齐函数名，右对齐latex表达式）
for idx, ax in enumerate(axes.flat):
    name, formula = titles[idx]
    # 函数名左侧，公式右侧
    ax.text(0.00, 1.06, name, fontsize=15, ha='left', va='bottom', transform=ax.transAxes, fontweight='bold')
    ax.text(1.00, 1.06, formula, fontsize=14, ha='right', va='bottom', transform=ax.transAxes, fontname='serif')
    # Softmax补充英文non-latex说明，不在latex 公式字符串内
    if name == "Softmax":
        ax.text(1.00, 0.98, "(showing $y_1$ for $[x_1,0,0]$)", fontsize=11, ha='right', va='top', transform=ax.transAxes, color='#888')

for ax in axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)

fig.tight_layout()
plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
plt.show()
