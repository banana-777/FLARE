import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

x = np.linspace(-6, 6, 400)
relu = np.maximum(0, x)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)

def softmax(x_vec):
    e_x = np.exp(x_vec - np.max(x_vec))
    return e_x / e_x.sum(axis=-1, keepdims=True)

softmax_x1 = np.linspace(-6, 6, 400)
y1_vals, y2_vals, y3_vals = [], [], []
for x1 in softmax_x1:
    y_sm = softmax([x1, 0, 0])
    y1_vals.append(y_sm[0])
    y2_vals.append(y_sm[1])
    y3_vals.append(y_sm[2])

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

# 4. Softmax (3类)
axes[1,1].plot(softmax_x1, y1_vals, color='#984ea3', lw=2, label='$y_1$')
axes[1,1].plot(softmax_x1, y2_vals, color='#ff7f00', lw=2, label='$y_2$')
axes[1,1].plot(softmax_x1, y3_vals, color='#377eb8', lw=2, label='$y_3$')
axes[1,1].set_xlim([-6,6]); axes[1,1].set_ylim([0,1.1])
axes[1,1].grid(alpha=0.3)
axes[1,1].axvline(0,ls=":",c="#222")
axes[1,1].axhline(0,ls=":",c="#222")
axes[1,1].legend(loc='center left', fontsize=12, bbox_to_anchor=(1., 0.5), title=None, frameon=False)
axes[1,1].text(1.00, 0.98, "with $[x_1,0,0]$", fontsize=11, ha='right', va='top', transform=axes[1,1].transAxes, color='#888')

for idx, ax in enumerate(axes.flat):
    name, formula = titles[idx]
    ax.text(0.00, 1.06, name, fontsize=15, ha='left', va='bottom', transform=ax.transAxes, fontweight='bold')
    ax.text(1.00, 1.06, formula, fontsize=14, ha='right', va='bottom', transform=ax.transAxes, fontname='serif')

for ax in axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)

fig.tight_layout()
plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
plt.show()
