import numpy as np
import matplotlib.pyplot as plt
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)

# 训练轮次
epochs = np.arange(1, 21)

# 不同恶意节点数量的曲线
num_malicious_nodes = [0, 1, 2, 3, 4]
markers = ['o', 's', '^', 'D', '*']


# 定义增长率先大后小的函数形式
def logistic_growth(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))


# 创建左右子图（共享y轴）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# ------------------ 左图：恶意节点=0时曲线较高，其他曲线纠缠密集 ------------------
for num_malicious in num_malicious_nodes:
    if num_malicious == 0:
        a = 85  # 恶意节点=0时最高点为85%
    else:
        a = 70 - (num_malicious * 2)  # 其他恶意节点的最高点为70%递减，差距更小

    # 为每条曲线添加随机偏移，使其更纠缠
    random_offset = random.uniform(-3, 3)
    base_accuracy = logistic_growth(epochs, a + random_offset, 0.3, 10)

    # 增大扰动范围，增加曲线交叉
    perturbation = [random.uniform(-1.5, 1.5) for _ in range(len(epochs))]
    accuracy = base_accuracy + np.array(perturbation)

    ax1.plot(epochs, accuracy, label=f'恶意节点数量: {num_malicious}',
             marker=markers[num_malicious], linewidth=2, alpha=0.8)

ax1.set_xlabel("训练轮次", fontsize=12)
ax1.set_ylabel("训练准确率", fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_xticks([])
ax1.set_yticks([])

# ------------------ 右图：曲线分离但间隔减小 ------------------
for num_malicious in num_malicious_nodes:
    if num_malicious < 4:
        a = 92 - (num_malicious * 2)  # 最高点从92%开始递减，差距更小
    else:
        a = 60  # 恶意节点=4时降至60%，与良性节点差距减小

    base_accuracy = logistic_growth(epochs, a, 0.45, 10)  # 适中增长速率
    perturbation = [random.uniform(-1, 1) for _ in range(len(epochs))]
    accuracy = base_accuracy + np.array(perturbation)

    ax2.plot(epochs, accuracy, label=f'恶意节点数量: {num_malicious}',
             marker=markers[num_malicious], linewidth=2, alpha=0.8)

ax2.set_xlabel("训练轮次", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.set_xticks([])
ax2.set_yticks([])

# 确保y轴范围为0-100
ax1.set_ylim(0, 100)

# 共享图例
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=12)

plt.tight_layout(pad=4.0)  # 增加子图间距
plt.show()