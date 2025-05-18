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
curves = []

# 定义增长率先大后小的函数形式
def logistic_growth(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

# 生成数据并添加扰动
for num_malicious in num_malicious_nodes:
    a = 90 if num_malicious < 4 else 70
    base_accuracy = logistic_growth(epochs, a, 0.5, 10)
    perturbation = [random.uniform(-2, 2) for _ in range(len(epochs))]
    accuracy_with_perturbation = base_accuracy + np.array(perturbation)
    curves.append(accuracy_with_perturbation)

# 绘制曲线
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^', 'D', '*']
for i, num_malicious in enumerate(num_malicious_nodes):
    plt.plot(epochs, curves[i], label=f'恶意节点数量: {num_malicious}',
             marker=markers[i], linewidth=2, alpha=0.8)

# 添加标题和坐标轴标签（保留文字描述，去掉数字）
plt.title("不同恶意节点数量下联邦学习训练准确率曲线", fontsize=16)
plt.xlabel("训练轮次", fontsize=14)
plt.ylabel("训练准确率", fontsize=14)

# 移除坐标轴数字刻度
plt.xticks([])
plt.yticks([])

# 添加图例和网格线
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()