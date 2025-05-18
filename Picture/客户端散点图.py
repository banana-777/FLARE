import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（解决乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

np.random.seed(42)

# 生成8个良性节点（绿色），分布在原点附近
良性节点 = np.random.normal(loc=0, scale=1, size=(8, 2))
# 生成2个恶意节点（红色），分布在不同方向且距离较远
恶意节点 = np.array([[6, 6], [ -6, -6]])  # 分别位于第一和第三象限，距离拉大

# 绘制散点图
plt.figure(figsize=(8, 6))
# 良性节点（绿色圆点）
plt.scatter(良性节点[:, 0], 良性节点[:, 1],
            c='green', label='良性节点', alpha=0.8, s=150, marker='o', edgecolor='black')
# 恶意节点（红色叉号，增大标记尺寸）
plt.scatter(恶意节点[:, 0], 恶意节点[:, 1],
            c='red', label='恶意节点', alpha=0.9, s=200, marker='x', linewidth=2)

# 添加标题和标签
# plt.title("Krum算法抗攻击示意图（10客户端）", fontsize=16)
# plt.xlabel("参数空间维度", fontsize=12)
# plt.ylabel("参数空间维度2", fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.3, axis='both')

# 调整坐标轴范围
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.gca().set_aspect('equal', adjustable='box')  # 等比例显示坐标轴
plt.show()