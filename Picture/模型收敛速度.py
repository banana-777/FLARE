import matplotlib.pyplot as plt
import numpy as np

# 解决中文显示问题
plt.rcParams['font.family'] = 'SimHei'  # 指定黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 模拟数据（20轮训练）
epochs = range(1, 21)
fedavg_acc = [65.0, 70.5, 75.2, 79.0, 82.3, 84.9, 87.1, 88.8, 90.2, 91.5,
             92.6, 93.5, 94.2, 94.8, 95.3, 95.7, 96.0, 96.3, 96.5, 96.7]
opt5_acc = [70.0, 76.8, 81.5, 85.0, 87.8, 89.9, 91.7, 93.2, 94.5, 95.6,
            96.4, 97.0, 97.5, 97.8, 98.1, 98.3, 98.5, 98.7, 98.8, 98.9]
opt10_acc = [68.0, 73.5, 78.0, 82.0, 85.5, 88.0, 90.2, 92.0, 93.5, 94.8,
             95.8, 96.6, 97.2, 97.7, 98.0, 98.3, 98.5, 98.7, 98.8, 98.9]

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制三种模型的收敛曲线
plt.plot(epochs, fedavg_acc, 'b-', linewidth=2, marker='o', label='FedAvg（基准组）')
plt.plot(epochs, opt5_acc, 'g--', linewidth=2, marker='s', label='5客户端优化组')
plt.plot(epochs, opt10_acc, 'r-.', linewidth=2, marker='^', label='10客户端优化组')

# 添加图表元素
plt.title('联邦学习模型收敛速度对比（20轮训练）', fontsize=16)
plt.xlabel('训练轮次（Epoch）', fontsize=12)
plt.ylabel('测试集准确率（%）', fontsize=12)
plt.xticks(epochs[::2])  # 每2轮显示一次刻度
plt.grid(True, linestyle='--', alpha=0.7, axis='both')
plt.legend(fontsize=10, loc='lower right')
plt.ylim(60, 100)
plt.xlim(0, 21)

# 保存图片到本地
plt.savefig('模型收敛速度对比_20轮_修复字体.png', dpi=300, bbox_inches='tight')
plt.show()