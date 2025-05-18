import matplotlib.pyplot as plt
import numpy as np

# 数据准备
epochs = range(1, 21)
fedavg_size = [1228.8]*20
opt5_size = [122.9]*20
opt10_size = [122.9]*20

# 计算累计传输量
fedavg_total = np.cumsum(fedavg_size)
opt5_total = np.cumsum(opt5_size)
opt10_total = np.cumsum(opt10_size)

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制单次传输量
plt.plot(epochs, fedavg_size, 'b-', linewidth=2, marker='o', label='FedAvg（单次）')
plt.plot(epochs, opt5_size, 'g--', linewidth=2, marker='s', label='5客户端优化组（单次）')
plt.plot(epochs, opt10_size, 'r-.', linewidth=2, marker='^', label='10客户端优化组（单次）')

# 绘制累计传输量（右轴）
ax2 = plt.twinx()
ax2.plot(epochs, fedavg_total/1024, 'b-', linewidth=2, alpha=0.6)
ax2.plot(epochs, opt5_total/1024, 'g--', linewidth=2, alpha=0.6)
ax2.plot(epochs, opt10_total/1024, 'r-.', linewidth=2, alpha=0.6)

# 图表装饰
plt.title('联邦学习参数传输量对比（20轮训练）', fontsize=16)
plt.xlabel('训练轮次（Epoch）', fontsize=12)
plt.ylabel('单次传输量（KB）', fontsize=12)
ax2.set_ylabel('累计传输量（MB）', fontsize=12)
plt.xticks(epochs[::2])
plt.grid(True, linestyle='--', alpha=0.7, axis='both')
plt.legend(fontsize=10, loc='upper left')
plt.ylim(0, 1500)
ax2.set_ylim(0, 25)

# 保存图片
plt.savefig('参数传输量对比_20轮.png', dpi=300, bbox_inches='tight')
plt.show()