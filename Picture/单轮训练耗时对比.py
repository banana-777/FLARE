import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import os
from datetime import datetime

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 数据准备
client_labels = ['客户端1', '客户端2', '客户端3', '客户端4', '客户端5']
fedavg_data = [22.3, 21.8, 22.1, 22.5, 22.0]
opt5_data = [21.8, 21.5, 21.9, 22.4, 22.2]
opt10_data = [21.5, 21.2, 21.6, 22.0, 21.9]

# 创建数据框用于统计
df = pd.DataFrame({
    '客户端': client_labels,
    'FedAvg': fedavg_data,
    '5客户端优化组': opt5_data,
    '10客户端优化组': opt10_data
})

# 计算各客户端的效率提升百分比
df['5客户端优化提升(%)'] = ((df['FedAvg'] - df['5客户端优化组']) / df['FedAvg'] * 100).round(2)
df['10客户端优化提升(%)'] = ((df['FedAvg'] - df['10客户端优化组']) / df['FedAvg'] * 100).round(2)

# 设置图表样式
colors = ['#165DFF', '#36D399', '#FF9F43']  # 蓝色、绿色、橙色
bar_width = 0.25
index = np.arange(len(client_labels))

# 创建横向双图布局（1行2列）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
plt.subplots_adjust(wspace=0.3, top=0.92)

# ------------------- 左图：各客户端耗时对比（原右上角子图2） -------------------
bar1 = ax1.bar(index - bar_width, fedavg_data, bar_width, label='FedAvg', color=colors[0], alpha=0.8)
bar2 = ax1.bar(index, opt5_data, bar_width, label='5客户端优化组', color=colors[1], alpha=0.8)
bar3 = ax1.bar(index + bar_width, opt10_data, bar_width, label='10客户端优化组', color=colors[2], alpha=0.8)

ax1.set_title('各客户端单轮训练耗时对比', fontsize=14, pad=10)
ax1.set_xlabel('客户端', fontsize=12)
ax1.set_ylabel('耗时 (秒)', fontsize=12)
ax1.set_xticks(index)
ax1.set_xticklabels(client_labels)
ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)

# 添加数值标签
def add_labels(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

add_labels(bar1, ax1)
add_labels(bar2, ax1)
add_labels(bar3, ax1)

# ------------------- 右图：效率提升百分比（原左下角子图3） -------------------
bar_width_pct = 0.35
bar1_pct = ax2.bar(index - bar_width_pct/2, df['5客户端优化提升(%)'], bar_width_pct,
                  label='5客户端优化组提升', color=colors[1], alpha=0.8)
bar2_pct = ax2.bar(index + bar_width_pct/2, df['10客户端优化提升(%)'], bar_width_pct,
                  label='10客户端优化组提升', color=colors[2], alpha=0.8)

ax2.set_title('优化组相对于FedAvg的效率提升百分比', fontsize=14, pad=10)
ax2.set_xlabel('客户端', fontsize=12)
ax2.set_ylabel('效率提升 (%)', fontsize=12)
ax2.set_xticks(index)
ax2.set_xticklabels(client_labels)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)

# 添加数值标签
add_labels(bar1_pct, ax2)
add_labels(bar2_pct, ax2)

# 设置主标题
plt.suptitle('联邦学习单轮训练耗时与效率提升对比', fontsize=18, fontweight='bold', y=1.0)

# 保存图片
def save_combined_figure(fig, title='combined_plots', directory='results', fmt='png'):
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{title}_{timestamp}.{fmt}"
    file_path = os.path.join(directory, filename)
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"已保存合并图: {file_path}")
    plt.close(fig)

save_combined_figure(fig, title='耗时与效率提升对比')

# 显示图表（可选，保存后可注释掉）
# plt.show()