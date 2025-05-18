import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

# 创建画布
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
ax.set_title("客户端本地训练时序图", fontsize=12, pad=15)

# 定义步骤位置（y轴分层：数据层、计算层、控制层）
y_data = 5.5    # 数据操作层
y_compute = 3   # 计算操作层
y_control = 1   # 控制流层

# 1. 数据加载阶段
data_load = patches.Rectangle(
    (0.5, y_data-1.2), 2, 1.2,
    facecolor='#4CAF50', edgecolor='black', alpha=0.9
)
ax.add_patch(data_load)
ax.text(1.5, y_data-0.6, "数据加载", ha='center', va='center',
        fontsize=10, color='white', weight='bold')
ax.text(1.5, y_data-1.2, "从本地路径读取client_train.pt/client_test.pt",
        ha='center', va='center', fontsize=8, color='white', style='italic')

# 2. 数据预处理
preprocess = patches.Rectangle(
    (3.5, y_data-1.2), 2, 1.2,
    facecolor='#2196F3', edgecolor='black', alpha=0.9
)
ax.add_patch(preprocess)
ax.text(4.5, y_data-0.6, "数据预处理", ha='center', va='center',
        fontsize=10, color='white', weight='bold')
ax.text(4.5, y_data-1.2, "标准化/划分批次（Batch Size=128）",
        ha='center', va='center', fontsize=8, color='white', style='italic')

# 3. 前向传播流程（计算层）
forward = patches.Rectangle(
    (0.5, y_compute-1.5), 3, 1.5,
    facecolor='#64B5F6', edgecolor='black', hatch='//'
)
ax.add_patch(forward)
ax.text(2, y_compute-0.75, "前向传播", ha='center', va='center',
        fontsize=10, weight='bold')
ax.text(2, y_compute-1.3, "卷积层 → ReLU激活 → 池化层 → 全连接层",
        ha='center', va='center', fontsize=8, style='italic')

# 4. 损失计算
loss_calc = patches.Rectangle(
    (4, y_compute-1.2), 2, 1.2,
    facecolor='#FFC107', edgecolor='black'
)
ax.add_patch(loss_calc)
ax.text(5, y_compute-0.6, "损失计算", ha='center', va='center',
        fontsize=10, weight='bold')
ax.text(5, y_compute-1.2, "交叉熵损失函数（CrossEntropyLoss）",
        ha='center', va='center', fontsize=8, style='italic')

# 5. 反向传播与参数更新（关键步骤）
backward = patches.Rectangle(
    (0.5, y_control-1), 5, 1,
    facecolor='#F44336', edgecolor='black', alpha=0.8
)
ax.add_patch(backward)
ax.text(3, y_control-0.5, "反向传播 & 参数更新", ha='center', va='center',
        fontsize=10, color='white', weight='bold')
ax.text(3, y_control-1, "SGD优化器（lr=0.01, momentum=0.9）",
        ha='center', va='center', fontsize=8, color='white', style='italic')

# 6. 循环训练轮次（控制流）
arrow = FancyArrowPatch(
    (6, y_control-0.5), (1, y_control-0.5),
    connectionstyle="arc3,rad=0",
    arrowstyle="Simple, tail_width=1, head_width=8, head_length=10",
    color="black",
    linestyle='--',
    mutation_scale=15
)
ax.add_patch(arrow)
ax.text(6.2, y_control-0.5, "训练轮次循环", ha='left', va='center',
        fontsize=9, style='italic', rotation=180)

# 7. 结果保存
save_step = patches.Rectangle(
    (7, y_data-1.2), 2, 1.2,
    facecolor='#9E9E9E', edgecolor='black'
)
ax.add_patch(save_step)
ax.text(8, y_data-0.6, "模型参数保存", ha='center', va='center',
        fontsize=10, color='white', weight='bold')
ax.text(8, y_data-1.2, "生成加密传输的参数包（STC压缩后）",
        ha='center', va='center', fontsize=8, color='white', style='italic')

# 添加时间轴
ax.plot([0, 10], [0, 0], color='black', linewidth=1.2)
for x in [1.5, 4.5, 5, 6, 8]:
    ax.plot([x, x], [0, 0.2], color='black', linewidth=1)
ax.text(5, -0.5, "时间 →", ha='center', va='center', fontsize=10, weight='bold')

# 美化设置
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')
plt.tight_layout()

# 保存为矢量图
plt.savefig("client_training_timeline.png", dpi=600, bbox_inches='tight')
plt.show()