import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 防乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(11, 7))
ax.axis('off')

# ====== 定义各层内容与位置 ======

# 层次描述（由高到低排列）
layers = [
    # 展示层
    {
        "y": 8.5, "height": 2.1,
        "name": "展示层",
        "modules": ["服务器GUI", "客户端GUI", "状态展示", "日志输出"],
        "xs": [1.3, 4.0, 6.7, 9.4]
    },
    # 通信层
    {
        "y": 5.7, "height": 2.1,
        "name": "通信层",
        "modules": ["模型结构交互", "模型参数交互", "STC", "AES加密", "压缩算法"],
        "xs": [0.85, 2.9, 5.0, 7.1, 9.2]
    },
    # 训练层
    {
        "y": 2.6, "height": 2.1,
        "name": "训练层",
        "modules": ["本地训练", "参数聚合", "模型评估"],
        "xs": [3.0, 5.0, 7.0]
    }
]

# ====== 绘图主体 ======

# 外整体大框
ax.add_patch(Rectangle((0.35, 1.8), 11.1, 9.6, fill=False, lw=2.0, zorder=0))

# 分层虚线框以及层名
for layer in layers:
    ax.add_patch(Rectangle((0.6, layer["y"]), 10.6, layer["height"],
                           fill=False, ls='dashed', lw=1.5, zorder=1))
    ax.text(0.6, layer["y"] + layer["height"] + 0.3,
            layer["name"], fontsize=14, ha='left', va='center', weight='bold')

    # 画模块框与文字
    for x, mod in zip(layer["xs"], layer["modules"]):
        ax.add_patch(Rectangle((x, layer["y"] + 0.47), 1.6, 1.07,
                               fill=False, lw=1.3))
        ax.text(x + 0.8, layer["y"] + 0.47 + 0.54,
                mod, fontsize=13, ha='center', va='center')

# （可以适当添加箭头连接，不加也很美观）

# 图注，可选
# ax.text(6, 1, "", fontsize=12.5, ha='center', va='center', family='SimHei')

plt.xlim(0, 12)
plt.ylim(0, 12)
plt.tight_layout()
plt.savefig("联邦学习系统分层架构图.png", dpi=350)
plt.show()
