import matplotlib.pyplot as plt
import networkx as nx

# 创建有向图
G = nx.DiGraph()

# 添加节点（层）
layers = ['输入层 (28x28)', '卷积层1 (32)', '池化层1 (14x14)',
          '卷积层2 (64)', '池化层2 (7x7)', '全连接层1 (128)', '输出层 (10)']

# 设置节点位置（手动布局）
pos = {
    '输入层 (28x28)': (0, 0),
    '卷积层1 (32)': (1, 0),
    '池化层1 (14x14)': (2, 0),
    '卷积层2 (64)': (3, 0),
    '池化层2 (7x7)': (4, 0),
    '全连接层1 (128)': (5, 0),
    '输出层 (10)': (6, 0)
}

# 添加节点和边
for layer in layers:
    G.add_node(layer)

edges = [
    ('输入层 (28x28)', '卷积层1 (32)'),
    ('卷积层1 (32)', '池化层1 (14x14)'),
    ('池化层1 (14x14)', '卷积层2 (64)'),
    ('卷积层2 (64)', '池化层2 (7x7)'),
    ('池化层2 (7x7)', '全连接层1 (128)'),
    ('全连接层1 (128)', '输出层 (10)')
]

G.add_edges_from(edges)

# 绘制图形
plt.figure(figsize=(12, 6))
nx.draw(
    G, pos, with_labels=True,
    node_size=3000,
    node_color='lightblue',
    font_size=10,
    font_weight='bold',
    arrows=True,
    arrowstyle='->',
    arrowsize=20,
    edge_color='gray'
)

# 添加标题和说明
plt.title('MNIST CNN 神经网络拓扑图', fontsize=16)
plt.axis('off')
plt.tight_layout()

# 保存图像
plt.savefig('nn_topology_simplified.png', dpi=300, bbox_inches='tight')
plt.show()