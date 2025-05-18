import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

G = nx.DiGraph()

server_pos = (4.3, 3)
sync_x = 1.9
sync_start_y = 4.7
sync_y_gap = 1
async_x = 6.5
async_start_y = 4.5
async_y_gap = 1.5
box_x = 0.5
box_y = 0.2
box_width = 3.5
box_height = 5.5
box_gap = 4.15

G.add_node("服务器", pos=server_pos)
sync_clients = [f"客户端{i}" for i in range(1, 6)]
sync_pos = {c: (sync_x, sync_start_y - i * sync_y_gap) for i, c in enumerate(sync_clients)}
for c in sync_clients:
    G.add_node(c, pos=sync_pos[c])
async_clients = [f"客户端{i}" for i in range(6, 9)]
async_pos = {c: (async_x, async_start_y - i * async_y_gap) for i, c in enumerate(async_clients)}
for c in async_clients:
    G.add_node(c, pos=async_pos[c])

for c in sync_clients:
    G.add_edge(c, "服务器", label="同步传输梯度")
for c in async_clients:
    G.add_edge(c, "服务器", label="异步传输梯度")

pos = nx.get_node_attributes(G, 'pos')

fig, ax = plt.subplots(figsize=(12, 7))

nx.draw_networkx_nodes(G, pos, nodelist=["服务器"], node_color='#1976D2', node_size=1500,
                       node_shape='o', edgecolors='black', linewidths=2, ax=ax)

nx.draw_networkx_nodes(G, pos, nodelist=sync_clients, node_color='#1E88E5', node_size=1800,
                       node_shape='s', edgecolors='black', linewidths=1.8, ax=ax)

nx.draw_networkx_nodes(G, pos, nodelist=async_clients, node_color='#FF7043', node_size=1800,
                       node_shape='D', edgecolors='black', linewidths=1.8, ax=ax)

nx.draw_networkx_labels(G, pos, font_size=11, font_family='SimHei', ax=ax)

def draw_curved_edges(G, pos, ax):
    for u, v, data in G.edges(data=True):
        style = data.get('style', 'solid')
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        dx = x2 - x1
        dy = y2 - y1

        curvature = 0.3

        if u == "服务器":
            ctrl_x = mid_x + curvature * (dy if dy != 0 else 1)
            ctrl_y = mid_y - curvature * (dx if dx != 0 else 1)
        else:
            ctrl_x = mid_x - curvature * (dy if dy != 0 else 1)
            ctrl_y = mid_y + curvature * (dx if dx != 0 else 1)

        bezier_points = bezier_curve([x1, ctrl_x, x2], [y1, ctrl_y, y2])

        ax.plot(bezier_points[0], bezier_points[1], color='gray', linestyle=style, linewidth=1.5, zorder=0)

        arrow_pos = 0.6
        arrow_x = (1 - arrow_pos)**2 * x1 + 2 * (1 - arrow_pos) * arrow_pos * ctrl_x + arrow_pos**2 * x2
        arrow_y = (1 - arrow_pos)**2 * y1 + 2 * (1 - arrow_pos) * arrow_pos * ctrl_y + arrow_pos**2 * y2

        dx_arrow = 2 * (1 - arrow_pos) * (ctrl_x - x1) + 2 * arrow_pos * (x2 - ctrl_x)
        dy_arrow = 2 * (1 - arrow_pos) * (ctrl_y - y1) + 2 * arrow_pos * (y2 - ctrl_y)

        ax.arrow(arrow_x, arrow_y, dx_arrow*0.0015, dy_arrow*0.0015,
                 shape='full', lw=0, length_includes_head=True,
                 head_width=0.12, head_length=0.15, color='gray', zorder=1)

        offset_scale = 0.15
        length = np.hypot(dx_arrow, dy_arrow)
        if length == 0:
            length = 1
        perp_x = -dy_arrow / length
        perp_y = dx_arrow / length

        label_x = mid_x + perp_x * offset_scale
        label_y = mid_y + perp_y * offset_scale

        label = data.get('label', '')
        ax.text(label_x, label_y, label, fontsize=9, color='black', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

def bezier_curve(x_points, y_points, num=100):
    t = np.linspace(0, 1, num)
    x = (1 - t)**2 * x_points[0] + 2 * (1 - t) * t * x_points[1] + t**2 * x_points[2]
    y = (1 - t)**2 * y_points[0] + 2 * (1 - t) * t * y_points[1] + t**2 * y_points[2]
    return x, y

draw_curved_edges(G, pos, ax)

def draw_group_box(xy, width, height, label, color, ax):
    rect = plt.Rectangle(xy, width, height, linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.15, zorder=-1)
    ax.add_patch(rect)
    ax.text(xy[0] + 0.1, xy[1] + height - 0.3, label, fontsize=13, weight='bold', color=color, ha='left', va='top')

draw_group_box((box_x, box_y), box_width, box_height, "同步阶段（最多5个客户端）", '#039BE5', ax)
draw_group_box((box_x + box_gap, box_y), box_width, box_height, "异步阶段（超过5个客户端）", '#AFB42B', ax)

ax.set_xlim(-0.5, 9)
ax.set_ylim(-0.5, 6)
ax.axis('off')
plt.tight_layout()
plt.savefig("同步异步客户端交互图_客户端更大.png", dpi=350)
plt.show()
