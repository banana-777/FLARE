import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arc, Circle

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6)
ax.axis('off')

# 坐标参数
client_x = 2.7
server_x = 8.0
client_ys = [4.7, 3.0, 1.3]
server_y = 3.0

# 神经网络结构
def draw_nn(ax, center, node_radius=0.07, color="#6baed6"):
    input_ys = [center[1]+0.18, center[1], center[1]-0.18]
    hidden_ys = [center[1]+0.09, center[1]-0.09]
    output_ys = [center[1]+0.18, center[1], center[1]-0.18]
    input_x, hidden_x, output_x = center[0]-0.19, center[0], center[0]+0.19
    for y in input_ys:
        ax.add_patch(Circle((input_x, y), node_radius, color=color, ec='k', lw=0.5, zorder=6))
    for y in hidden_ys:
        ax.add_patch(Circle((hidden_x, y), node_radius, color=color, ec='k', lw=0.5, zorder=6))
    for y in output_ys:
        ax.add_patch(Circle((output_x, y), node_radius, color=color, ec='k', lw=0.5, zorder=6))
    for y0 in input_ys:
        for y1 in hidden_ys:
            ax.plot([input_x, hidden_x], [y0, y1], '-', color=color, lw=0.8, zorder=5)
    for y0 in hidden_ys:
        for y2 in output_ys:
            ax.plot([hidden_x, output_x], [y0, y2], '-', color=color, lw=0.8, zorder=5)

# 客户端
for i, cy in enumerate(client_ys):
    cli_box = FancyBboxPatch(
        (client_x-0.8, cy-0.55), 1.60, 1.24,
        boxstyle="round,pad=0.08", edgecolor="#228B22", facecolor='white', linewidth=2)
    ax.add_patch(cli_box)
    ax.text(client_x, cy+0.37, f"Client {i+1}", fontsize=13, ha='center', va='bottom', color="#228B22", fontweight='semibold')
    ax.text(client_x, cy+0.16, "本地数据/训练", fontsize=9.5, ha='center', va='bottom', color="#444")
    # datacircle = Circle((client_x-1.4, cy-0.27), 0.14, ec='#888', fc='#ddd', lw=2, zorder=2)
    # ax.add_patch(datacircle)
    # ax.text(client_x-1.4, cy-0.27, "D", fontsize=12, ha='center', va='center', color='#666')
    draw_nn(ax, (client_x, cy-0.18), color="#A3E4D7")

# 服务器
server_box = FancyBboxPatch(
    (server_x-0.8, server_y-0.55), 1.6, 1.4,
    boxstyle="round,pad=0.07", edgecolor="#377eb8", facecolor='white', linewidth=2)
ax.add_patch(server_box)
ax.text(server_x, server_y+0.35, "Server\n（聚合中心）", fontsize=15, ha='center', va='center', color="#377eb8", fontweight='semibold')
draw_nn(ax, (server_x, server_y-0.18), color="#5DADE2")

# 箭头及文本（精细对齐）
arrow_args_up   = dict(arrowstyle='-|>', lw=2, color='#e41a1c', shrinkA=11, shrinkB=11)
arrow_args_down = dict(arrowstyle='-|>', lw=2, color='#377eb8', shrinkA=13, shrinkB=13)

# 箭头起止点微调，文字紧贴箭头中点
for i, cy in enumerate(client_ys):
    # 上行箭头
    up_x0 = client_x+0.75
    up_y0 = cy+0.3
    up_x1 = server_x-0.75
    up_y1 = server_y + [0.5,0.15,-0.3][i]
    ax.annotate('', xy=(up_x1, up_y1), xytext=(up_x0, up_y0), arrowprops=arrow_args_up, zorder=10)
    # 文字紧贴箭头中点，错层排列
    ax.text((up_x0+up_x1)/2 - 0.18, (up_y0+up_y1)/2 + 0.13, "上传参数", fontsize=11, color='#e41a1c', va='center', ha='left')

    # 下行箭头
    down_x0 = server_x-0.75
    down_y0 = server_y + [0.3,0,-0.5][i]
    down_x1 = client_x+0.65
    down_y1 = cy-0.18
    ax.annotate('', xy=(down_x1, down_y1), xytext=(down_x0, down_y0), arrowprops=arrow_args_down, zorder=10)
    ax.text((down_x0+down_x1)/2 - 0.2, (down_y0+down_y1)/2 - 0.13, "下发全局模型", fontsize=11, color='#377eb8', va='center', ha='left')

# 多轮通信
arc = Arc(
    (server_x+1.65, server_y + 0.1), 1.05, 1.05, angle=0, theta1=60, theta2=300, color="#8e44ad", lw=2.4
)
ax.add_patch(arc)
ax.annotate("", xy=(server_x+1.65, server_y+0.625), xytext=(server_x+1.665, server_y+0.525),
            arrowprops=dict(arrowstyle='-|>', color="#8e44ad", lw=2.4, mutation_scale=18))
ax.text(server_x+2.1, server_y + 0.09, "多轮通信\n(repeat)", fontsize=12.5, color="#8e44ad", rotation=-90, va='center', ha='center')

# 图例
legend_x = 10
legend_y = 0.65
legend_nn_c = (legend_x+0.55, legend_y+0.7)
ax.add_patch(FancyBboxPatch((legend_x, legend_y), 1.15, 1.12, boxstyle="round,pad=0.09", facecolor="#fff", edgecolor="#aaa", lw=1))
draw_nn(ax, legend_nn_c, color="#a0cfff")
ax.text(legend_x+1.35, legend_y+0.7, "神经网络结构\n(模型)", fontsize=11, ha='left', va='center', color="#222")
# 图例箭头
ax.annotate('', xy=(legend_x + 0.95, legend_y + 0.25), xytext=(legend_x + 0.2, legend_y + 0.25),
            arrowprops=dict(arrowstyle='-|>', color='#e41a1c', lw=1.4), zorder=13)
ax.annotate('', xy=(legend_x + 0.95, legend_y + 0.05), xytext=(legend_x+0.2, legend_y + 0.05),
            arrowprops=dict(arrowstyle='-|>', color='#377eb8', lw=1.4), zorder=13)
ax.text(legend_x+1.35, legend_y+0.2, "上传参数", va="center", fontsize=10, color='#e41a1c', ha='left')
ax.text(legend_x-1.35, legend_y+0.15, "下发全局模型", va="center", fontsize=10, color='#377eb8', ha='left')

ax.set_aspect('equal')
plt.tight_layout(pad=1.8)
plt.savefig("federated_learning_flow_v4.png", dpi=300)
plt.show()
