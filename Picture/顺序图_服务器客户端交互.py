import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 4.5))

# 参与者及其水平位置
participants = ['服务器', '客户端']
x_pos = {p: i*3 for i, p in enumerate(participants)}

# 蓝绿配色 + 黑色箭头文字
colors = {
    'server_box_edge': '#1F497D',  # 深蓝
    'server_box_face': '#D9E1F2',  # 浅蓝
    'client_box_edge': '#548235',  # 深绿
    'client_box_face': '#D9EAD3',  # 浅绿
    'server_activation': '#9BBB59',  # 绿黄（服务器激活块）
    'client_activation': '#C6D9F1',  # 浅蓝（客户端激活块）
    'arrow': '#000000',  # 黑色箭头和文字
    'text': '#000000',  # 黑色文字
    'wait_text': '#A05252'  # 棕红等待接收文字
}

# 画参与者框和名字
for p in participants:
    if p == '服务器':
        edge_c = colors['server_box_edge']
        face_c = colors['server_box_face']
    else:
        edge_c = colors['client_box_edge']
        face_c = colors['client_box_face']

    rect = patches.FancyBboxPatch((x_pos[p]-0.8, 3.8), 1.6, 0.6,
                                  boxstyle="round,pad=0.1",
                                  linewidth=1.8, edgecolor=edge_c, facecolor=face_c, zorder=5)
    ax.add_patch(rect)
    ax.text(x_pos[p], 4.1, p, ha='center', va='bottom', fontsize=15, weight='bold', color=edge_c, zorder=6)

# 画生命线（虚线）
for p in participants:
    ax.plot([x_pos[p], x_pos[p]], [0, 3.8], color=colors['arrow'], linestyle='--', linewidth=1.3, zorder=1)

# 消息列表： (发送者, 接收者, 消息内容, y坐标, 激活者)
messages = [
    ('服务器', '客户端', 'SEND_MODEL_STRUCTURE', 3.3, '客户端'),
    ('客户端', '服务器', 'READY_MODEL_STRUCTURE', 2.7, '服务器'),
    ('服务器', '客户端', '发送模型结构（含SHA256）', 2.1, '服务器'),
    ('客户端', '服务器', 'MODEL_STRUCTURE_RECEIVED', 1.5, '客户端'),
]

# 画激活块函数
def draw_activation(ax, participant, y_start, y_end, color):
    x = x_pos[participant]
    width = 0.6
    rect = patches.Rectangle((x - width/2, y_end), width, y_start - y_end,
                             linewidth=0, facecolor=color, alpha=0.5, zorder=3)
    ax.add_patch(rect)

# 画消息箭头函数
def draw_message(ax, sender, receiver, y, text):
    x_start = x_pos[sender]
    x_end = x_pos[receiver]
    ax.annotate("",
                xy=(x_end, y), xycoords='data',
                xytext=(x_start, y), textcoords='data',
                arrowprops=dict(arrowstyle="->", color=colors['arrow'], lw=2))
    ax.text((x_start + x_end)/2, y + 0.1, text, ha='center', va='bottom', fontsize=11, color=colors['text'])

# 依次画激活块和消息
for i, (sender, receiver, text, y, active) in enumerate(messages):
    if active == '服务器':
        color = colors['server_activation']
    else:
        color = colors['client_activation']
    y_next = messages[i+1][3] if i+1 < len(messages) else 0.5
    draw_activation(ax, active, y_start=y, y_end=y_next, color=color)
    draw_message(ax, sender, receiver, y, text)

# 画底部小圆点表示生命线结束
for p in participants:
    ax.plot(x_pos[p], 0.5, 'o', color=colors['arrow'], markersize=6, zorder=6)

# 添加“等待接收”文字，客户端下移0.1单位
offset = 0.1

# 服务器等待接收：2.7 ~ 2.1
wait_server_y = (2.7 + 2.1) / 2
ax.text(x_pos['服务器'], wait_server_y, "等待接收", ha='center', va='center',
        fontsize=10, color=colors['wait_text'], style='italic', alpha=0.8)

# 客户端等待接收两段，下移offset
wait_client_y1 = (3.3 + 2.7) / 2 - offset
ax.text(x_pos['客户端'], wait_client_y1 + 0.1, "等待接收", ha='center', va='center',
        fontsize=10, color=colors['wait_text'], style='italic', alpha=0.8)

wait_client_y2 = (2.1 + 1.5) / 2 - offset
ax.text(x_pos['客户端'], wait_client_y2 - 0.6, "等待接收", ha='center', va='center',
        fontsize=10, color=colors['wait_text'], style='italic', alpha=0.8)

# 设置坐标轴范围和隐藏坐标轴
ax.set_xlim(-1, x_pos[participants[-1]] + 1)
ax.set_ylim(0, 4.5)
ax.axis('off')

# 调整上下边距，避免内容被裁剪
plt.subplots_adjust(top=0.95, bottom=0.1)

plt.tight_layout()
plt.savefig("时序图_服务器客户端交互_蓝绿配色_等待接收下移黑色箭头.png", dpi=350)
plt.show()
