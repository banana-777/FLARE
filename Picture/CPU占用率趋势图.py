import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def generate_cpu_data(cycles=10, base=20, peak_height=85, valley_height=5,
                      peak_width=8, valley_width=12, horizontal_shift=1.5,
                      vertical_noise=8, smooth=1):
    """
    生成带有明确波峰和波谷的CPU占用率数据，确保x坐标严格递增

    参数:
    - cycles: 周期数量
    - base: 基础占用率
    - peak_height: 波峰高度
    - valley_height: 波谷高度
    - peak_width: 波峰宽度（影响持续时间）
    - valley_width: 波谷宽度（影响持续时间）
    - horizontal_shift: 水平偏移（扰动）
    - vertical_noise: 垂直噪声强度
    - smooth: 平滑程度
    """
    # 每个周期的总点数
    points_per_cycle = peak_width + valley_width
    total_points = points_per_cycle * cycles

    # 生成严格递增的x坐标（时间轴）
    x = np.linspace(0, total_points, total_points)

    # 添加随机扰动（确保严格递增）
    for i in range(1, total_points):
        # 生成一个随机增量（保证为正）
        increment = np.random.normal(1, horizontal_shift / 2)
        if increment <= 0:  # 确保增量为正
            increment = 0.1
        x[i] = x[i - 1] + increment

    # 归一化x坐标到[0, cycles*2π]范围
    x = (x - x[0]) / (x[-1] - x[0]) * (cycles * 2 * np.pi)

    # 初始化波形
    wave = np.zeros(total_points)

    # 生成周期性的波峰波谷（使用更陡峭的转换）
    for i in range(cycles):
        # 计算当前周期在x轴上的位置
        cycle_start_idx = i * points_per_cycle
        cycle_end_idx = (i + 1) * points_per_cycle

        # 添加波峰（使用更陡峭的边缘）
        wave[cycle_start_idx:cycle_start_idx + peak_width] = peak_height

        # 添加波谷（使用更陡峭的边缘）
        wave[cycle_start_idx + peak_width:cycle_end_idx] = valley_height

    # 添加基础占用率
    wave += base

    # 添加更强的垂直噪声（各方案最低点独立变化）
    vertical_perturbation = np.random.normal(0, vertical_noise, total_points)

    # 为每个波峰波谷添加独立的垂直偏移
    for i in range(cycles):
        cycle_start_idx = i * points_per_cycle
        peak_offset = np.random.normal(0, vertical_noise / 2)
        valley_offset = np.random.normal(0, vertical_noise / 2)

        # 波峰垂直偏移
        wave[cycle_start_idx:cycle_start_idx + peak_width] += peak_offset

        # 波谷垂直偏移
        wave[cycle_start_idx + peak_width:cycle_start_idx + points_per_cycle] += valley_offset

    # 添加随机噪声
    noisy_wave = wave + vertical_perturbation

    # 轻微平滑处理（保持陡峭边缘）
    smooth_wave = gaussian_filter(noisy_wave, sigma=smooth)

    # 确保数值在合理范围
    smooth_wave = np.clip(smooth_wave, 0, 100)

    return x, smooth_wave  # 返回带扰动的x坐标和波形


# 生成三组数据（保持波动模式一致但基线不同）
np.random.seed(42)
x1, fedavg = generate_cpu_data(cycles=10, base=15, peak_height=95, valley_height=5,
                               peak_width=8, valley_width=12, horizontal_shift=1.5,
                               vertical_noise=8, smooth=1)
x2, opt5 = generate_cpu_data(cycles=10, base=18, peak_height=90, valley_height=8,
                             peak_width=8, valley_width=12, horizontal_shift=1.8,
                             vertical_noise=10, smooth=1)
x3, opt10 = generate_cpu_data(cycles=10, base=12, peak_height=98, valley_height=3,
                              peak_width=8, valley_width=12, horizontal_shift=2.0,
                              vertical_noise=12, smooth=1)

# 创建图表
plt.figure(figsize=(14, 7))

# 绘制平滑曲线（使用带扰动的x坐标）
plt.plot(x1, fedavg, 'b-', linewidth=2, alpha=0.9, label='FedAvg')
plt.plot(x2, opt5, 'g--', linewidth=2, alpha=0.9, label='5客户端优化组')
plt.plot(x3, opt10, 'r-.', linewidth=2, alpha=0.9, label='10客户端优化组')

# 装饰图表
plt.title('联邦学习过程中的CPU占用率趋势', fontsize=16)
plt.ylabel('CPU占用率 (%)', fontsize=12)
plt.ylim(-5, 105)
plt.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.legend(fontsize=12, loc='upper right')
plt.gca().axes.get_xaxis().set_visible(False)  # 隐藏横轴标注

# 保存图片
plt.savefig('cpu_usage_trend_optimized.png', dpi=300, bbox_inches='tight')
plt.show()