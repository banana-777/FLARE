import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# 解决中文字体问题（优先使用系统自带的中文字体）
plt.rcParams["font.family"] = ["SimHei", "Times New Roman"]  # 中文使用黑体，英文使用Times New Roman
plt.rcParams["axes.unicode_minus"] = False

# 定义仅用于可视化的增强转换
augment_visualization_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomRotation(15),
])

# 加载数据集（使用PIL图像版本）
train_data = datasets.MNIST(
    root='./data', train=True, download=True, transform=None
)

# 创建保存目录
os.makedirs('paper_figures', exist_ok=True)


def create_comparison_figure(
        dataset: torch.utils.data.Dataset,
        num_samples: int = 5,
        num_augmentations: int = 3,
        dpi: int = 300,
        border_width: float = 0.8  # 边框宽度（磅）
) -> None:
    """生成带边框的多样本数据增强对比图"""
    fig, axes = plt.subplots(
        nrows=num_samples,
        ncols=num_augmentations + 1,
        figsize=(4 * (num_augmentations + 1), 1.5 * num_samples),
        dpi=dpi,
        gridspec_kw={'wspace': 0.1, 'hspace': 0.3}
    )

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for row, idx in enumerate(indices):
        # 获取原始PIL图像
        img, label = dataset[idx]

        # 绘制原始图像
        ax = axes[row, 0]
        ax.imshow(img, cmap='gray_r')
        ax.set_title(f'原始图像\nLabel: {label}', fontsize=8)
        ax.axis('off')

        # 添加图像边框
        for spine in ax.spines.values():
            spine.set_linewidth(border_width)

        # 绘制增强图像
        for col in range(num_augmentations):
            aug_img = augment_visualization_transform(img.copy())
            ax = axes[row, col + 1]
            ax.imshow(aug_img, cmap='gray_r')
            ax.set_title(f'增强#{col + 1}', fontsize=8)
            ax.axis('off')

            # 添加图像边框
            for spine in ax.spines.values():
                spine.set_linewidth(border_width)

    # 设置全局标题和注释的字体（中文用黑体，英文用Times New Roman）
    fig.suptitle(
        '数据增强效果对比（随机裁剪+旋转）',
        fontsize=10,
        y=1.02,
        fontfamily="SimHei"  # 中文标题使用黑体
    )
    fig.text(
        0.5, 0.01,
        '注：每行展示一个原始样本及其增强版本，增强操作包括随机4像素裁剪和±15°旋转',
        ha='center',
        fontsize=8,
        fontfamily="SimHei"  # 中文注释使用黑体
    )

    # 单独设置英文标签的字体为Times New Roman
    for ax in axes.flat:
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontfamily('Times New Roman')

    fig.savefig(
        'paper_figures/data_augmentation_comparison.png',
        dpi=dpi,
        bbox_inches='tight'
    )
    fig.savefig(
        'paper_figures/data_augmentation_comparison.pdf',
        dpi=dpi,
        bbox_inches='tight'
    )

    print(f"已生成带边框的对比图，保存至 'paper_figures' 目录")
    plt.close(fig)


# 运行函数生成图表
create_comparison_figure(train_data, num_samples=5, num_augmentations=3)