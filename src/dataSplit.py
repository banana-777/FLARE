# 该文件的作用是将MNIST划分成五份并保存到本地

import torch
from torchvision import datasets, transforms
import numpy as np
import os

# 参数配置
NUM_CLIENTS = 5          # 客户端数量
DATA_ROOT = "./mnist_data" # 数据存储根目录
SEED = 42                # 随机种子

# 创建存储目录
os.makedirs(DATA_ROOT, exist_ok=True)

# 下载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform
)

# IID分配
def split_iid(dataset, num_clients):
    data_indices = list(range(len(dataset)))
    np.random.shuffle(data_indices)
    # 计算每个客户端分配的数据量
    split_size = len(dataset) // num_clients
    splits = [data_indices[i * split_size: (i + 1) * split_size]
              for i in range(num_clients)]
    return splits

# Non-IID分配
def split_non_iid(dataset, num_clients, classes_per_client=2):
    # 按类别组织数据索引
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # 为每个客户端分配类别组合
    client_data = []
    for _ in range(num_clients):
        selected_classes = np.random.choice(
            list(class_indices.keys()),
            size=classes_per_client,
            replace=False
        )

        # 从每个选中类别中随机抽取样本
        client_indices = []
        for cls in selected_classes:
            samples = np.random.choice(
                class_indices[cls],
                size=200,  # 每个类别抽取200个样本
                replace=False
            )
            client_indices.extend(samples)

        client_data.append(client_indices)

    return client_data


def save_client_data(client_id, train_indices, test_indices):
    """ 保存客户端数据到本地 """
    # 保存训练数据
    client_train = {
        'images': torch.stack([train_dataset[i][0] for i in train_indices]),
        'labels': torch.tensor([train_dataset[i][1] for i in train_indices])
    }
    torch.save(client_train,
               os.path.join(DATA_ROOT, f"client_{client_id}_train.pt"))

    # 保存测试数据
    client_test = {
        'images': torch.stack([test_dataset[i][0] for i in test_indices]),
        'labels': torch.tensor([test_dataset[i][1] for i in test_indices])
    }
    torch.save(client_test,
               os.path.join(DATA_ROOT, f"client_{client_id}_test.pt"))


import matplotlib.pyplot as plt


def plot_data_distribution():
    # 统计各客户端标签分布
    label_dist = []
    for client_id in range(NUM_CLIENTS):
        data = torch.load(os.path.join(DATA_ROOT, f"client_{client_id}_train.pt"))
        label_dist.append(np.bincount(data['labels'].numpy()))

    # 绘制热力图
    plt.figure(figsize=(12, 6))
    plt.imshow(label_dist, cmap='Blues', aspect='auto')
    plt.xlabel('标签分布')
    plt.ylabel('客户端ID')
    plt.colorbar(label='标签计数')
    plt.title('客户端数据分布热图')
    plt.show()


# 主处理流程
if __name__ == "__main__":
    np.random.seed(SEED)

    # 划分训练数据
    train_splits = split_iid(train_dataset, NUM_CLIENTS)  # 可切换为split_non_iid

    # 划分测试数据（所有客户端共享完整测试集）
    test_indices = list(range(len(test_dataset)))

    # 保存各客户端数据
    for client_id in range(NUM_CLIENTS):
        save_client_data(client_id, train_splits[client_id], test_indices)
        print(f"客户端 {client_id} 数据已保存 | 训练样本数: {len(train_splits[client_id])}")

    plot_data_distribution()