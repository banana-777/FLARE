import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 创建保存图像的目录
os.makedirs("mnist_plots", exist_ok=True)


# 加载MNIST数据集
def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True,
                                   download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False,
                                  download=True, transform=transform)
    return train_dataset, test_dataset


# IID划分：随机均匀分配
def iid_partition(dataset, num_clients):
    num_items = int(len(dataset) / num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# Non-IID划分：按标签排序后分配
def non_iid_partition(dataset, num_clients, num_shards_per_client=2):
    total_shards = num_clients * num_shards_per_client
    num_imgs_per_shard = int(len(dataset) / total_shards)

    # 获取所有样本的标签
    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # 按标签排序
    idxs_labels = np.vstack((np.arange(len(dataset)), labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # 划分成碎片
    shard_idxs = [idxs[i * num_imgs_per_shard:(i + 1) * num_imgs_per_shard]
                  for i in range(total_shards)]

    # 为每个客户端分配碎片
    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}
    for i in range(num_clients):
        shard_indices = np.random.choice(range(total_shards), num_shards_per_client, replace=False)
        for shard in shard_indices:
            dict_users[i] = np.concatenate((dict_users[i], shard_idxs[shard]), axis=0)
    return dict_users


# 绘制IID划分的数据分布图
def plot_iid_distribution(dataset, partition, num_clients, save_path="mnist_plots/iid_distribution.png"):
    fig, axes = plt.subplots(1, num_clients, figsize=(20, 4))

    for i in range(num_clients):
        client_indices = list(partition[i])
        client_labels = [dataset[idx][1] for idx in client_indices]
        label_counts = [client_labels.count(j) for j in range(10)]

        axes[i].bar(range(10), label_counts)
        axes[i].set_title(f'客户端 {i + 1}')
        axes[i].set_xlabel('数字类别')
        axes[i].set_ylabel('样本数量')
        axes[i].set_xticks(range(10))

    plt.suptitle('IID划分: 各客户端数字类别分布')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为suptitle留出空间
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# 绘制Non-IID划分的数据分布图
def plot_non_iid_distribution(dataset, partition, num_clients, save_path="mnist_plots/non_iid_distribution.png"):
    fig, axes = plt.subplots(1, num_clients, figsize=(20, 4))

    for i in range(num_clients):
        client_indices = list(partition[i])
        client_labels = [dataset[idx][1] for idx in client_indices]
        label_counts = [client_labels.count(j) for j in range(10)]

        axes[i].bar(range(10), label_counts)
        axes[i].set_title(f'客户端 {i + 1}')
        axes[i].set_xlabel('数字类别')
        axes[i].set_ylabel('样本数量')
        axes[i].set_xticks(range(10))

    plt.suptitle('Non-IID划分: 各客户端数字类别分布')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为suptitle留出空间
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# 绘制样本数量饼图
def plot_client_samples(partition, num_clients, save_path="mnist_plots/client_samples.png"):
    client_sizes = [len(partition[i]) for i in range(num_clients)]

    plt.figure(figsize=(8, 6))
    plt.pie(client_sizes, labels=[f'客户端 {i + 1}' for i in range(num_clients)],
            autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # 使饼图为正圆形
    plt.title('各客户端样本数量分布')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# 主函数
def main():
    # 加载数据
    train_dataset, _ = load_mnist()
    num_clients = 5

    # IID划分
    iid_part = iid_partition(train_dataset, num_clients)
    plot_iid_distribution(train_dataset, iid_part, num_clients,
                          "mnist_plots/iid_class_distribution.png")
    plot_client_samples(iid_part, num_clients, "mnist_plots/iid_client_samples.png")

    # Non-IID划分
    non_iid_part = non_iid_partition(train_dataset, num_clients)
    plot_non_iid_distribution(train_dataset, non_iid_part, num_clients,
                              "mnist_plots/non_iid_class_distribution.png")
    plot_client_samples(non_iid_part, num_clients, "mnist_plots/non_iid_client_samples.png")

    print("数据分布图已保存至'mnist_plots'目录")


if __name__ == "__main__":
    main()