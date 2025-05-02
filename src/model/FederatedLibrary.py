# 04 29
# 训练使用到的库及函数

import os
import re
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from MNIST_CNN import Model_CNN
from pathlib import Path

# 配置参数
class FLConfig:
    # 联邦学习参数
    NUM_CLIENTS = 10  # 总客户端数
    FRAC_CLIENTS = 0.5  # 每轮选择比例
    NUM_ROUNDS = 20  # 通信轮次

    # 训练参数
    BATCH_SIZE = 512
    LOCAL_EPOCHS = 3
    LEARNING_RATE = 0.01

    # 数据分布模式
    IID = True  # 是否使用IID分布
    CLASS_PER_CLIENT = 2  # 非IID时每个客户端类别数

    # 系统参数
    SAVE_PATH = "./fl_models"
    LOG_FILE = "training.log"

# 数据分区器
class DataPartitioner:
    def __init__(self, dataset, num_clients, iid=True, class_per_client=2):
        self.dataset = dataset
        self.num_clients = num_clients
        self.iid = iid
        self.class_per_client = class_per_client

        if iid:
            self._iid_partition()
        else:
            self._noniid_partition()

    def _iid_partition(self):
        idxs = np.random.permutation(len(self.dataset))
        self.partitions = np.array_split(idxs, self.num_clients)

    def _noniid_partition(self):
        label_idxs = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(self.dataset):
            label_idxs[label].append(idx)

        client_data = [[] for _ in range(self.num_clients)]
        for label in range(10):
            np.random.shuffle(label_idxs[label])
            splits = np.array_split(label_idxs[label], self.num_clients // self.class_per_client)
            for i, split in enumerate(splits):
                client_idx = i % self.num_clients
                client_data[client_idx].extend(split)

        self.partitions = [np.array(idxs) for idxs in client_data]

    def get_subset(self, client_id):
        return Subset(self.dataset, self.partitions[client_id])

# 联邦客户端
class FLClient:
    def __init__(self, model, train_data, device):
        self.model = model.to(device)
        self.train_loader = DataLoader(train_data,
                                       batch_size=FLConfig.BATCH_SIZE,
                                       shuffle=True)
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=FLConfig.LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def local_train(self):
        self.model.train()
        for _ in range(FLConfig.LOCAL_EPOCHS):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()

# 联邦服务器
class FLServer:
    def __init__(self, global_model, test_loader, device):
        self.global_model = global_model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.history = {'accuracy': [], 'loss': []}

        if not os.path.exists(FLConfig.SAVE_PATH):
            os.makedirs(FLConfig.SAVE_PATH)

    # 实现参数平均聚合
    def aggregate(self, client_weights):
        global_weights = self.global_model.state_dict()
        for key in global_weights:
            global_weights[key] = torch.stack(
                [w[key].float() for w in client_weights], 0).mean(0)
        self.global_model.load_state_dict(global_weights)

    # 进行模型评估
    def evaluate(self):
        self.global_model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                loss = nn.functional.cross_entropy(outputs, labels)
                total_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += pred.eq(labels).sum().item()

        accuracy = 100 * correct / len(self.test_loader.dataset)
        avg_loss = total_loss / len(self.test_loader)
        self.history['accuracy'].append(accuracy)
        self.history['loss'].append(avg_loss)
        return accuracy, avg_loss

    def save_model(self, filename="fl_global_model.pt"):
        save_path = os.path.join(FLConfig.SAVE_PATH, filename)
        torch.save(self.global_model.state_dict(), save_path)
        print(f"模型已保存至 {save_path}")

# 训练流程控制
class FLTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_data()
        self._init_models()

    def _prepare_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST('./data', train=False, transform=transform)

        self.partitioner = DataPartitioner(
            train_set,
            FLConfig.NUM_CLIENTS,
            iid=FLConfig.IID,
            class_per_client=FLConfig.CLASS_PER_CLIENT
        )
        self.test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    def _init_models(self):
        self.global_model = Model_CNN().to(self.device)
        self.server = FLServer(self.global_model, self.test_loader, self.device)

    def train(self):
        print(f"启动联邦学习 | 设备: {self.device}")
        print(f"客户端数量: {FLConfig.NUM_CLIENTS} | 每轮选择: {int(FLConfig.NUM_CLIENTS * FLConfig.FRAC_CLIENTS)}")

        for round in range(1, FLConfig.NUM_ROUNDS + 1):
            # 客户端选择
            selected_clients = np.random.choice(
                FLConfig.NUM_CLIENTS,
                size=int(FLConfig.NUM_CLIENTS * FLConfig.FRAC_CLIENTS),
                replace=False
            )

            # 本地训练
            client_weights = []
            for client_id in selected_clients:
                # 模型初始化
                client_model = Model_CNN().to(self.device)
                client_model.load_state_dict(self.global_model.state_dict())

                # 获取本地数据
                train_data = self.partitioner.get_subset(client_id)

                # 训练并收集参数
                client = FLClient(client_model, train_data, self.device)
                weights = client.local_train()
                client_weights.append(weights)

            # 参数聚合
            self.server.aggregate(client_weights)

            # 全局评估
            accuracy, loss = self.server.evaluate()
            log_msg = (f"Round {round:02d} | "
                       f"Accuracy: {accuracy:.2f}% | "
                       f"Loss: {loss:.4f}")
            print(log_msg)

            # 保存日志
            with open(FLConfig.LOG_FILE, "a") as f:
                f.write(log_msg + "\n")

            # 每5轮保存模型
            if round % 5 == 0:
                self.server.save_model(f"round_{round}.pt")

        # 最终保存
        self.server.save_model("final_model.pt")
        print("训练完成！")

# 绘制图片
class TrainingVisualizer:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.rounds = []
        self.accuracies = []
        self.losses = []
        self.phase_changes = []

        self._validate_file()
        self._parse_log()

    def _validate_file(self):
        """验证日志文件是否存在且可读"""
        if not self.log_path.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_path}")
        if not self.log_path.is_file():
            raise ValueError(f"Path is not a file: {self.log_path}")

    def _parse_log(self):
        """解析训练日志文件"""
        pattern = r"Round (\d+).*Accuracy: (\d+\.\d+)%.*Loss: (\d+\.\d+)"
        absolute_round = 0
        prev_round = 0

        with open(self.log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                match = re.search(pattern, line)
                if match:
                    absolute_round += 1
                    current_round = int(match.group(1))

                    # 检测阶段变化
                    if current_round <= prev_round:
                        self.phase_changes.append(absolute_round)
                    prev_round = current_round

                    # 记录数据
                    self.rounds.append(absolute_round)
                    self.accuracies.append(float(match.group(2)))
                    self.losses.append(float(match.group(3)))

    def plot(self, show=True, figsize=(12, 6), annotate_every=1, offset=(0, 1)):
        """生成训练曲线图"""
        plt.style.use('seaborn-v0_8')
        self.fig, ax1 = plt.subplots(figsize=figsize)

        # 绘制准确率曲线
        ax1.set_xlabel('Training Round', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', color='tab:blue', fontsize=12)
        ax1.plot(self.rounds, self.accuracies,
                 color='tab:blue', marker='o', markersize=4,
                 linewidth=1.5, alpha=0.8, label='Accuracy', zorder=3)
        ax1.set_ylim(0, 100)
        ax1.grid(True, linestyle='--', alpha=0.5)

        for idx, (x, y) in enumerate(zip(self.rounds, self.accuracies)):
            if idx % annotate_every == 0:
                ax1.text(x + offset[0], y + offset[1],
                         f'{y:.2f}%',
                         color='tab:blue',
                         fontsize=8,
                         rotation=45,
                         ha='center',
                         va='bottom',
                         alpha=0.7,
                         zorder=5)  # 确保标注在最上层

        # 绘制损失曲线
        ax2 = ax1.twinx()
        ax2.set_ylabel('Loss', color='tab:red', fontsize=12)
        ax2.plot(self.rounds, self.losses,
                 color='tab:red', marker='x', markersize=4,
                 linewidth=1.5, alpha=0.8, label='Loss')
        ax2.set_ylim(0, max(self.losses) * 1.1)

        # 添加阶段分割线
        for pc in self.phase_changes:
            ax1.axvline(x=pc, color='gray', linestyle='--', alpha=0.5)
            ax1.text(pc, ax1.get_ylim()[0] + 5, f'Phase {self.phase_changes.index(pc) + 1}',
                     rotation=90, verticalalignment='bottom', fontsize=8)

        # 合并图例
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.title(f"Training Progress: {self.log_path.name}", fontsize=14, pad=20)
        plt.tight_layout()

        if show:
            plt.show()

    def save_plot(self, save_path, dpi=300):
        """保存图表到文件"""
        self.fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()

# 测试代码
if __name__ == "__main__":
    try:
        # 假设 FLConfig 是包含 LOG_FILE 路径的配置类
        visualizer = TrainingVisualizer(FLConfig.LOG_FILE)
        visualizer.plot()
        visualizer.save_plot("training_plot.png")
    except Exception as e:
        print(f"Error generating plot: {str(e)}")