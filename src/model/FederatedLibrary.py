# 04 29
# 训练使用到的库及函数

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from MNIST_CNN import Model_CNN

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