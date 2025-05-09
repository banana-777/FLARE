# Client Core function class

import io
from time import sleep

import torch
import hashlib
import pickle
import socket
import struct
import threading

import torch.nn as nn
import torch.optim as optim
from fontTools.misc.timeTools import epoch_diff
from torch.utils.data import TensorDataset, DataLoader

class ClientCore:
    def __init__(self, FATHER_CLASS):
        self.TEST_STATUS = True
        self.father = FATHER_CLASS

    # 连接到服务器
    def connect_server(self, host, port):
        def _connect_server_thread():
            if not self.father.is_connected:
                self.father.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.father.server_socket.connect((host, port))
                self.father.is_connected = True
                self.father.gui.update_conn_status(True)
                if self.TEST_STATUS:
                    print("connect server success")
        threading.Thread(target=_connect_server_thread, daemon=True).start()

    # 消息接收与分发
    def msg_handler(self):
        sleep(1)
        while True:
            data = self.father.server_socket.recv(1024).decode('utf-8')
            # 接收模型参数
            if data == "SEND_MODEL_STRUCTURE":
                self.father.server_socket.sendall("READY_MODEL_STRUCTURE".encode())
                self.recv_model_structure(self.father.server_socket)
                continue
            if data == "SEND_MODEL_PARAMETERS":
                self.father.server_socket.sendall("READY_MODEL_PARAMETERS".encode())
                self.recv_model_parameters(self.father.server_socket)
                continue
            if data == "START_TRAIN":
                # 接收开始训练命令  发送响应    执行训练    发送参数
                print("接收到 START_TRAIN")
                self.father.server_socket.sendall("READY_START_TRAIN".encode())
                self.train_model(1)
                self.send_model_parameters()

    #接收模型结构
    def recv_model_structure(self, sock):
        header = self._recv_exact(sock, 4)
        data_len = struct.unpack('>I', header)[0]
        # 接收校验和（32字节）
        received_checksum = self._recv_exact(sock, 32)
        # 接收模型数据
        serialized = self._recv_exact(sock, data_len)
        # 验证校验和
        calculated_checksum = hashlib.sha256(serialized).digest()
        if calculated_checksum != received_checksum:
            raise ValueError("数据校验失败")
        else:
            print("recv model structure success")
            sock.sendall("MODEL_STRUCTURE_RECEIVED".encode())
        # 反序列化
        self.father.model_arch = pickle.loads(serialized)
        self.father.model = self.build_model(self.father.model_arch)
        # print(self.father.model)

    # 接收模型参数
    def recv_model_parameters(self, sock):
        header = self._recv_exact(sock, 4)
        data_len = struct.unpack('>I', header)[0]
        received_checksum = self._recv_exact(sock, 32)
        serialized = self._recv_exact(sock, data_len)
        # print(f"[CLIENT] 接收参数总长度: {4 + 32 + data_len} bytes")
        calculated_checksum = hashlib.sha256(serialized).digest()
        if calculated_checksum != received_checksum:
            raise ValueError("参数校验失败: 数据可能被篡改")
        else:
            print("recv model parameters success")
            sock.sendall("MODEL_PARAMETERS_RECEIVED".encode())
        buffer = io.BytesIO(serialized)
        state_dict = torch.load(buffer, map_location='cpu', weights_only=False)  # 强制加载到CPU
        # 加载到当前模型
        if hasattr(self.father, 'model'):
            self.father.model.load_state_dict(state_dict)
            print("模型参数加载成功")
        else:
            raise RuntimeError("未检测到模型结构，请先接收模型架构")
        # print("5秒后开始训练")
        # sleep(5)
        # self.test_train()
        # self.func()

    # 接收指定字节数
    def _recv_exact(self, sock, n: int) -> bytes:
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                raise ConnectionError("连接中断")
            data.extend(packet)
        return bytes(data)

    # 根据模型架构字典构建PyTorch模型
    def build_model(self, cfg: dict) -> nn.Module:
        class CustomModel(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.layers = nn.ModuleDict()
                conv_idx = 1
                linear_idx = 1

                for i, layer in enumerate(cfg['layers']):
                    layer_type = layer['class']
                    if layer_type == 'Conv2d':
                        name = f"conv{conv_idx}"
                        self.layers[name] = nn.Conv2d(
                            in_channels=layer['in_channels'],
                            out_channels=layer['out_channels'],
                            kernel_size=layer['kernel_size'],
                            padding=layer['padding']
                        )
                        self.layers[f"{name}_relu"] = nn.ReLU()
                        conv_idx += 1
                    elif layer_type == 'MaxPool2d':
                        name = f"pool{conv_idx - 1}"
                        self.layers[name] = nn.MaxPool2d(
                            kernel_size=layer['kernel_size'],
                            stride=layer['stride']
                        )
                    elif layer_type == 'Dropout':
                        self.layers["dropout"] = nn.Dropout(p=layer['p'])
                    elif layer_type == 'Linear':
                        name = f"fc{linear_idx}"
                        self.layers[name] = nn.Linear(
                            in_features=layer['in_features'],
                            out_features=layer['out_features']
                        )
                        if 'activation' in layer:
                            self.layers[f"{name}_relu"] = nn.ReLU()
                        linear_idx += 1

            def forward(self, x):
                for name, layer in self.layers.items():
                    if "_relu" in name:  # 激活层需要前一层输出
                        x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and x.dim() > 2:
                            x = x.view(x.size(0), -1)
                        x = layer(x)
                return x

        return CustomModel(cfg)

    # 发送训练好的模型参数
    def send_model_parameters(self):
        data = self.pack_model_parameters(self.father.model.state_dict())
        remote_addr = self.father.server_socket.getpeername()
        print(f"Send model parameters to {remote_addr[0]}:{remote_addr[1]}")
        # 协调准备
        # self.father.server_socket.sendall("SEND_MODEL_PARAMETERS".encode())
        # ack = self.father.server_socket.recv(1024).decode('utf-8')
        # if ack == "READY_MODEL_PARAMETERS":
        self.father.server_socket.sendall(data)
            # print(f"[SERVER] 发送参数总长度: {len(data)} bytes")
            # ack = self.father.server_socket.recv(1024).decode('utf-8')
            # if ack == "MODEL_PARAMETERS_RECEIVED":
            #     self.father.clients_status[self.father.server_socket] = "READY1"
            #     print(f"Send model parameters success")

    # 模型参数打包函数
    def pack_model_parameters(self, state_dict: dict) -> bytes:
        if not isinstance(state_dict, dict) or not all(isinstance(k, str) for k in state_dict.keys()):
            raise TypeError("输入必须为PyTorch state_dict格式")
        buffer = io.BytesIO()
        torch.save(state_dict, buffer, _use_new_zipfile_serialization=True, pickle_protocol=5)
        serialized = buffer.getvalue()
        checksum = hashlib.sha256(serialized).digest()
        header = struct.pack('>I', len(serialized))  # 大端4字节长度
        return header + checksum + serialized

    # 训练模型
    def train_model(self, epochs):
        if not all([self.father.model, self.father.train_data, self.father.test_data]):
            raise ValueError("模型或数据未正确初始化")

        train_dataset = TensorDataset(
            self.father.train_data['images'],
            self.father.train_data['labels'].long()  # 确保标签为int64类型
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=2
        )
        optimizer = optim.SGD(
            self.father.model.parameters(),
            lr=0.01,
            momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()
        self.father.model.train()
        print("\n=== 开始本地训练 ===")
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.father.model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch}/{epochs}], "
                          f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}")
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch}/{epochs}] 完成, "
                  f"平均损失: {avg_loss:.4f}, "
                  f"训练准确率: {accuracy:.2f}%")
        print("=== 本地训练完成 ===")
        return self.father.model.state_dict()

    # 测试训练流程
    def test_train(self):
        epochs = 10

        # ================= 0. 前置校验 =================
        if not all([self.father.model, self.father.train_data, self.father.test_data]):
            raise ValueError("模型或数据未正确初始化")

        # ================= 1. 准备数据 =================
        # 创建TensorDataset
        train_dataset = TensorDataset(
            self.father.train_data['images'],
            self.father.train_data['labels'].long()  # 确保标签为int64类型
        )

        # 创建DataLoader
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=2
        )

        # ================= 2. 训练配置 =================
        optimizer = optim.SGD(
            self.father.model.parameters(),
            lr=0.01,
            momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()

        # ================= 3. 训练循环 =================
        self.father.model.train()
        print("\n=== 开始本地训练 ===")
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()

                # 前向传播
                outputs = self.father.model(data)
                loss = criterion(outputs, target)

                # 反向传播
                loss.backward()
                optimizer.step()

                # 统计指标
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                # 打印批次日志
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch}/{epochs}], "
                          f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}")

            # 打印epoch统计
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch}/{epochs}] 完成, "
                  f"平均损失: {avg_loss:.4f}, "
                  f"训练准确率: {accuracy:.2f}%")

        # ================= 4. 返回更新参数 =================
        print("=== 本地训练完成 ===")
        return self.father.model.state_dict()


    def func(self):
        """本地测试模型性能"""
        # ================= 1. 准备数据 =================
        test_dataset = TensorDataset(
            self.father.test_data['images'],
            self.father.test_data['labels'].long()
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False
        )

        # ================= 2. 测试过程 =================
        self.father.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        print("\n=== 开始本地测试 ===")
        with torch.no_grad():
            for data, target in test_loader:
                outputs = self.father.model(data)
                loss = criterion(outputs, target)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        # ================= 3. 输出结果 =================
        avg_loss = total_loss / len(test_loader)
        accuracy = 100 * correct / total
        print(f"测试完成: 平均损失 {avg_loss:.4f}, "
              f"准确率 {accuracy:.2f}%")
        return accuracy