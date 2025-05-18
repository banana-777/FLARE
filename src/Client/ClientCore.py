# Client Core function class

import io
import sys
import time
from time import sleep

import torch
import hashlib
import pickle
import socket
import struct
import threading

import torch.nn as nn
import torch.optim as optim
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
                self.father.logger.log("成功连接到服务器","INFO")
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
            remote_addr = sock.getpeername()
            self.father.logger.log_comm_stats(f"接收结构 {remote_addr[0]}:{remote_addr[1]}",
                                              data_size=sys.getsizeof(serialized))
            sock.sendall("MODEL_STRUCTURE_RECEIVED".encode())
        # 反序列化
        self.father.model_arch = pickle.loads(serialized)
        self.father.model = self.build_model(self.father.model_arch)

    # 接收模型参数
    def recv_model_parameters(self, sock):
        header = self._recv_exact(sock, 4)
        data_len = struct.unpack('>I', header)[0]
        received_checksum = self._recv_exact(sock, 32)
        serialized = self._recv_exact(sock, data_len)
        calculated_checksum = hashlib.sha256(serialized).digest()
        if calculated_checksum != received_checksum:
            raise ValueError("参数校验失败: 数据可能被篡改")
        else:
            sock.sendall("MODEL_PARAMETERS_RECEIVED".encode())
            remote_addr = sock.getpeername()
            self.father.logger.log_comm_stats(f"接收参数 {remote_addr[0]}:{remote_addr[1]}",
                                              data_size=sys.getsizeof(serialized))
        buffer = io.BytesIO(serialized)
        state_dict = torch.load(buffer, map_location='cpu', weights_only=False)  # 强制加载到CPU
        # 加载到当前模型
        if hasattr(self.father, 'model'):
            self.father.model.load_state_dict(state_dict)
            print("模型参数加载成功")
        else:
            raise RuntimeError("未检测到模型结构，请先接收模型架构")

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
        self.father.server_socket.sendall(data)
        self.father.logger.log_comm_stats(f"接收结构 {remote_addr[0]}:{remote_addr[1]}",
                                          data_size=sys.getsizeof(data))

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
        start_time = time.time()
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
        for epoch in range(1, epochs):
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
                if (batch_idx + 1) % 16 == 0:
                    print(f"Epoch [{epoch}/{epochs}], "
                          f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}")
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            end_time = time.time()
            duration = end_time - start_time
            print(f"Epoch [{epoch}/{epochs}] 完成, "
                  f"平均损失: {avg_loss:.4f}, "
                  f"训练准确率: {accuracy:.2f}%")
            self.father.logger.log_train_stats(epoch = epoch, loss = avg_loss,
                                               accuracy = accuracy,duration = duration)
        print("=== 本地训练完成 ===")
        return self.father.model.state_dict()

    # 客户端参数压缩
    def stc_compress(self, state_dict, sparsity=0.01, gamma=0.001):
        compressed_dict = {}
        for name, param in state_dict.items():
            # 展平参数为1D张量
            flat_tensor = param.data.view(-1)

            # 计算阈值保留前k个最大元素
            k = max(1, int(sparsity * flat_tensor.numel()))
            values, indices = torch.topk(flat_tensor.abs(), k)
            threshold = values[-1]

            # 创建三元掩码
            mask = torch.zeros_like(flat_tensor)
            mask[indices] = 1
            positive_mask = (flat_tensor > threshold).float() * mask
            negative_mask = (flat_tensor < -threshold).float() * mask

            # 生成三元值
            compressed_tensor = gamma * (positive_mask - negative_mask)

            # 记录非零元素的索引和符号
            nonzero_indices = torch.nonzero(compressed_tensor).squeeze()
            signs = torch.sign(compressed_tensor[nonzero_indices])

            compressed_dict[name] = {
                'shape': param.shape,
                'indices': nonzero_indices.cpu().numpy(),
                'signs': signs.cpu().numpy(),
                'gamma': gamma
            }
        return compressed_dict

    # STC集成压缩打包
    def pack_model_parameters_2(self, state_dict: dict) -> bytes:
        compressed_dict = self.stc_compress(state_dict)  # 添加压缩步骤
        buffer = io.BytesIO()
        torch.save(compressed_dict, buffer, _use_new_zipfile_serialization=True, pickle_protocol=5)
        serialized = buffer.getvalue()
        checksum = hashlib.sha256(serialized).digest()
        header = struct.pack('>I', len(serialized))
        return header + checksum + serialized