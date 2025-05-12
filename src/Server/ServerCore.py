# Client Core function class

import io
import socket
import sys
import threading
import pickle
from time import sleep
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import struct
import hashlib
import tkinter as tk


class ServerCore:
    def __init__(self, FATHER_CLASS):
        self.father = FATHER_CLASS

    def start_server(self):
        # 创建监听
        self.father.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.father.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.father.server_socket.bind((self.father.host, self.father.port))
        self.father.server_socket.listen(5)
        self.father.is_running = True

        print(f"Server started on {self.father.host}:{self.father.port}")
        self.father.gui.start_btn.config(state=tk.NORMAL)
        self.father.logger.log("服务器初始化完成","INFO")

    # 处理连接
    def wait_connection(self):
        while True:
            client_socket, addr = self.father.server_socket.accept()
            self.father.clients_status[client_socket] = "CONNECTED"
            print(f"Receive connection from {addr}")
            self.father.logger.log(f"连接到 {addr}", "COMM")
            self.father.gui.update_client_count(len(self.father.clients_status))

            # 为每个客户端创建新线程
            client_thread = threading.Thread(
                target=self.handle_client,
                args=(client_socket, addr)
            )
            client_thread.start()

    # 客户端处理函数
    def handle_client(self, client_socket, addr):
        sleep(1)
        with client_socket:
            while True:
                # 刚完成连接
                if self.father.clients_status[client_socket] == "CONNECTED":
                    self.func_CONNECTED(client_socket, addr)
                # 接收完模型结构
                if self.father.clients_status[client_socket] == "READY0":
                    self.func_READY0(client_socket, addr)
                # 接受完模型参数
                if self.father.clients_status[client_socket] == "READY1" and self.father.is_started is True:
                    self.func_READY1(client_socket)
                sleep(0.5)

    # CONNECTED状态 发送模型结构
    def func_CONNECTED(self, client_socket, addr):
        data = self.pack_model_data(self.father.model_arch)
        remote_addr = client_socket.getpeername()
        # 协调准备
        client_socket.sendall("SEND_MODEL_STRUCTURE".encode())
        ack = client_socket.recv(1024).decode('utf-8')
        if ack == "READY_MODEL_STRUCTURE":
            client_socket.sendall(data)
            ack = client_socket.recv(1024).decode('utf-8')
            self.father.logger.log_comm_stats(f"结构上传 {remote_addr[0]}:{remote_addr[1]}",
                                              data_size=sys.getsizeof(data))
            if ack == "MODEL_STRUCTURE_RECEIVED":
                self.father.clients_status[client_socket] = "READY0"

    # READY0状态 发送模型参数
    def func_READY0(self, client_socket, addr):
        data = self.pack_model_parameters(self.father.model.state_dict())
        remote_addr = client_socket.getpeername()
        # 协调准备
        client_socket.sendall("SEND_MODEL_PARAMETERS".encode())
        ack = client_socket.recv(1024).decode('utf-8')
        if ack == "READY_MODEL_PARAMETERS":
            client_socket.sendall(data)
            self.father.logger.log_comm_stats(f"参数上传 {remote_addr[0]}:{remote_addr[1]}",
                                              data_size=sys.getsizeof(data))
            ack = client_socket.recv(1024).decode('utf-8')
            if ack == "MODEL_PARAMETERS_RECEIVED":
                self.father.clients_status[client_socket] = "READY1"

    # READY1状态 协调进入TRAINING状态
    def func_READY1(self, client_socket):
        # 发送START_TRAIN命令   接收响应    接收模型参数  参数写入数组  将状态置为TRAINING
        client_socket.sendall("START_TRAIN".encode())
        ack = client_socket.recv(1024).decode('utf-8')
        if ack == "READY_START_TRAIN":
            new_dict = self.recv_model_parameters(client_socket)
            remote_addr = client_socket.getpeername()
            self.father.logger.log_comm_stats(f"参数接收 {remote_addr[0]}:{remote_addr[1]}",
                                              data_size=sys.getsizeof(new_dict))
            self.father.model_data.append(new_dict)
            self.father.clients_status[client_socket] = "TRAINING"

    # 协调训练的函数
    def func_training(self, rounds):
        round = 1
        while True:
            flag = 0
            for value in self.father.clients_status.values():
                if value != "TRAINING":
                    flag += 1
            if flag == 0 and self.father.is_started is True and round <= rounds:
                # 遍历数组  执行聚合    清空数组    写入model 状态都置为READY1
                print(f"===  第 {round} 轮  ===")
                self.father.gui.after(0, lambda: self.father.gui.round_label.config(text=f"当前轮次: {round}"))
                selected_state_dict = self.krum_aggregate(self.father.model_data, 1)
                self.father.model_data.clear()
                self.father.model.load_state_dict(selected_state_dict)
                for key in  self.father.clients_status.keys():
                    self.father.clients_status[key] = "READY1"
                self.test_model(round)
                round += 1

    # 模型结构打包函数
    def pack_model_data(self, model_arch: dict) -> bytes:
        serialized = pickle.dumps(model_arch)
        checksum = hashlib.sha256(serialized).digest()
        data_len = len(serialized)
        header = struct.pack('>I', data_len)  # 4字节大端无符号整型
        return header + checksum + serialized

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
            print("recv model parameters success")
        buffer = io.BytesIO(serialized)
        state_dict = torch.load(buffer, map_location='cpu', weights_only=False)  # 强制加载到CPU
        return state_dict

    # 接收指定字节数
    def _recv_exact(self, sock, n: int) -> bytes:
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                raise ConnectionError("连接中断")
            data.extend(packet)
        return bytes(data)

    # krum聚合算法
    def krum_aggregate(self, client_params, f=1):
        # 步骤1：参数向量化
        param_vectors = []
        for params in client_params:
            # 将各层参数展平并拼接成单个向量
            vec = []
            for name, param in params.items():
                vec.append(param.view(-1).cpu().numpy())
            param_vectors.append(np.concatenate(vec))
        param_vectors = np.array(param_vectors)  # (num_clients, dim)

        # 步骤2：计算距离矩阵
        n = len(client_params)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i, j] = np.linalg.norm(param_vectors[i] - param_vectors[j])

        # 步骤3：寻找最优参数
        scores = []
        for i in range(n):
            # 对每个客户端找出最近的n-f-2个距离
            distances = np.sort(distance_matrix[i])
            selected = distances[:n - f - 2]
            scores.append(selected.sum())

        # 步骤4：选择分数最小的参数
        selected_idx = np.argmin(scores)

        # 步骤5：还原参数结构
        selected_params = client_params[selected_idx]

        return selected_params

    # 测试模型
    def test_model(self, round):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))  # MNIST 标准归一化参数
        ])
        test_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        accuracy, loss = self.evaluate_accuracy( self.father.model, test_loader, device)
        print(f'测试准确率: {accuracy:.2f}%')
        self.father.logger.log(f"第 {round} 轮 准确率:{accuracy} 损失:{loss}", "TEST")

    # 计算正确率
    def evaluate_accuracy(self, model, test_loader, device='cpu', total_loss=0.0):
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                # 计算损失
                loss = nn.CrossEntropyLoss()(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        avg_loss = total_loss / total  # 计算平均损失
        return accuracy, avg_loss

    # STC解压
    def stc_decompress(self, compressed_dict):
        state_dict = {}
        for name, meta in compressed_dict.items():
            # 创建全零张量
            tensor = torch.zeros(meta['shape'].numel(), dtype=torch.float32)

            # 恢复三元值
            tensor[meta['indices']] = meta['gamma'] * torch.from_numpy(meta['signs'])

            state_dict[name] = tensor.view(*meta['shape'])
        return state_dict