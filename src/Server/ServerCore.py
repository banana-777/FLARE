# Client Core function class

import io
import socket
import threading
import pickle
from time import sleep

import torch
import zlib
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

    # 处理连接
    def wait_connection(self):
        while True:
            client_socket, addr = self.father.server_socket.accept()
            self.father.clients_status[client_socket] = "CONNECTED"
            print(f"Receive connection from {addr}")
            self.father.gui.update_client_count(len(self.father.clients_status))

            # 为每个客户端创建新线程
            client_thread = threading.Thread(
                target=self.handle_client,
                args=(client_socket, addr)
            )
            client_thread.start()

    # 客户端处理函数
    def handle_client(self, client_socket, addr):
        sleep(2)
        with client_socket:
            while True:
                # 刚完成连接
                if self.father.clients_status[client_socket] == "CONNECTED":
                    self.func_CONNECTED(client_socket, addr)
                    sleep(4)
                # 接收完模型结构
                if self.father.clients_status[client_socket] == "READY0":
                    self.func_READY0(client_socket, addr)
                # 接受完模型参数
                if self.father.clients_status[client_socket] == "READY1":
                    self.func_READY1(client_socket, addr)
                # 训练中
                if self.father.clients_status[client_socket] == "TRAINING":
                    self.func_TRAINING(client_socket, addr)

    # CONNECTED状态 发送模型结构
    def func_CONNECTED(self, client_socket, addr):
        data = self.pack_model_data(self.father.model_arch)
        remote_addr = client_socket.getpeername()
        print(f"Send model structure to {remote_addr[0]}:{remote_addr[1]}")
        # 协调准备
        client_socket.sendall("SEND_MODEL_STRUCTURE".encode())
        ack = client_socket.recv(1024).decode('utf-8')
        if ack == "READY_MODEL_STRUCTURE":
            client_socket.sendall(data)
            ack = client_socket.recv(1024).decode('utf-8')
            if ack == "MODEL_STRUCTURE_RECEIVED":
                self.father.clients_status[client_socket] = "READY0"
                print(f"Send model structure success")

    # READY0状态 发送模型参数
    def func_READY0(self, client_socket, addr):
        data = self.pack_model_parameters(self.father.model.state_dict())
        remote_addr = client_socket.getpeername()
        print(f"Send model parameters to {remote_addr[0]}:{remote_addr[1]}")
        # 协调准备
        client_socket.sendall("SEND_MODEL_PARAMETERS".encode())
        ack = client_socket.recv(1024).decode('utf-8')
        if ack == "READY_MODEL_PARAMETERS":
            client_socket.sendall(data)
            print(f"[SERVER] 发送参数总长度: {len(data)} bytes")
            ack = client_socket.recv(1024).decode('utf-8')
            if ack == "MODEL_PARAMETERS_RECEIVED":
                self.father.clients_status[client_socket] = "READY1"
                print(f"Send model parameters success")

    # READY1状态 协调进入TRAINING状态
    def func_READY1(self, client_socket, addr):
        pass

    # TRAINING状态 进行参数迭代
    def func_TRAINING(self, client_socket, addr):
        pass

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
