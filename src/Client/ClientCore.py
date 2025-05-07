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
        # try:
        header = self._recv_exact(sock, 4)
        data_len = struct.unpack('>I', header)[0]
        received_checksum = self._recv_exact(sock, 32)
        serialized = self._recv_exact(sock, data_len)
        print(f"[CLIENT] 接收参数总长度: {4 + 32 + data_len} bytes")
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
        #
        # except Exception as e:
        #     print(f"参数接收失败: {str(e)}")
        #     sock.sendall(f"PARAM_ERROR:{str(e)}".encode())

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
    # def build_model(self, cfg: dict) -> nn.Module:
    #     model = nn.ModuleDict()  # 改用ModuleDict显式命名
    #     layers = []
    #
    #     conv_idx = 1
    #     linear_idx = 1
    #
    #     for i, layer in enumerate(cfg['layers']):
    #         layer_type = layer['class']
    #         if layer_type == 'Conv2d':
    #             name = f"conv{conv_idx}"
    #             model[name] = nn.Conv2d(
    #                 in_channels=layer['in_channels'],
    #                 out_channels=layer['out_channels'],
    #                 kernel_size=layer['kernel_size'],
    #                 padding=layer['padding']
    #             )
    #             layers.append(model[name])
    #             layers.append(nn.ReLU())
    #             conv_idx += 1
    #         elif layer_type == 'MaxPool2d':
    #             name = f"pool{conv_idx - 1}"
    #             model[name] = nn.MaxPool2d(
    #                 kernel_size=layer['kernel_size'],
    #                 stride=layer['stride']
    #             )
    #             layers.append(model[name])
    #         elif layer_type == 'Dropout':
    #             name = "dropout"
    #             model[name] = nn.Dropout(p=layer['p'])
    #             layers.append(model[name])
    #         elif layer_type == 'Linear':
    #             name = f"fc{linear_idx}"
    #             model[name] = nn.Linear(
    #                 in_features=layer['in_features'],
    #                 out_features=layer['out_features']
    #             )
    #             layers.append(model[name])
    #             if 'activation' in layer:
    #                 layers.append(nn.ReLU())
    #             linear_idx += 1
    #
    #     # 最后包装成Sequential
    #     return nn.Sequential(*layers)

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
