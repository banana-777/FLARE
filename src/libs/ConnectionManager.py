# Socket连接池管理 05 02 ShenJiaLong

import socket
import threading
import pickle
import zlib
import hashlib
import struct
from queue import Queue
from time import sleep
from model.MNIST_CNN import Model_CNN


class ConnectionManager:
    def __init__(self, host='0.0.0.0', port=8888):
        # 服务器状态
        self.host = host
        self.port = port
        self.clients_status = {}
        self.status_queue = Queue()
        self.running = False

        # 客户端状态
        self.is_connected = False
        # 服务器套接字
        self.server_socket = None

        # 初始化模型
        self.model = Model_CNN()

    def start_server(self):
        try:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind((self.host, self.port))
            self.server.listen(5)
            self.running = True

            # 先返回成功再启动线程
            accept_thread = threading.Thread(target=self._accept_connections)
            accept_thread.daemon = True
            accept_thread.start()

            return True  # 确保在最后返回
        except Exception as e:
            print(f"服务器启动失败: {str(e)}")
            return False

    def stop_server(self):
        self.running = False
        self.server.close()

    # 接受客户端连接
    def _accept_connections(self):
        while self.running:
            try:
                client, addr = self.server.accept()
                print(f"客户端连接: {addr[0]}:{addr[1]}")
                self.clients_status.setdefault(client, "CONNECTED")
                self.status_queue.put(('connect', len(self.clients_status)))
                threading.Thread(target=self._handle_client, args=(client, )).start()
            except:
                break

    # 处理客户端连接
    def _handle_client(self, client):
        remote_addr = client.getpeername()
        # 模型结构
        model_arch = {
            'type': 'CNN',
            'input_channels': 1,  # MNIST是单通道图像
            'layers': [
                # 第一卷积块
                {
                    'class': 'Conv2d',
                    'in_channels': 1,
                    'out_channels': 32,
                    'kernel_size': 3,
                    'padding': 1,
                    'activation': 'ReLU'
                },
                {
                    'class': 'MaxPool2d',
                    'kernel_size': 2,
                    'stride': 2
                },

                # 第二卷积块
                {
                    'class': 'Conv2d',
                    'in_channels': 32,
                    'out_channels': 64,
                    'kernel_size': 3,
                    'padding': 1,
                    'activation': 'ReLU'
                },
                {
                    'class': 'MaxPool2d',
                    'kernel_size': 2,
                    'stride': 2
                },

                # 正则化层
                {
                    'class': 'Dropout',
                    'p': 0.5
                },

                # 全连接层
                {
                    'class': 'Linear',
                    'in_features': 7 * 7 * 64,  # 经过两次池化后的尺寸: 28→14→7
                    'out_features': 128,
                    'activation': 'ReLU'
                },
                {
                    'class': 'Linear',
                    'in_features': 128,
                    'out_features': 10
                }
            ]
        }
        # 向client发送模型结构
        self.send_model_structure(model_arch, client)
        # 等待客户端响应
        ack = client.recv(1024).decode('utf-8')
        if ack == "MODEL_STRUCTURE_RECEIVED":
            print(f"客户端接收模型结构成功 {remote_addr[0]}:{remote_addr[1]}")
            # 向client发送初始参数
            initial_params = self.model.state_dict()
            self.send_model_parameters(initial_params, client)
            # 等待客户端响应
            ack = client.recv(1024).decode('utf-8')
            if ack == "MODEL_PARAMETERS_RECEIVED":
                print(f"客户端接收模型参数成功 {remote_addr[0]}:{remote_addr[1]}")
                self.clients_status[client] = "READY"


        # 等待客户端确认
        # ack = self.receive_data(client)

    # 客户端连接服务器
    def connect_to_server(self, server_host, server_port):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.connect((server_host, server_port))
            self.server_socket.settimeout(15)  # 超时时间
            self.is_connected = True
            return True
        except socket.timeout:
            print("连接超时，请检查服务器状态")
            return False
        except ConnectionRefusedError:
            print("连接被拒绝，服务器未启动")
            return False
        except Exception as e:
            print(f"连接异常: {str(e)}")
            return False

    def send_data(self, data, mysocket):
        if isinstance(data, str):
            data = data.encode('utf-8')
            return mysocket.sendall(data)
        return self._send_binary(data, mysocket)

    def receive_data(self, mysocket):
        data = self._receive_binary(mysocket)
        return data.decode('utf-8') if data else None

    # 发送二进制数据
    def _send_binary(self, data, mysocket):
        try:
            length = len(data).to_bytes(4, byteorder='big')
            mysocket.sendall(length + data)
            return True
        except Exception as e:
            print(f"二进制发送失败: {str(e)}")
            self.is_connected = False
            return False

    # 接收二进制数据
    def _receive_binary(self, mysocket, expected_length=None):
        if not self.is_connected:
            return None
        try:
            # 读取长度头
            if expected_length is None:
                length_bytes = self._receive_all(4, mysocket)
                if not length_bytes:
                    return None
                expected_length = int.from_bytes(length_bytes, byteorder='big')

            # 接收主体数据
            return self._receive_all(expected_length, mysocket)
        except Exception as e:
            print(f"二进制接收失败: {str(e)}")
            self.is_connected = False
            return None

    # 接收指定长度的数据
    def _receive_all(self, n, mysocket):
        data = bytearray()
        while len(data) < n:
            packet = mysocket.recv(n - len(data))
            print(f"packet: {packet}")
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)

    # 发送模型结构
    def send_model_structure(self, model_arch, mysocket):
        try:
            serialized = pickle.dumps(model_arch)
            checksum = hashlib.md5(serialized).digest()
            data = checksum + serialized
            remote_addr = mysocket.getpeername()
            print(f"向客户端发送模型结构 {remote_addr[0]}:{remote_addr[1]}")
            return self._send_binary(data, mysocket)
        except Exception as e:
            print(f"发送模型结构失败: {str(e)}")
            return False

    # 发送模型参数
    def send_model_parameters(self, params, mysocket, compress=True):
        try:
            serialized = pickle.dumps(params)
            if compress:
                serialized = zlib.compress(serialized)
            # 添加头部信息（压缩标志 + 数据长度）
            header = struct.pack('!?I', compress, len(serialized))
            return self._send_binary(header + serialized, mysocket)
        except Exception as e:
            print(f"发送模型参数失败: {str(e)}")
            return False

    # 接收并验证模型结构
    def receive_model_structure(self, mysocket):
        try:
            remote_addr = mysocket.getpeername()
            print(f"接收模型结构 {remote_addr[0]}:{remote_addr[1]}")
            data = self._receive_binary(mysocket)
            if not data:
                return None
            # 验证校验码
            received_checksum = data[:16]
            actual_checksum = hashlib.md5(data[16:]).digest()
            if received_checksum != actual_checksum:
                print("模型结构校验失败!")
                return None
            return pickle.loads(data[16:])
        except Exception as e:
            print(f"接收模型结构失败: {str(e)}")
            return None

    # 接收模型参数
    def receive_model_parameters(self, mysocket):
        try:
            header = self._receive_binary(mysocket, 5)  # 1字节压缩标志 + 4字节长度
            if not header:
                return None
            compress_flag = struct.unpack('!?', header[:1])[0]
            data_len = struct.unpack('!I', header[1:5])[0]
            # 接收主体数据
            data = self._receive_binary(mysocket, data_len)
            if not data:
                return None
            # 解压缩处理
            if compress_flag:
                data = zlib.decompress(data)
            return pickle.loads(data)
        except Exception as e:
            print(f"接收模型参数失败: {str(e)}")
            return None