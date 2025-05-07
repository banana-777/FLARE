# Socket连接池管理 05 02 ShenJiaLong

import pickle
import zlib
import hashlib
import struct
from model.MNIST_CNN import Model_CNN


class ConnectionManager:
    def __init__(self, host='0.0.0.0', port=8888):
        # 客户端状态
        self.is_connected = False
        # 服务器套接字
        self.server_socket = None
        # 初始化模型
        self.model = Model_CNN()

    def stop_server(self):
        self.running = False
        self.server.close()

    # 处理客户端连接
    def _handle_client(self, client):

        # 模型结构
        pass



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