# Socket连接池管理 05 02 ShenJiaLong

import socket
import threading
from queue import Queue


class ConnectionManager:
    def __init__(self, host='0.0.0.0', port=8888):
        # 服务器状态
        self.host = host
        self.port = port
        self.clients = set()
        self.status_queue = Queue()
        self.running = False

        # 客户端状态
        self.is_connected = False
        self.client_socket = None

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
        """停止Socket服务器"""
        self.running = False
        self.server.close()

    def _accept_connections(self):
        """接受客户端连接"""
        while self.running:
            try:
                client, addr = self.server.accept()
                self.clients.add(client)
                self.status_queue.put(('connect', len(self.clients)))
                threading.Thread(target=self._handle_client, args=(client,)).start()
            except:
                break

    def _handle_client(self, client):
        """处理客户端连接"""
        with client:
            while self.running:
                try:
                    data = client.recv(1024)
                    if not data:
                        break
                except:
                    break
            self.clients.remove(client)
            self.status_queue.put(('disconnect', len(self.clients)))

    # 客户端连接服务器方法
    def connect_to_server(self, server_host, server_port):
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((server_host, server_port))
            self.client_socket.settimeout(2)  # 超时时间
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

    # 发送数据到服务器
    def send_data(self, data):
        if self.is_connected:
            try:
                self.client_socket.sendall(data.encode())
                return True
            except:
                self.is_connected = False
                return False
        return False

    # 接收服务器数据
    def receive_data(self):
        if self.is_connected:
            try:
                return self.client_socket.recv(1024).decode()
            except:
                self.is_connected = False
                return None
        return None


