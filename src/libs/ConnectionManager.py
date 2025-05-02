# Socket连接池管理 05 02 ShenJiaLong

import socket
import threading
from queue import Queue


class ConnectionManager:
    def __init__(self, host='0.0.0.0', port=8888):
        self.host = host
        self.port = port
        self.clients = set()
        self.status_queue = Queue()
        self.running = False

    def start_server(self):
        """启动Socket服务器"""
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(5)

        self.running = True
        accept_thread = threading.Thread(target=self._accept_connections)
        accept_thread.daemon = True
        accept_thread.start()

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
