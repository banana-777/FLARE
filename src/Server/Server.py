# Server.py

import queue
import time
import threading
import tkinter as tk
from queue import Queue
from ServerGUI import ServerGUI
from ServerCore import ServerCore
from libs.ConnectionManager import ConnectionManager
from model.MNIST_CNN import Model_CNN


class Server:
    def __init__(self):
        self.gui = ServerGUI()
        self.core = ServerCore(self)
        self.model = Model_CNN()
        self.conn_mgr = ConnectionManager()
        self.model_arch = {
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
        self.model_data = []

        self.is_training = False
        self.is_running = False
        self.is_started = False # 是否开始训练
        self.host = '0.0.0.0'
        self.port = 8888
        self.rounds = 10
        self.server_socket = None
        self.clients_status = {}
        self.status_queue = Queue()
        '''
        clients_status : key-client socket value-status
        CONNECTED : 刚完成连接
        READY0 : 接收完模型结构
        READY1 : 接受完模型参数
        TRAINING : 训练中
        DEAD : 失去连接
        '''

        # 设置回调
        self.gui.set_callbacks(self.start_training, self.stop_training)

        # 启动网络状态监控
        self.core.start_server()
        threading.Thread(target=self.core.wait_connection, daemon=True).start()
        threading.Thread(target=self.core.func_training, args = (self.rounds, ), daemon=True).start()

    # 安全更新客户端数量
    def _safe_update_client(self, count):
        self.gui.client_count.set(count)
        self.gui.update_idletasks()  # 强制立即刷新

    def start_training(self):
        self.is_started = True
        print("===  开始训练  ===")

    def stop_training(self):
        # 停止Socket服务器
        self.training = False
        self.gui.start_btn.config(state=tk.NORMAL)
        self.gui.stop_btn.config(state=tk.DISABLED)
        self.conn_mgr.stop_server()
        print("=== 训练停止 ===")

if __name__ == "__main__":
    server = Server()
    server.gui.mainloop()

