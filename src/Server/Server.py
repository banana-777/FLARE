# Server.py

import queue
import time
import threading
import tkinter as tk
from ServerGUI import ServerGUI
from libs.ConnectionManager import ConnectionManager
from model.MNIST_CNN import Model_CNN


class Server:
    def __init__(self):
        self.gui = ServerGUI()
        self.model = Model_CNN()
        self.conn_mgr = ConnectionManager()

        self.training = False

        # 启动网络状态监控
        self._start_server()
        self._start_connection_monitor()

        # 设置回调
        self.gui.set_callbacks(self.start_training, self.stop_training)

    def _start_server(self):
        if self.conn_mgr.start_server():
            print(f"=== 服务器已启动在 {self.conn_mgr.host}:{self.conn_mgr.port} ===")
            self.gui.start_btn.config(state=tk.NORMAL)  # 启用开始按钮
        else:
            print("服务器启动失败")

    # 监控线程
    def _start_connection_monitor(self):
        # 监控及更新客户端连接数
        def monitor():
            while self.conn_mgr.running:
                try:
                    # 非阻塞获取数据
                    action, count = self.conn_mgr.status_queue.get_nowait()
                    self.gui.after(0, self._safe_update_client, count)
                except queue.Empty:
                    self.gui.after(100, monitor)  # 重新调度
                    break

        self.gui.after(100, monitor)  # 通过GUI事件循环调度

    # 安全更新客户端数量
    def _safe_update_client(self, count):
        self.gui.client_count.set(count)
        self.gui.update_idletasks()  # 强制立即刷新

    def start_training(self):
        def _real_training():
            print("=== 训练开始 ===")
            for i in range(1, 6):
                time.sleep(1)
                print(f"第 {i} 轮训练完成")
                self.gui.after(0, lambda: self.gui.round_label.config(text=f"当前轮次: {i - 1}"))

        train_thread = threading.Thread(target=_real_training)
        train_thread.daemon = True  # 设为守护线程
        train_thread.start()

    def stop_training(self):
        # 停止Socket服务器
        self.training = False
        self.gui.start_btn.config(state=tk.NORMAL)
        self.gui.stop_btn.config(state=tk.DISABLED)
        self.conn_mgr.stop_server()
        print("=== 训练停止 ===")

if __name__ == "__main__":
    server = Server()  # 使用Server类包装实例
    server.gui.mainloop()

