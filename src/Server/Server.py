# Server.py
import time
import threading
import tkinter as tk
from ServerGUI import ServerGUI
from libs.ConnectionManager import ConnectionManager


class Server:
    def __init__(self):
        self.gui = ServerGUI()
        self.conn_mgr = ConnectionManager()
        # self.min_clients = 3  # 最小客户端数量要求
        self.training = False

        # 启动网络状态监控
        self._start_server()
        self._start_connection_monitor()

        # 设置回调
        self.gui.set_callbacks(self.start_training, self.stop_training)

    def _start_server(self):
        """启动服务器监听"""
        if self.conn_mgr.start_server():
            print(f"服务器已启动在 {self.conn_mgr.host}:{self.conn_mgr.port}")
            self.gui.start_btn.config(state=tk.NORMAL)  # 启用开始按钮
        else:
            print("服务器启动失败")

    def _start_connection_monitor(self):
        """启动连接状态监控线程"""

        def monitor():
            while True:
                if not self.conn_mgr.status_queue.empty():
                    action, count = self.conn_mgr.status_queue.get()
                    self.gui.after(0, self.gui.update_client_count, count)
                    print(f"客户端{action}，当前数量: {count}")
                time.sleep(0.1)

        threading.Thread(target=monitor, daemon=True).start()

    def start_training(self):
        # 启动Socket服务器
        self.conn_mgr.start_server()
        def _real_training():
            # print("=== 训练开始 ===")
            for i in range(1, 6):
                time.sleep(1)
                print(f"第 {i} 轮训练完成")
                # 更新GUI需使用after方法
                self.gui.after(0, lambda: self.gui.round_label.config(text=f"当前轮次: {i - 1}"))
            # 启动新线程执行耗时操作

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
    # app = ServerGUI()
    # app.set_callbacks(Server.start_training, Server.stop_training)
    # app.mainloop()
