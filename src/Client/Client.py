# 客户端主入口 05 02 ShenJiaLong

import threading
import sys
import tkinter as tk
from time import sleep
from tkinter import ttk, filedialog
from libs.ConnectionManager import ConnectionManager

class ClientGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("联邦学习客户端")
        self.geometry("600x400")
        # 初始化数据路径
        self.data_path = tk.StringVar()
        # 创建主容器
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        # 创建各功能区域
        self._create_control_panel()
        self._create_data_panel()
        self._create_status_panel()
        self._create_log_panel()
        # 重定向标准输出
        self._redirect_stdout()

    # 顶部控制按钮区域
    def _create_control_panel(self):
        control_frame = ttk.LabelFrame(self.main_frame, text="控制面板")
        control_frame.pack(fill='x', pady=5)

        btn_container = ttk.Frame(control_frame)
        btn_container.pack(pady=5)

        self.connect_btn = ttk.Button(
            btn_container,
            text="连接服务器",
            width=15,
            command=self._handle_connect
        )
        self.connect_btn.pack(side='left', padx=5)

        self.train_btn = ttk.Button(
            btn_container,
            text="开始训练",
            width=15,
            state=tk.DISABLED,
            command=self._handle_train
        )
        self.train_btn.pack(side='left', padx=5)

        self.disconnect_btn = ttk.Button(
            btn_container,
            text="断开连接",
            width=15,
            state=tk.DISABLED,
            command=self._handle_disconnect
        )
        self.disconnect_btn.pack(side='left', padx=5)

    # 数据选择区域
    def _create_data_panel(self):
        data_frame = ttk.LabelFrame(self.main_frame, text="训练数据选择")
        data_frame.pack(fill='x', pady=5)

        # 路径选择组件
        path_container = ttk.Frame(data_frame)
        path_container.pack(fill='x', pady=3)

        ttk.Button(
            path_container,
            text="选择数据",
            width=10,
            command=self._select_data
        ).pack(side='left')

        self.path_entry = ttk.Entry(
            path_container,
            textvariable=self.data_path,
            state='readonly',
            width=50
        )
        self.path_entry.pack(side='left', padx=5, fill='x', expand=True)

        # 数据类型选择
        self.data_type = tk.StringVar(value='dir')
        type_container = ttk.Frame(data_frame)
        type_container.pack(fill='x', pady=3)

        ttk.Radiobutton(
            type_container,
            text="文件夹",
            variable=self.data_type,
            value='dir'
        ).pack(side='left')

        ttk.Radiobutton(
            type_container,
            text="文件",
            variable=self.data_type,
            value='file'
        ).pack(side='left', padx=10)

    # 状态显示区域
    def _create_status_panel(self):
        status_frame = ttk.LabelFrame(self.main_frame, text="训练状态")
        status_frame.pack(fill='x', pady=5)

        # 状态信息
        ttk.Label(status_frame, text="连接状态:").pack(side='left')
        self.conn_status = ttk.Label(status_frame, text="未连接", foreground='red')
        self.conn_status.pack(side='left', padx=10)

        ttk.Label(status_frame, text="训练进度:").pack(side='left')
        self.progress = ttk.Progressbar(status_frame, mode='determinate')
        self.progress.pack(side='left', padx=5, fill='x', expand=True)

    # 日志输出区域
    def _create_log_panel(self):
        log_frame = ttk.LabelFrame(self.main_frame, text="系统日志")
        log_frame.pack(fill='both', expand=True, pady=5)

        self.log_text = tk.Text(
            log_frame,
            wrap='word',
            state='disabled',
            font=('Consolas', 10)
        )
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    # 重定向标准输出到日志框
    def _redirect_stdout(self):
        class RedirectOutput:
            def __init__(self, text_widget):
                self.text_widget = text_widget

            def write(self, string):
                self.text_widget.configure(state='normal')
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)
                self.text_widget.configure(state='disabled')

            def flush(self):
                pass

        sys.stdout = RedirectOutput(self.log_text)
        sys.stderr = RedirectOutput(self.log_text)

    # 处理数据选择
    def _select_data(self):
        if self.data_type.get() == 'dir':
            path = filedialog.askdirectory(title="选择训练数据文件夹")
        else:
            path = filedialog.askopenfilename(title="选择训练数据文件")

        if path:
            self.data_path.set(path)
            self.train_btn.config(state=tk.NORMAL)
            print(f"已选择数据路径: {path}")

    # 连接按钮点击
    def _handle_connect(self):
        # 子线程承担实际操作
        def _connect_thread():
            if self.conn_mgr.connect_to_server(self.server_host, self.server_port):
                self.after(0, self.update_conn_status, True)
                print("成功连接到服务器")
            else:
                self.after(0, self.update_conn_status, False)

        # 禁用按钮防止重复点击
        self.connect_btn.config(state=tk.DISABLED)
        threading.Thread(target=_connect_thread, daemon=True).start()

    # 更新连接状态显示
    def update_conn_status(self, is_connected):
        if is_connected:
            self.conn_status.config(text="已连接", foreground='green')
            self.train_btn.config(state=tk.NORMAL)
        else:
            self.conn_status.config(text="连接失败", foreground='red')
        self.connect_btn.config(state=tk.NORMAL)  # 恢复按钮状态

    # 训练按钮点击
    def _handle_train(self):
        print(f"开始使用 {self.data_path.get()} 数据进行训练")
        self._simulate_training()

    # 断开连接处理
    def _handle_disconnect(self):
        if self.conn_mgr.client_socket:
            self.conn_mgr.client_socket.close()
        self.conn_status.config(text="未连接", foreground='red')
        self.train_btn.config(state=tk.DISABLED)
        print("已断开服务器连接")

    # 模拟训练进度
    def _simulate_training(self):
        def update_progress(progress):
            self.progress['value'] = progress
            self.update()

        for i in range(1, 101):
            self.after(50, update_progress, i)

class Client:
    def __init__(self):
        self.gui = ClientGUI()
        self.conn_mgr = ConnectionManager()
        self.training = False
        self.server_host = '127.0.0.1'
        self.server_port = 8888
        # 连接到服务器
        self.connect_server()

    # 连接服务器函数
    def connect_server(self):
        def _connect_thread():
            if self.conn_mgr.connect_to_server(self.server_host, self.server_port):
                self.gui.update_conn_status(True)
                print("成功连接到服务器")
            else:
                self.gui.update_conn_status(False)

        threading.Thread(target=_connect_thread, daemon=True).start()

    def start_training(self, data_path):
        def _training_thread():
            self.training = True
            try:
                # 模拟训练过程
                for epoch in range(1, 6):
                    if not self.training: break
                    self.gui.update_progress(epoch * 20)
                    self._send_model_update()
                    sleep(1)
                self.gui.log("本地训练完成")
            finally:
                self.training = False

        if self.conn_mgr.is_connected():
            threading.Thread(target=_training_thread, daemon=True).start()
        else:
            self.gui.log("错误：未连接服务器")

    def _send_model_update(self):
        # 此处添加实际模型传输逻辑
        self.conn_mgr.send_data(b"model_update")

    def disconnect_server(self):
        if self.conn_mgr.client_socket:
            self.conn_mgr.client_socket.close()
        self.training = False
        self.gui.update_connection_status(False)
        self.gui.log("已断开服务器连接")

if __name__ == "__main__":
    # 初始化并启动客户端
    client = Client()
    client.gui.mainloop()