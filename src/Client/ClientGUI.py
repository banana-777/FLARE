# Client GUI class

import sys
import multiprocessing
import tkinter as tk
from pathlib import Path
from time import sleep
from tkinter import ttk, filedialog

import torch


class ClientGUI(tk.Tk):
    def __init__(self, FATHER_CLASS):
        super().__init__()
        self.father = FATHER_CLASS
        self.title("联邦学习客户端")
        self.geometry("600x400")
        # 初始化数据路径
        self.data_path = tk.StringVar()
        self.test_data = None
        self.train_data = None
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

        self.disconnect_btn = ttk.Button(
            btn_container,
            text="断开连接",
            width=15,
            state=tk.DISABLED,
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
            # 构建文件路径
            train_path = Path(path) / f"client_train.pt"
            test_path = Path(path) / f"client_test.pt"
            # 验证文件存在性
            if not (train_path.exists() and test_path.exists()):
                raise FileNotFoundError("缺失必要的训练/测试文件")

            # 加载数据
            self.father.train_data = torch.load(str(train_path))
            self.father.test_data = torch.load(str(test_path))
            print(f"已选择并加载数据: {path}")

    # 更新连接状态显示
    def update_conn_status(self, is_connected):
        def update_conn_status_thread():
            if is_connected:
                self.conn_status.config(text="已连接", foreground='green')
            else:
                self.conn_status.config(text="连接失败", foreground='red')
        self.after(10, update_conn_status_thread)

def start_process(process_id):
    print(f"进程 {process_id} 开始工作")
    client = ClientGUI()
    client.mainloop()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    # 启动n个客户端进程
    n = 5
    processes = []
    for i in range(n):
        p = multiprocessing.Process(target=start_process, args=(i,))
        processes.append(p)
        p.start()
        sleep(1)
    # 监控进程状态
    while any(p.is_alive() for p in processes):
        sleep(1)

    print("所有进程执行完毕")
