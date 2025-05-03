import tkinter as tk
from tkinter import ttk, filedialog
import sys


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

    def _create_control_panel(self):
        """创建顶部控制按钮区域"""
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

    def _create_data_panel(self):
        """创建数据选择区域"""
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

    def _create_status_panel(self):
        """创建状态显示区域"""
        status_frame = ttk.LabelFrame(self.main_frame, text="训练状态")
        status_frame.pack(fill='x', pady=5)

        # 状态信息
        ttk.Label(status_frame, text="连接状态:").pack(side='left')
        self.conn_status = ttk.Label(status_frame, text="未连接", foreground='red')
        self.conn_status.pack(side='left', padx=10)

        ttk.Label(status_frame, text="训练进度:").pack(side='left')
        self.progress = ttk.Progressbar(status_frame, mode='determinate')
        self.progress.pack(side='left', padx=5, fill='x', expand=True)

    def _create_log_panel(self):
        """创建日志输出区域"""
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

    def _redirect_stdout(self):
        """重定向标准输出到日志框"""

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

    def _select_data(self):
        """处理数据选择"""
        if self.data_type.get() == 'dir':
            path = filedialog.askdirectory(title="选择训练数据文件夹")
        else:
            path = filedialog.askopenfilename(title="选择训练数据文件")

        if path:
            self.data_path.set(path)
            self.train_btn.config(state=tk.NORMAL)
            print(f"已选择数据路径: {path}")

    def _handle_connect(self):
        """处理连接按钮点击"""
        # 连接服务器逻辑
        self.conn_status.config(text="已连接", foreground='green')
        print("成功连接到服务器")

    def _handle_train(self):
        """处理训练按钮点击"""
        # 本地训练逻辑
        print(f"开始使用 {self.data_path.get()} 进行训练")
        self._simulate_training()

    def _simulate_training(self):
        """模拟训练进度"""

        def update_progress(progress):
            self.progress['value'] = progress
            self.update()

        for i in range(1, 101):
            self.after(50, update_progress, i)


if __name__ == "__main__":
    client = ClientGUI()
    client.mainloop()
