# 服务器GUI界面 05 02 ShenJiaLong

import sys
import tkinter as tk
from tkinter import ttk

class ServerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("联邦学习服务器")
        self.geometry("600x400")
        # 初始化回调函数
        self.start_callback = None
        self.stop_callback = None
        # 创建主容器
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        # 客户端数量
        self.client_count = tk.IntVar(value=0)
        # 控制面板区域
        self._create_control_panel()
        # 状态显示区域
        self._create_status_panel()
        # 日志输出区域
        self._create_log_panel()
        # 重定向标准输出
        self._redirect_stdout()

    def _create_control_panel(self):
        """创建顶部控制按钮区域"""
        control_frame = ttk.LabelFrame(self.main_frame, text="控制面板")
        control_frame.pack(fill='x', pady=5)

        # 按钮组
        btn_container = ttk.Frame(control_frame)
        btn_container.pack(pady=5)

        self.start_btn = ttk.Button(
            btn_container,
            text="启动训练",
            width=15,
            command=self._handle_start
        )
        self.start_btn.pack(side='left', padx=5)

        self.stop_btn = ttk.Button(
            btn_container,
            text="停止训练",
            width=15,
            state=tk.DISABLED,
            command=self._handle_stop
        )
        self.stop_btn.pack(side='left', padx=5)

    def _create_status_panel(self):
        """创建中间状态显示区域"""
        status_frame = ttk.LabelFrame(self.main_frame, text="训练状态")
        status_frame.pack(fill='x', pady=5)

        # 状态信息
        self.round_label = ttk.Label(status_frame, text="当前轮次: 0")
        self.round_label.pack(side='left', padx=10)

        self.client_label = ttk.Label(status_frame, textvariable=self.client_count)
        ttk.Label(status_frame, text="在线客户端:").pack(side='left')
        ttk.Label(status_frame, textvariable=self.client_count).pack(side='left')
        # self.client_label = ttk.Label(status_frame, text="在线客户端: 0")
        # self.client_label.pack(side='left', padx=10)

    def _create_log_panel(self):
        """创建底部日志输出区域"""
        log_frame = ttk.LabelFrame(self.main_frame, text="系统日志")
        log_frame.pack(fill='both', expand=True, pady=5)

        # 日志文本框
        self.log_text = tk.Text(
            log_frame,
            wrap='word',
            state='disabled',
            font=('Consolas', 10)
        )

        # 滚动条
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # 布局
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

    def _handle_start(self):
        """处理启动按钮点击"""
        if self.start_callback:
            try:
                self.start_callback()
                self.start_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.NORMAL)
            except Exception as e:
                print(f"启动失败: {str(e)}")

    def _handle_stop(self):
        """处理停止按钮点击"""
        if self.stop_callback:
            try:
                self.stop_callback()
                self.start_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
            except Exception as e:
                print(f"停止失败: {str(e)}")

    def set_callbacks(self, start_func, stop_func):
        """设置回调函数"""
        self.start_callback = start_func
        self.stop_callback = stop_func

    def update_client_count(self, count):
        """线程安全更新客户端数量"""
        self.client_count.set(count)


if __name__ == "__main__":
    app = ServerGUI()
    app.mainloop()
