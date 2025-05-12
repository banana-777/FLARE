# 日志记录 05 12 ShenJiaLong

import os
from datetime import datetime


class Logger:
    def __init__(self, log_file):
        # 分解文件名和扩展名
        base, ext = os.path.splitext(log_file)
        # 生成时间戳格式：年月日_时分秒
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        # 组合新文件名
        self.log_file = f"{base}_{timestamp}{ext}"
        self._init_log_file()

    def _init_log_file(self):
        header = f"\n{'=' * 40}\n联邦学习系统日志\n初始化时间: {self._get_time()}\n{'=' * 40}\n\n\n"
        with open(self.log_file, 'a') as f:
            f.write(header)

    def log(self, message, log_type):
        log_entry = f"[{self._get_time()}] [{log_type}] {message}\n"
        with open(self.log_file, 'a') as f:
            f.write(log_entry)

    def log_train_stats(self, epoch, loss, accuracy, duration):
        stats = f"Epoch {epoch} | Loss: {loss:.4f} | Acc: {accuracy:.2f}% | Time: {duration:.1f}s"
        self.log(stats, "TRAIN")

    def log_comm_stats(self, event, data_size, compressed_size=None):
        if compressed_size:
            ratio = compressed_size / data_size
            msg = f"{event} | 原始大小: {data_size / 1024:.1f}KB | 压缩后: {compressed_size / 1024:.1f}KB | 压缩率: {ratio:.1%}"
        else:
            msg = f"{event} | 数据大小: {data_size / 1024:.1f}KB"
        self.log(msg, "COMM")

    def _get_time(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
