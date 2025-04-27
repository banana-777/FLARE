import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QStackedWidget
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6多页面演示")
        self.setGeometry(100, 100, 800, 600)

        # ======================
        # 主界面布局
        # ======================
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 使用水平布局分为侧边栏和内容区
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)  # 去除默认边距
        main_layout.setSpacing(0)

        # ======================
        # 侧边栏 (200px宽度)
        # ======================
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet("background-color: #f0f0f0;")
        sidebar_layout = QVBoxLayout(sidebar)

        # 导航按钮
        self.btn_page1 = QPushButton("仪表盘")
        self.btn_page2 = QPushButton("数据视图")
        self.btn_page3 = QPushButton("系统设置")

        # 统一设置按钮样式
        # btn_style = """
        #     QPushButton {
        #         padding: 12px;
        #         text-align: left;
        #         border: none;
        #         background: transparent;
        #     }
        #     QPushButton:hover {
        #         background-color: #e0e0e0;
        #     }
        # """
        btn_style = """
            QPushButton {
                padding: 12px;
                text-align: left;
                border: none;
                background: transparent;
                color: #333333;  /* 新增：默认字体颜色为深灰 */
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                color: #000000;  /* 新增：悬停时字体更黑 */
            }
            QPushButton:checked {
                background-color: #d0d0d0;
                color: #0000FF;  /* 新增：选中状态字体为蓝色 */
            }
        """

        for btn in [self.btn_page1, self.btn_page2, self.btn_page3]:
            btn.setStyleSheet(btn_style)

        # 添加按钮到侧边栏
        sidebar_layout.addWidget(self.btn_page1)
        sidebar_layout.addWidget(self.btn_page2)
        sidebar_layout.addWidget(self.btn_page3)
        sidebar_layout.addStretch()  # 在下方添加弹性空间

        # ======================
        # 内容区 - 使用堆叠容器
        # ======================
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setStyleSheet("background-color: white;")

        # 初始化三个示例页面
        self.init_pages()

        # ======================
        # 组合布局
        # ======================
        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.stacked_widget)

        # ======================
        # 信号连接
        # ======================
        self.btn_page1.clicked.connect(lambda: self.switch_page(0))
        self.btn_page2.clicked.connect(lambda: self.switch_page(1))
        self.btn_page3.clicked.connect(lambda: self.switch_page(2))

    def init_pages(self):
        """初始化所有子页面"""
        # 页面1 - 仪表盘
        page1 = QWidget()
        layout1 = QVBoxLayout(page1)
        layout1.addWidget(QLabel("<h1>系统仪表盘</h1>"))
        layout1.addWidget(QLabel("训练进度: 75%"))
        layout1.addStretch()
        self.stacked_widget.addWidget(page1)

        # 页面2 - 数据视图
        page2 = QWidget()
        layout2 = QVBoxLayout(page2)
        layout2.addWidget(QLabel("<h1>数据可视化</h1>"))
        layout2.addWidget(QLabel("数据分布图表区域"))
        layout2.addStretch()
        self.stacked_widget.addWidget(page2)

        # 页面3 - 系统设置
        page3 = QWidget()
        layout3 = QVBoxLayout(page3)
        layout3.addWidget(QLabel("<h1>系统配置</h1>"))
        layout3.addWidget(QLabel("参数设置面板"))
        layout3.addStretch()
        self.stacked_widget.addWidget(page3)

    def switch_page(self, index):
        """切换页面并更新按钮状态"""
        # 参数有效性检查
        if 0 <= index < self.stacked_widget.count():
            self.stacked_widget.setCurrentIndex(index)

            # 更新按钮选中状态
            buttons = [self.btn_page1, self.btn_page2, self.btn_page3]
            for i, btn in enumerate(buttons):
                if i == index:
                    btn.setStyleSheet("background-color: #d0d0d0;")
                else:
                    btn.setStyleSheet("background-color: transparent;")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
