import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QWidget, QFormLayout
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt

class DemoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('PyWebView 混合应用')
        self.setGeometry(100, 100, 400, 300)

        # 创建布局
        mainLayout = QVBoxLayout()

        # 创建头部
        headerLayout = QHBoxLayout()
        logoLabel = QLabel()
        logoLabel.setPixmap(QIcon("https://via.placeholder.com/40").pixmap(40, 40))
        headerLabel = QLabel('PyWebView 混合应用')
        headerLayout.addWidget(logoLabel)
        headerLayout.addWidget(headerLabel)
        mainLayout.addLayout(headerLayout)

        # 创建按钮
        button = QPushButton('获取系统信息')
        button.clicked.connect(self.on_button_clicked)
        mainLayout.addWidget(button)

        # 创建输入框和保存按钮
        inputLayout = QHBoxLayout()
        inputLabel = QLabel('nihaome')
        inputField = QLineEdit()
        saveButton = QPushButton('保存数据')
        saveButton.clicked.connect(self.on_save_button_clicked)
        inputLayout.addWidget(inputLabel)
        inputLayout.addWidget(inputField)
        inputLayout.addWidget(saveButton)
        mainLayout.addLayout(inputLayout)

        self.setLayout(mainLayout)

    def on_button_clicked(self):
        print("获取系统信息按钮被点击")

    def on_save_button_clicked(self):
        print("保存数据按钮被点击")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DemoApp()
    ex.show()
    sys.exit(app.exec())