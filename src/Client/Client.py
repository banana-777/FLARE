# 客户端主入口 05 02 ShenJiaLong

import threading
from ClientGUI import ClientGUI
from ClientCore import ClientCore
from Logger import Logger

class Client:
    def __init__(self):
        self.gui = ClientGUI(self)
        self.core = ClientCore(self)
        self.logger = Logger("客户端日志.log")

        self.is_training = False
        self.is_connected = False
        self.server_host = '127.0.0.1'
        self.server_port = 8888
        self.server_socket = None
        self.model_arch = None
        self.model = None
        self.train_data = None
        self.test_data = None

        self.core.connect_server(self.server_host, self.server_port)
        threading.Thread(target=self.core.msg_handler, daemon=True).start()


if __name__ == "__main__":
    client = Client()
    client.gui.mainloop()