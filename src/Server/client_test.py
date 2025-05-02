# 测试客户端（可另存为client_test.py）
import socket
import time

def test_client():
    for _ in range(3):
        try:
            with socket.create_connection(('localhost', 8888)) as s:
                print("连接成功，等待10秒...")
                time.sleep(10)
        except Exception as e:
            print(f"连接失败: {e}")
        time.sleep(1)

if __name__ == "__main__":
    test_client()
