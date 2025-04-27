# -*- coding: utf-8 -*-
# FLARE混合应用核心框架 v1.2 - 安全稳定版

import os
import sys
import webview
import platform

# ======================
# 关键修复1：禁用CLR调试端口
# ======================
if sys.platform == 'win32':
    os.environ.update({
        'WEBVIEW_ENABLE_CLR_DEBUGGING': '0',  # 禁用.NET调试接口
        'WEBVIEW_DISABLE_EDGE_CHROMIUM_DEBUG': '1'  # 关闭Edge调试
    })


# ======================
# 联邦学习API接口层
# ======================
class FLAREApi:
    def __init__(self):
        self._window = None

    def expose(self, window):
        """安全绑定窗口对象"""
        self._window = window

    def get_platform_info(self):
        """获取运行时信息"""
        return {
            'os': platform.system(),
            'arch': platform.architecture()[0],
            'python_version': sys.version.split()[0],
            'gui_engine': self._detect_gui_engine()
        }

    def _detect_gui_engine(self):
        """检测当前使用的渲染引擎"""
        if 'edgechromium' in webview.settings.get('gui', '').lower():
            return 'Edge Chromium'
        return 'System Default'


# ======================
# 资源路径处理模块
# ======================
def _resource_path(relative_path):
    """统一资源路径处理（兼容打包环境）"""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    full_path = os.path.join(base_path, relative_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"关键资源丢失: {full_path}")

    # Windows路径转换
    if sys.platform == 'win32':
        return full_path.replace('\\', '/')
    return full_path


# ======================
# 主程序入口
# ======================
if __name__ == '__main__':
    try:
        # ======================
        # 初始化配置
        # ======================
        webview.settings.update({
            # 'debug': False,  # 彻底关闭调试模式
            # 'http_server': False,  # 禁用HTTP服务
            # 'private_mode': True  # 隐私模式禁止缓存
            
        })

        # ======================
        # 创建应用实例
        # ======================
        api = FLAREApi()
        html_path = f'file:///{_resource_path("index.html")}'

        window = webview.create_window(
            title='联邦学习监控平台',
            url=html_path,
            js_api=api,
            width=1280,
            height=720,
            min_size=(1024, 768),
            confirm_close=True,
            text_select=False  # 禁止文本选择提升安全性
        )
        api.expose(window)

        # ======================
        # 启动引擎配置
        # ======================
        gui_engine = 'edgechromium' if sys.platform == 'win32' else None

        webview.start(
            gui=gui_engine,
            http_server=False,
            user_agent='FLARE-Monitor/1.0'  # 定制UA标识
        )

    except Exception as e:
        import traceback

        error_msg = f"""
        === 联邦学习平台启动失败 ===
        错误类型: {type(e).__name__}
        错误详情: {str(e)}
        堆栈跟踪:
        {traceback.format_exc()}
        """
        print(error_msg)
        sys.exit(1)
