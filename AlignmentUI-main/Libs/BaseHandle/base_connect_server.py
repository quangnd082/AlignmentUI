from abc import ABCMeta, abstractmethod
from PyQt5.QtCore import QObject, pyqtSignal
import socket


# Metaclass kết hợp
class MetaQObjectABC(ABCMeta, type(QObject)):
    pass


class AbstractServer(QObject, metaclass=MetaQObjectABC):
    triggerOn = pyqtSignal(str)
    sendData = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.client_address = None
        self.client_socket = None
        self.server_socket = None
        self.is_connected = False
        self.ip = None
        self.port = None

    @abstractmethod
    def start_server(self, ip='127.0.0.1', port=8000):
        """Khởi động server và lắng nghe kết nối từ client"""
        pass

    @abstractmethod
    def stop_server(self):
        """Ngừng server và đóng kết nối"""
        pass

    @abstractmethod
    def accept_client(self):
        """Chấp nhận kết nối từ client"""
        pass

    @abstractmethod
    def loop_recv_client(self, conn: socket.socket):
        """Lặp lại việc nhận dữ liệu từ client"""
        pass
