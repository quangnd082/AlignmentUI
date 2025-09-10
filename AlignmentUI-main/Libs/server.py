from Logging import Logger
from PyQt5.QtCore import QObject, pyqtSignal
import time
import re
import socket
import threading
from BaseHandle.base_connect_server import AbstractServer


class Server(AbstractServer):
    def __init__(self):
        super().__init__()
        self.server_logger = Logger('Server')  # Dùng instance logger
        self.client_sockets = []  # Danh sách các client đang kết nối

    def start_server(self, ip='127.0.0.1', port=8000):
        try:
            self.ip = ip
            self.port = port
            self.server_logger.info('Starting the server')
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.ip, self.port))
            self.server_socket.listen(1)
            self.is_connected = True
            threading.Thread(target=self.accept_client, daemon=True).start()
        except Exception as e:
            self.is_connected = False
            self.server_logger.error(e)

    def accept_client(self):
        while self.is_connected:
            try:
                client_socket, client_address = self.server_socket.accept()
                if not self.is_connected:
                    client_socket.close()
                    break
                else:
                    self.client_sockets.append(client_socket)
                    threading.Thread(target=self.loop_recv_client, args=(client_socket,), daemon=True).start()
                    self.server_logger.info(f'{client_address} connected')
            except Exception as e:
                if self.is_connected:
                    self.server_logger.error(e)
                break

    def stop_server(self):
        try:
            self.is_connected = False
            for client in self.client_sockets:
                try:
                    client.close()
                except:
                    pass
            self.client_sockets.clear()
            if self.server_socket:
                self.server_socket.close()
                self.server_logger.info('Disconnected')
        except Exception as e:
            self.server_logger.error(e)

    def loop_recv_client(self, conn: socket.socket):
        try:
            msg = conn.recv(1024).decode().strip()
            self.sendData.emit(msg)
            if re.fullmatch(r"T\d*", msg):
                self.triggerOn.emit(msg)
                time.sleep(0.01)
            else:
                if conn in self.client_sockets:
                    self.client_sockets.remove(conn)
                    conn.close()
                
        except Exception as e:
            self.server_logger.error(f"Error receiving data: {e}")         

    def send_data_to_client(self, data: str, conn: socket.socket):
        try:
            if conn:
                conn.sendall(data.encode())
                self.server_logger.info(data)
                if conn in self.client_sockets:
                    self.client_sockets.remove(conn)
                conn.close()
        except Exception as e:
            self.server_logger.error(e)
            

    def send_data_to_all_clients(self, data: str):
        for conn in self.client_sockets[:]:
            try:
                conn.sendall(data.encode())
                self.server_logger.info(data)
                self.client_sockets.remove(conn)
                conn.close()
            except Exception as e:
                self.server_logger.error(e)
