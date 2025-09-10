import serial
from Logging import Logger
from PyQt5.QtCore import pyqtSignal
from BaseHandle.base_connect_serial import AbstractSerialReceiver

class SerialReceiver(AbstractSerialReceiver):
    checkSignalScanner = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.ser = None
        self.scanner_logger = Logger('Scanner')

    def connect(self, port, baudrate):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            self.scanner_logger.info(f'Connected to {port} at {baudrate}')
            return True
        except serial.SerialException as e:
            self.scanner_logger.error(f"Failed to connect COM: {e}")
            return False

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.scanner_logger.info('Disconnected')

    def read_data(self):
        if self.ser and self.ser.is_open and self.ser.in_waiting:
            try:
                code = self.ser.readline().decode('utf-8').strip()
                self.scanner_logger.info(f'Received: {code}')
                self.checkSignalScanner.emit(code)
                return code
            except Exception as e:
                self.scanner_logger.error(f"Read error: {e}")
                return None
        return None

    def is_connected(self):
        return self.ser and self.ser.is_open

    def send_data(self, data):
        """Gửi dữ liệu ra COM (kết thúc bằng newline)."""
        if self.ser and self.ser.is_open:
            try:
                self.ser.write((data + '\n').encode('utf-8'))
                self.scanner_logger.info(f'Sent: {data}')
                return True
            except serial.SerialException as e:
                self.scanner_logger.error(f'Send error: {e}')
        else:
            self.scanner_logger.warning("Serial port not open")
        return False
    
    