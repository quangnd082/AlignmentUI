from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from Logging import Logger
from serial import Serial
import time

class DCPController:
    def __init__(self):
        super().__init__()
        self.comport: Serial = None
        self.light_logger = Logger('DCPController')

    def is_open(self):
        return self.comport is not None and self.comport.is_open
    
    def open(self, com, baud):
        self.close()
        time.sleep(0.1)
        try:
            self.comport = Serial(port=com, baudrate=baud)
            self.light_logger.info(f"Opened COM port {com}")
            return self.comport.is_open
        except Exception as e:
            self.light_logger.error(f"Failed to open {com}")
            self.comport = None
            return False

    def close(self):
        try:
            if self.comport is not None:
                self.off_all_channels()
                self.comport.close()
                self.light_logger.info("Closed COM port")
            return True
        except Exception as e:
            self.light_logger.error(f"Failed to close COM port")
            return False

    def off_all_channels(self):
        self.set_light_value(0, 0)
        self.set_light_value(1, 0)
        self.set_light_value(2, 0)
        self.set_light_value(3, 0)
        # self.light_logger.info("Turned off all channels")  # Comment để giảm log khi start/stop liên tục

    def send_data(self, data):
        if self.is_open():
            # self.light_logger.debug(f"Sending data: {data}")
            self.comport.write(data.encode())
        else:
            # self.light_logger.warning("Attempted to send data while COM port is closed")
            pass

    def set_light_value(self, channel=0, val=10):
        channel_str = {0: "SA", 1: "SB", 2: "SC", 3: "SD"}.get(channel, "")
        value_str = str(val).zfill(3)
        data = f"{channel_str}0{value_str}#"
        self.send_data(data)
        # self.light_logger.info(f"Set light value on channel {channel}: {val}")


class LCPController():
    def __init__(self):
        super().__init__()
        self.comport: Serial = None
        self.light_logger = Logger('LCPController')

    def is_open(self):
        return self.comport is not None and self.comport.is_open

    def set_light_value(self, channel=0, val=100):
        data = "\x02%sw%.4d\x03" % (channel, val)
        self.send_data(data)
        # self.light_logger.info(f"Set light value on channel {channel}: {val}")

    def on_channel(self, channel=0):
        data = "\x02%so\x03" % (channel)
        self.send_data(data)
        # self.light_logger.info(f"Turned on channel {channel}")  # Comment để giảm log

    def off_channel(self, channel=0):
        data = "\x02%sf\x03" % (channel)
        self.send_data(data)
        # self.light_logger.info(f"Turned off channel {channel}")

    def send_data(self, data):
        if self.is_open():
            # self.light_logger.debug(f"Sending data: {data}")
            self.comport.write(data.encode())
        else:
            # self.light_logger.warning("Attempted to send data while COM port is closed")
            pass

    def off_all_channels(self):
        for i in range(4):
            self.off_channel(i)
        # self.light_logger.info("Turned off all channels")

    def open(self, com, baud):
        self.close()
        time.sleep(0.1)
        try:
            self.comport = Serial(port=com, baudrate=baud)
            self.light_logger.info(f"Opened COM port {com}")
            return self.comport.is_open
        except Exception as e:
            self.light_logger.error(f"Failed to open {baud}")
            self.comport = None
            return False

    def close(self):
        try:
            if self.comport is not None:
                self.off_all_channels()
                self.comport.close()
                self.light_logger.info("Closed COM port")
            return True
        except Exception as e:
            self.light_logger.error(f"Failed to close COM port")
            return False


if __name__ == "__main__":
    lcp = LCPController(com="COM3")
    print(lcp.open())
    lcp.set_light_value(1, 500)
    time.sleep(3)
    lcp.close()