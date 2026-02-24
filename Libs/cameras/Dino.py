import cv2
import threading
import time
from Libs.cameras.DNX64.DNX64 import DNX64
from Libs.cameras.base_camera import *

class DINO(BaseCamera):

    def __init__(self, config=None) -> None:
        super().__init__(config=config)

        self._dnx = None
        self._device_index = 0

        self._running = False
        self._grab_thread = None
        self._lock = threading.Lock()

        self._latest_frame = None

        # Auto LED
        self._auto_led = False
        self._led_state = False

    # ---------------- REQUIRED ----------------

    def get_error(self) -> str:
        return self._error

    def get_devices(self) -> dict:
        devices = {}
        try:
            dll_path = self._config.get("dll_path")
            dnx = DNX64(dll_path)

            if not dnx.Init():
                return devices

            count = dnx.GetVideoDeviceCount()

            for i in range(count):
                devices[str(i)] = i

        except:
            pass

        return devices

    def set_config(self, config):
        if config is None:
            self._error = ERR_CONFIG_IS_NONE
            return

        self._config = config
        self.create_device()

    def get_config(self):
        return self._config

    # ---------------- CREATE DEVICE ----------------

    def create_device(self):
        try:
            dll_path = self._config.get("dll_path")

            self._dnx = DNX64(dll_path)

            if not self._dnx.Init():
                self._error = ERR_NOT_FOUND_DEVICE
                return

            count = self._dnx.GetVideoDeviceCount()
            if count == 0:
                self._error = ERR_NOT_FOUND_DEVICE
                return

            device_id = self._config.get("id", 0)
            self._device_index = device_id if device_id < count else 0

            self._dnx.SetVideoDeviceIndex(self._device_index)

            # Auto LED flag
            self._auto_led = self._config.get("auto_led", False)

            self._cap = cv2.VideoCapture(self._device_index, cv2.CAP_DSHOW)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self._model_name = f"DINO_{self._device_index}"

        except:
            self._error = ERR_CREATE_DEVICE_FAIL

    # ---------------- OPEN / CLOSE ----------------

    def open(self) -> bool:
        if self._cap is None:
            return False
        return self._cap.isOpened()

    def close(self) -> bool:
        self.stop_grabbing()
        try:
            if self._cap:
                self._cap.release()
            return True
        except:
            return False

    # ---------------- STREAM ----------------

    def start_grabbing(self) -> bool:
        if self._running:
            return True

        self._running = True
        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thread.start()

        return True

    def stop_grabbing(self) -> bool:
        self._running = False
        return True

    def _grab_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._latest_frame = frame

    # ---------------- LED CONTROL ----------------

    def _led_on(self):
        if self._dnx and not self._led_state:
            self._dnx.SetLEDState(self._device_index, 1)
            self._led_state = True
            time.sleep(0.02)  # LED stabilize

    def _led_off(self):
        if self._dnx and self._led_state:
            self._dnx.SetLEDState(self._device_index, 0)
            self._led_state = False

    # ---------------- GRAB ----------------

    def grab(self, timeout=2.0):

        start = time.time()

        if self._auto_led:
            self._led_on()

        while True:
            with self._lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()

                    if self._auto_led:
                        self._led_off()

                    return NO_ERROR, frame

            if time.time() - start > timeout:
                if self._auto_led:
                    self._led_off()

                self._error = ERR_GRAB_FAIL
                return self._error, None

            time.sleep(0.01)