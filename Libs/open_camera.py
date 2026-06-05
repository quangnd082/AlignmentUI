from Libs.cameras.hik import HIK
from Libs.cameras.soda import SODA
from Libs.cameras.webcam import Webcam
from cameras.Dino import DINO
from Logging import Logger


class Camera:
    def __init__(self):
        self.camera = None
        self.cap = None
        self.is_open = False
        self.camera_logger = Logger('Camera')

    # ================= OPEN =================

    def open_camera(self, name, config={
        "id": 0,
        "feature": ""
    }):

        if name == 'DINO':
            self.camera = DINO(config={
                "dll_path": "DNX64.dll",
                "id": int(config.get("id", 0))
            })

        elif name == 'SODA':
            self.camera = SODA(config=config)

        elif name == 'HIK':
            self.camera = HIK(config=config)

        else:
            self.camera = Webcam(config=config)

        if not self.is_open:

            self.cap = self.camera.open()

            if self.cap:
                self.cap &= self.camera.start_grabbing()
                self.is_open = True
                self.camera_logger.info('Camera was opened')
                return True
            else:
                self.camera_logger.error('Failed to open camera')
                return False

    # ================= FRAME =================

    def get_frame(self):
        if not self.is_open:
            return None

        ret, frame = self.camera.grab()
        return frame

    # ================= CLOSE =================

    def close_camera(self):
        try:
            if self.camera:
                self.camera.stop_grabbing()
                self.camera.close()

            self.is_open = False
            self.camera_logger.info('Camera was closed')

        except Exception as ex:
            self.camera_logger.error(f'Failed to close camera: {ex}')

    # ================= LED CONTROL =================

    def led_on(self):
        """
        Works only for camera that supports LED (ex: DINO)
        """
        if self.camera and hasattr(self.camera, "led_on"):
            self.camera.led_on()

    def led_off(self):
        """
        Works only for camera that supports LED (ex: DINO)
        """
        if self.camera and hasattr(self.camera, "led_off"):
            self.camera.led_off()