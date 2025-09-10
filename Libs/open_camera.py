from tkinter import E
import cameras
from cameras import Webcam, HIK, SODA
import cv2
from Logging import Logger


class Camera:
    def __init__(self):
        self.camera = None
        self.cap = None
        self.is_open = False
        self.camera_logger = Logger('Camera')

    def open_camera(self, name, config = {
        "id": 0,
        "feature": ""
    }):
        if name == 'HIK':
            self.camera = HIK(config=config)
        elif name == 'SODA':
            self.camera = SODA(config=config)
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

    def get_frame(self):
        ret, frame = self.camera.grab()
        return frame

    def close_camera(self):
        try:
            self.camera.stop_grabbing()
            # cv2.destroyAllWindows()
            self.is_open = False
            self.camera.close()
            self.camera_logger.info('Camera was closed')            
        except Exception as ex:
            self.camera_logger.error('Failed to close camera')
            

if __name__ == '__main__':
    camera = Camera()
    camera.open_camera('Webcam')
    while True:
        frame = camera.get_frame()
        cv2.namedWindow('Trung', cv2.WINDOW_FREERATIO)
        cv2.imshow('Trung', frame)
        if cv2.waitKey(1) & 0xff == 27:
            cv2.destroyAllWindows()
            break
