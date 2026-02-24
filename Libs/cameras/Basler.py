from Libs.cameras.base_camera import *
from pypylon import pylon
import numpy as np


class Basler(BaseCamera):

    def __init__(self, config=None) -> None:
        super().__init__(config=config)

        self._camera = None
        self._converter = None
        self._is_grabbing = False

    # ---------------- REQUIRED ----------------

    def get_error(self) -> str:
        return self._error

    def get_devices(self) -> dict:
        devices = {}
        tl_factory = pylon.TlFactory.GetInstance()
        devs = tl_factory.EnumerateDevices()

        for i, dev in enumerate(devs):
            devices[str(i)] = dev.GetSerialNumber()

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
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()

            if len(devices) == 0:
                self._error = ERR_NOT_FOUND_DEVICE
                return

            device_id = self._config.get("id", 0)
            device = devices[device_id] if device_id < len(devices) else devices[0]

            self._camera = pylon.InstantCamera(
                tl_factory.CreateDevice(device)
            )

            self._camera.Open()

            self._model_name = self._camera.GetDeviceInfo().GetModelName()

            # Pixel converter
            self._converter = pylon.ImageFormatConverter()
            self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self._setup_trigger()

        except:
            self._error = ERR_CREATE_DEVICE_FAIL

    # ---------------- TRIGGER SETUP ----------------

    def _setup_trigger(self):

        self._camera.TriggerSelector.SetValue("FrameStart")
        self._camera.TriggerMode.SetValue("On")

        # Hardware trigger
        self._camera.TriggerSource.SetValue("Line1")

        self._camera.TriggerActivation.SetValue("RisingEdge")

        # Optional: disable auto exposure if needed
        # self._camera.ExposureAuto.SetValue("Off")

    # ---------------- OPEN / CLOSE ----------------

    def open(self) -> bool:
        return self._camera.IsOpen()

    def close(self) -> bool:
        try:
            if self._camera:
                self._camera.Close()
            return True
        except:
            return False

    # ---------------- GRAB CONTROL ----------------

    def start_grabbing(self) -> bool:
        try:
            self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self._is_grabbing = True
            return True
        except:
            return False

    def stop_grabbing(self) -> bool:
        try:
            self._camera.StopGrabbing()
            self._is_grabbing = False
            return True
        except:
            return False

    # ---------------- TRIGGER GRAB ----------------

    def grab(self, timeout=2000):
        """
        Wait for hardware trigger
        """
        if not self._is_grabbing:
            return ERR_GRAB_FAIL, None

        grab_result = self._camera.RetrieveResult(timeout)

        if grab_result.GrabSucceeded():
            image = self._converter.Convert(grab_result)
            frame = image.GetArray()
            grab_result.Release()
            return NO_ERROR, frame

        grab_result.Release()
        return ERR_GRAB_FAIL, None