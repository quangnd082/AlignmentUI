from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum
import cv2

class Step(Enum):
    SCANNER = "STEP_SCANNER"
    READ_TRIGGER = "STEP_READ_TRIGGER"
    LOAD_CONFIG = "STEP_LOAD_CONFIG"
    CHANGE_MODEL = "STEP_CHANGE_MODEL"
    PREPROCESS = "STEP_PREPROCESS"
    ON_LIGHTING = "STEP_ON_LIGHTING"
    OFF_LIGHTING = "STEP_OFF_LIGHTING"
    UNDISTORT_CAMERA = "UNDISTORT_CAMERA"
    VISION_DETECTION = "STEP_VISION_DETECTION"  # Đã sửa từ DETETION -> DETECTION
    VISION_CHECKING_LED = "STEP_VISION_CHECKING_LED"
    VISION_COMBINE = "STEP_VISION_COMBINE"
    OUTPUT = "STEP_OUTPUT"
    RECHECK_READ_TRIGGER = "STEP_RECHECK_READ_TRIGGER"
    RECHECK_HANDLE = "STEP_RECHECK_HANDLE"
    RECHECK_SCAN_GEN = "STEP_RECHECK_SCAN_GEN"
    WRITE_LOG = "STEP_WRITE_LOG"
    RELEASE = "STEP_RELEASE"
    ERROR = "STEP_ERROR"
    CHECK_SENSOR_ON = "STEP_CHECK_SENSOR_ON"
    CHECK_SENSOR_OFF = "STEP_CHECK_SENSOR_OFF"

class StepResult(Enum):
    WAIT = "WAIT"
    PASS_ = "PASS"
    FAIL = "FAIL"

aruco_dict_mapping = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

@dataclass
class RESULT:
    model_name: Optional[Any] = None
    
    src: Optional[Any] = None
   
    dst: Optional[Any] = None
    
    ret: Optional[Any] = None
    
    label_counts: Optional[Any] = None
    
    timecheck: Optional[Any] = None
    
    error: Optional[Any] = None
    
    offset: Optional[Any] = None
    
    