import cv2
import json
import math

def show_image(image):
    cv2.namedWindow('Trung', cv2.WINDOW_FREERATIO)
   
    cv2.imshow('Trung', image)
   
    if 0xff == 27 & cv2.waitKey(0):
        cv2.destroyAllWindows()  

def cal_signed_angle_beetween_two_vecto(vec1: tuple, vec2: tuple) -> float:
    x1, y1 = vec1
    x2, y2 = vec2

    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2

    angle_rad = math.atan2(det, dot)
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def load_config(save_path):
    try:
        with open(save_path,'r') as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(e)

def save_config(data, save_path):
    try:
        with open(save_path, 'w') as file:
            json.dump(data, file, indent=4)
            return True
    except Exception as e:
        print(e)
        return False