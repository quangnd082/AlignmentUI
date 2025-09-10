import math
import cv2
import numpy as np
import os
import glob
import json
from dataclasses import dataclass
from typing import Any, Optional
from Utilities import *


class Pose:
    def __init__(self, center, angle, anchor, size, tcp_pos=None):
        self._center = center
        self._anchor = anchor
        self._angle = angle
        self._size = size
        self._tcp_pos = tcp_pos

    @property
    def center(self):
        return self._center
   
    @property
    def anchor(self):
        return self._anchor

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value
       
    @property
    def size(self):
        return self._size

    @property
    def data(self):
        return {
            "center": self.center,
            "anchor": self.anchor,
            "angle": self.angle,
            "size": self.size,
            "tcp_pos": self._tcp_pos,
        }

    def __sub__(self, other):
        tcp_pos_current = self.get_tcp_pos()
        
        tcp_pos_origin = other.get_tcp_pos()
        
        offset_tcp_pos_x = tcp_pos_current[0] - tcp_pos_origin[0]
        
        offset_tcp_pos_y = tcp_pos_current[1] - tcp_pos_origin[1]
        
        offset_angle = self.angle - other.angle
        
        return Pose(None, offset_angle, None, None, (offset_tcp_pos_x, offset_tcp_pos_y))

    def __str__(self):
        tcp_pos = self.get_tcp_pos()
        return f'T_X: {tcp_pos[0]: .2f}mm, T_Y: {tcp_pos[1]: .2f}mm, Rot from origin: {self.angle:.2f} degrees'

    def get_tcp_pos(self):
        return self._tcp_pos

    def set_tcp_pos(self, tcp_pos):
        self._tcp_pos = tcp_pos

    def cal_scale_pixel_to_mm(self, real_size=(40, 20)):
        if self.size is not None:
            t_x = real_size[0] / self.size[0]
            t_y = real_size[1] / self.size[1]
            scale = (t_x + t_y) / 2
            return scale
        return None
    
    def scale(self, scale):
        tcp_pos = self.get_tcp_pos()
        return Pose(
            None,
            self.scale_angle(self.angle),
            None,
            None,
            (tcp_pos[0] * scale, tcp_pos[1] * scale)
        )
    
    def scale_angle(self, angle):
        if angle > 180:
            angle -= 360
        elif angle <= -180:
            angle += 360
        
        return angle

    def cal_tcp_pos(self, other):

        centroid_org = (other.center[0], other.center[1])
        angle_org = other.angle
        tcp_pos_org = other.get_tcp_pos()

        centroid_current = (self.center[0], self.center[1])
        angle_current = self.angle

        # Vector tịnh tiến từ pose gốc sang pose hiện tại
        v_translate = (
            centroid_current[0] - centroid_org[0],
            centroid_current[1] - centroid_org[1]
        )

        # Dịch TCP theo tịnh tiến
        tcp_pos_translate = (
            tcp_pos_org[0] + v_translate[0],
            tcp_pos_org[1] + v_translate[1]
        )

        # Đưa về tọa độ local so với tâm hiện tại
        tcp_pos_local = (
            tcp_pos_translate[0] - centroid_current[0],
            tcp_pos_translate[1] - centroid_current[1]
        )

        # Góc quay tương đối
        theta = angle_current - angle_org
        
        theta = self.scale_angle(theta)
        
        theta_radian = np.radians(theta)

        # Xoay TCP quanh tâm hiện tại
        px = tcp_pos_local[0] * np.cos(theta_radian) - (-tcp_pos_local[1]) * np.sin(theta_radian)
        py = tcp_pos_local[0] * np.sin(theta_radian) + (-tcp_pos_local[1]) * np.cos(theta_radian)

        # Tính TCP sau khi xoay và dịch ngược trục y
        tcp_pos_current = (
            px + centroid_current[0],
            -py + centroid_current[1]
        )

        # Cập nhật giá trị
        # self.angle = theta
        # self.set_tcp_pos(tcp_pos)

        return tcp_pos_current    
        
    @classmethod
    def from_json_to_pose(cls, data: dict):
        return Pose(data['center'], data['angle'], data['anchor'], data['size'], data['tcp_pos'])

class ImageDrawer:
    def __init__(self, image):
        self.original = image.copy()
        self.image = image.copy()

    def draw_contours(self, contours, color=(0, 255, 0), thickness=2):
        for contour in contours:
            cv2.drawContours(self.image, [contour], 0, color, thickness)
        return self

    def draw_box(self, box, color=(0, 255, 0), thickness=2):
        cv2.drawContours(self.image, [box], 0, color, thickness)
        return self

    def draw_point(self, point, color=(0, 0, 255), radius=5, thickness=-1):
        cv2.circle(self.image, (int(point[0]), int(point[1])), radius, color, thickness)
        return self

    def draw_text(self, text, locate, font=cv2.FONT_HERSHEY_SIMPLEX,
                  font_scale=1.0, color=(0, 255, 0), thickness=2):
        cv2.putText(self.image, text, locate, font, font_scale, color, thickness)
        return self
    
    def draw_line(self, pt1, pt2, color=(255, 0, 0), thickness=2, tip_length=0.1):
        cv2.arrowedLine(self.image, 
                        (int(pt1[0]), int(pt1[1])), 
                        (int(pt2[0]), int(pt2[1])), 
                        color, thickness, tipLength=tip_length)
        return self

    def get_origin_image(self):
        return self.original

    def get_drawn_image(self):
        return self.image

@dataclass
class RESULT:
    src: Optional[Any] = None
   
    dst: Optional[Any] = None
   
    image_binary: Optional[Any] = None
   
    pose: Pose = None
   

def run_preprocessing_image(image, **kwargs):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
           
           
    blur_kernel_size = kwargs.get('blur_kernel_size', 3)
    blur = cv2.blur(gray, (blur_kernel_size, blur_kernel_size))
   
   
    threshold_method = kwargs.get('threshold_method', 'Normal')
    threshold = kwargs.get('threshold_value', 55)
    block_size = kwargs.get('block_size', 3)
    threshold_type = kwargs.get('threshlod_type', 'Normal')
    use_otsu = kwargs.get('use_otsu', False)
   
    if threshold_type == 'Normal':
        threshold_type = cv2.THRESH_BINARY
    else:
        threshold_type = cv2.THRESH_BINARY_INV
   
    if threshold_method == 'Normal':
        if use_otsu:
            threshold_type += cv2.THRESH_OTSU
        _, image_binary = cv2.threshold(blur, threshold, 255, threshold_type)
    else:
        image_binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             threshold_type, block_size, threshold)
       
    contour_mode = kwargs.get('contour_mode', cv2.RETR_EXTERNAL)
    contour_method = kwargs.get('contour_method', cv2.CHAIN_APPROX_SIMPLE)
    
    if contour_mode == 'RETR_EXTERNAL':
        contour_mode = cv2.RETR_EXTERNAL
    elif contour_mode == 'RETR_CCOMP':
        contour_mode = cv2.RETR_CCOMP
    elif contour_mode == 'RETR_LIST':
        contour_mode = cv2.RETR_LIST
    elif contour_mode == 'RETR_TREE':
        contour_mode = cv2.RETR_TREE

    # Kiểm tra contour_method
    if contour_method == 'CHAIN_APPROX_SIMPLE':
        contour_method = cv2.CHAIN_APPROX_SIMPLE
    elif contour_method == 'CHAIN_APPROX_NONE':
        contour_method = cv2.CHAIN_APPROX_NONE

    contours, _ = cv2.findContours(image_binary, contour_mode, contour_method)
   
   
    sortting_mode = kwargs.get('sortting_mode', 'Max')
    if sortting_mode == 'Max':
        largest_contour = max(contours, key=cv2.contourArea)
    else:
        largest_contour = min(contours, key=cv2.contourArea)
   
   
    centroid, box_points, _, size = find_min_area_rect(largest_contour)
   
   
    circle_method = kwargs.get('circle_method', 'Normal')
    dp = kwargs.get('dp', 1)
    min_dist = kwargs.get('min_dist', 100)
    param_1 = kwargs.get('param_1', 10)
    param_2 = kwargs.get('param_2', 20)
    min_radius = kwargs.get('min_radius', 30)
    max_radius = kwargs.get('max_radius', 0)
   
    if circle_method == 'Normal':
        circle_method = cv2.HOUGH_GRADIENT
    else:
        circle_method = cv2.HOUGH_GRADIENT_ALT
       
    circles = cv2.HoughCircles(image_binary, circle_method, dp, min_dist, None,
                               param_1, param_2, min_radius, max_radius)
   
    largest_circle = max(circles[0], key=lambda c: c[2])
   
    anchor = (float(largest_circle[0]), float(largest_circle[1]))
   
    vecto_ox = (1, 0)
    vecto_1 = (anchor[0] - centroid[0], -(anchor[1] - centroid[1]))
   
    angle = cal_signed_angle_beetween_two_vecto(vecto_ox, vecto_1)
   
    pose = Pose(
        center=centroid,
        angle=angle,
        anchor=anchor,
        size=size
    )
   
    dst = image.copy()
    drawn = ImageDrawer(dst)
    dst = drawn.draw_box(box_points).get_drawn_image()
   
    return RESULT(
        src=image,
        dst=dst,
        image_binary=image_binary,
        pose=pose
    )      
   
def find_min_area_rect(contour):
    rect = cv2.minAreaRect(contour)
   
    centroid = (rect[0][0], rect[0][1])
    angle = rect[2]
   
    if rect[1][0] > rect[1][1]:
        angle = -angle
       
    if rect[1][0] < rect[1][1]:
        angle = -angle + 90
   
    box_points = cv2.boxPoints(rect)
    box_points = np.int32(box_points)
   
    size = (max(rect[1]), min(rect[1]))

    return centroid, box_points, angle, size 

def run_progress(image, pose_org: Pose, scale, **kwargs):
    result_current = run_preprocessing_image(image, **kwargs)
    
    pose_current = result_current.pose
    
    tcp_pos_current = pose_current.cal_tcp_pos(pose_org)
    
    pose_current.set_tcp_pos(tcp_pos_current)
    
    pose_rotate = pose_current - pose_org
    
    pose_rotate_scale = pose_rotate.scale(scale)
    
    drawn = ImageDrawer(result_current.dst)
    
    (
        drawn
        .draw_point(pose_current.center)
        .draw_point(pose_current.get_tcp_pos(), color=(255, 0, 0))
        .draw_text(str(pose_rotate_scale), (50, 90))
        .draw_point(pose_current.anchor, color=(255, 255, 0))
        .draw_line(pose_current.center, pose_current.anchor, (0, 255, 0))
    )
    
    dst = drawn.get_drawn_image()
    
    return RESULT(
        src=image,
        dst=dst,
        image_binary=result_current.image_binary,
        pose=pose_current
    )


if __name__ == '__main__':
    kwargs = {
        "threshold_value": 40
    }
   
    origin_image = cv2.imread("ImagesTest/Test360/0.bmp")
    result_origin = run_preprocessing_image(origin_image, **kwargs)
    result_origin.pose.set_tcp_pos((result_origin.pose.center[0], result_origin.pose.center[1] + 162))
    save_config(result_origin.pose.data, 'config.json')
    data_config = load_config('config.json')
    pose_org = Pose.from_json_to_pose(data_config)
    scale = pose_org.cal_scale_pixel_to_mm()
   
    image_folder = 'ImagesTest/Test360/'
    image_paths = glob.glob(os.path.join(image_folder, "*.bmp"))
    for image_path in image_paths:
        test_image = cv2.imread(image_path)
        result = run_progress(test_image, pose_org, scale, **kwargs)
        show_image(result.dst)
   