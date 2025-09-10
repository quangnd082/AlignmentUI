import cv2
import numpy as np
import json

with open(f'Settings/CalibrationSettings/config.json', 'r', encoding='utf-8') as file:
    config = json.load(file)
    
img = cv2.imread('Images/ImagesCalib/Source/PASS_2025_08_11_14_49_25.jpg')

camera_matrix = np.load('Settings/CalibrationSettings/camera_matrix.npy')
dist_coeffs = np.load('Settings/CalibrationSettings/dist_coeffs.npy')

h, w = img.shape[:2]

# Tính ma trận hiệu chỉnh mới (giúp giảm cắt xén ảnh)
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), alpha=1
)

# Hiệu chỉnh ảnh
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# Cắt ảnh về vùng có thông tin
x, y, w, h = roi
undistorted_img = undistorted_img[y:y+h, x:x+w]

cv2.namedWindow("Img", cv2.WINDOW_FREERATIO)
cv2.imshow("Img", img)

cv2.namedWindow("Undistorted", cv2.WINDOW_FREERATIO)
cv2.imshow("Undistorted", undistorted_img)
cv2.waitKey(0)
