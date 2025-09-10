import json
import sys
import os
from unittest import result
from ultralytics import YOLO

from constant import *
sys.path.append('Libs')
import numpy as np
import cv2
from PIL import Image
from Libs import *
from Libs.canvas import Canvas
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication
from types import SimpleNamespace
from datetime import datetime


def set_canvas(canvas: Canvas, mat: np.ndarray):
    mat_rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    
    pixmap = ndarray2pixmap(mat_rgb)
    
    canvas.load_pixmap(pixmap)

def ndarray2pixmap(mat: np.ndarray):
    height, width, channel = mat.shape
    
    bytes_per_line = channel * width
    
    qimage = QImage(mat.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    
    pixmap = QPixmap.fromImage(qimage)
    
    return pixmap

def pixmap2ndarray(pixmap: QPixmap) -> np.ndarray:
    image = pixmap.toImage()
    
    image = image.convertToFormat(QImage.Format_RGB888)
    
    width, height = image.width(), image.height()
    
    ptr = image.bits()
    
    ptr.setsize(image.byteCount())
    
    arr = np.array(ptr).reshape(height, width, 3)
    
    return arr

def convert_path(self, relative_path):
    base_path = os.getcwd()
    absolute_path = os.path.join(base_path, relative_path.replace("/", "\\"))
    return absolute_path

def flatten_namespace(ns):
    result = {}
    for key, value in vars(ns).items():
        if isinstance(value, SimpleNamespace):
            result.update(flatten_namespace(value))
        else:
            result[key] = value
    return result


def pixel_to_world(H, pixel_point):
   H = np.array(H, dtype=np.float32)
   if isinstance(pixel_point, (list, tuple)) and len(pixel_point) == 2:
       x, y = pixel_point
   else:
       raise ValueError("pixel_point phải là tuple (x, y) hoặc list [x, y]")
   point_pixel = np.array([x, y, 1], dtype=np.float32)
   point_world = H @ point_pixel
   point_world = point_world / point_world[2]
   return float(point_world[0]), float(point_world[1])


def world_to_pixel(H, mm_point):
   H = np.array(H, dtype=np.float32)
   if isinstance(mm_point, (list, tuple)) and len(mm_point) == 2:
       x_mm, y_mm = mm_point
   else:
       raise ValueError("mm_point phải là tuple (x_mm, y_mm) hoặc list [x_mm, y_mm]")
   H_inv = np.linalg.inv(H)
   point_world = np.array([x_mm, y_mm, 1], dtype=np.float32)
   point_pixel = H_inv @ point_world
   point_pixel = point_pixel / point_pixel[2]
   return float(point_pixel[0]), float(point_pixel[1])

def transform_dict_points(H, input_dict):
    transformed = {}
    for k, (x, y) in input_dict.items():
        pt = np.array([x, y, 1.0])
        transformed_pt = H @ pt
        transformed_pt /= transformed_pt[2]  # chuẩn hóa (x', y', 1)
        transformed[k] = (float(transformed_pt[0]), float(transformed_pt[1]))
    return transformed

def calc_offset_and_average(dict_center_mm, dict_ref_mm):
    offset_dict = {}
    total_dx = total_dy = 0
    count = 0

    for key in dict_center_mm:
        if key in dict_ref_mm:
            cx, cy = dict_center_mm[key]
            rx, ry = dict_ref_mm[key]
            dx = cx - rx
            dy = cy - ry
            offset_dict[key] = {"dx": dx, "dy": dy}
            total_dx += dx
            total_dy += dy
            count += 1

    avg_dx = total_dx / count if count else 0
    avg_dy = total_dy / count if count else 0

    return offset_dict, (avg_dx, avg_dy)


def undistort_image(image, camera_matrix, dist_coeffs, alpha=1.0, crop=True):
    h, w = image.shape[:2]

    # Tính ma trận camera mới để giảm méo và kiểm soát cắt xén
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha
    )

    # Hiệu chỉnh ảnh
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Cắt ảnh nếu cần
    if crop:
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]

    return undistorted


def is_inside_any_shape(x1, y1, x2, y2, shapes_dict, threshold):
    rect_area = (x2 - x1) * (y2 - y1)
    if rect_area <= 0:
        return None, False
    
    best_key = None
    best_valid = False
    
    for key in shapes_dict:
        sx, sy, sw, sh = shapes_dict[key]
        sx2 = sx + sw
        sy2 = sy + sh
        
        overlap_x1 = max(x1, sx)
        overlap_y1 = max(y1, sy)
        overlap_x2 = min(x2, sx2)
        overlap_y2 = min(y2, sy2)
        
        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
            
            if not best_key:
                best_key = key
            
            if y1 >= sy * threshold and y2 * threshold <= sy2 and overlap_area >= rect_area * threshold:
                return key, True
            
            if not best_valid:
                best_valid = False
    
    return best_key, best_valid

def detect_object(name_model, model_ai, image, shapes, threshold, box_threshold, conf, iou, max_det, agnostic_nms, matrix_H, mode="no_ng"):
    try:
        results = model_ai.predict(source=image, conf=conf, save=False, 
                                   verbose=False, show_labels=True, show_conf=True,
                                   iou=iou, max_det=max_det, agnostic_nms=agnostic_nms)

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = results[0]
        name_classes = result.names
        boxes = result.boxes
        index_box_classes = boxes.cls
        confidences = boxes.conf
        output_img = result.orig_img.copy()

        colors = {
            'purple': (128, 0, 128),
            'red': (0, 0, 255),
            'green': (0, 255, 0),
        }
        
        if 0 not in index_box_classes:
            raise Exception('No box detected')
        
        sort = {}
        for key in shapes.keys():
            if key.lower() != 'box':
                sort[key] = 0
        
        set_shapes = {k: v for k, v in shapes.items() if k.lower() != 'box'}
        draw_info = []
        has_class_1 = False
        box_outside_shapes = False
        object_outside_shapes = False

        shapes_dict_center = {}
        for key, (x, y, w, h) in set_shapes.items():
            center_x = x + w/2
            center_y = y + h/2
            shapes_dict_center[key] = (center_x, center_y)
        
        result_dict_center = {}
        
        for i, (id_box, conf) in enumerate(zip(index_box_classes, confidences)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            x_center, y_center, _, _ = map(float, boxes.xywh[i])
            class_id = int(id_box.cpu().numpy())
            confidence = float(conf.cpu().numpy())
            class_name = name_classes[class_id]
            
            if id_box == 0:
                box_shapes = {k: v for k, v in shapes.items() if k.lower() == 'box'}
                
                if box_shapes:
                    box_key, is_valid = is_inside_any_shape(x1, y1, x2, y2, box_shapes, box_threshold)
                    if box_key and is_valid:
                        draw_info.append({
                            'coords': (x1, y1, x2, y2),
                            'color': colors['purple'],
                            'label': f'{class_name}: {confidence:.2f}',
                            'class_id': class_id
                        })
                    else:
                        box_outside_shapes = True
                        draw_info.append({
                            'coords': (x1, y1, x2, y2),
                            'color': colors['red'],
                            'label': f'{class_name}: {confidence:.2f}',
                            'class_id': class_id
                        })
                else:
                    draw_info.append({
                        'coords': (x1, y1, x2, y2),
                        'color': colors['red'],
                        'label': f'{class_name}: {confidence:.2f}',
                        'class_id': class_id
                    })
                    
            elif id_box == 2:
                key, is_valid = is_inside_any_shape(x1, y1, x2, y2, set_shapes, threshold)
                if not key:
                    object_outside_shapes = True
                    draw_info.append({
                        'coords': (x1, y1, x2, y2),
                        'color': colors['red'],
                        'label': f'{class_name}: {confidence:.2f} (Outside)',
                        'class_id': class_id
                    })
                elif key and is_valid:
                    if sort[key] != 'NG':
                        sort[key] = 1
                    draw_info.append({
                        'coords': (x1, y1, x2, y2),
                        'color': colors['green'],
                        'label': f'{class_name}: {confidence:.2f}',
                        'class_id': class_id
                    })
                    
                    result_dict_center[key] = (float(x_center), float(y_center))
                    
                else:
                    if key:
                        sort[key] = 'NG'
                    draw_info.append({
                        'coords': (x1, y1, x2, y2),
                        'color': colors['red'],
                        'label': f'{class_name}: {confidence:.2f}',
                        'class_id': class_id
                    })
                    
            elif id_box == 1:
                has_class_1 = True
                
                if mode == "bypass":
                    key, is_valid = is_inside_any_shape(x1, y1, x2, y2, set_shapes, threshold)
                    if key and is_valid:
                        if sort[key] != 'NG':
                            sort[key] = 1
                        draw_info.append({
                            'coords': (x1, y1, x2, y2),
                            'color': colors['green'],
                            'label': f'{class_name}: {confidence:.2f}',
                            'class_id': class_id
                        })
                        result_dict_center[key] = (float(x_center), float(y_center))
                    else:
                        if key:
                            sort[key] = 'NG'
                        draw_info.append({
                            'coords': (x1, y1, x2, y2),
                            'color': colors['red'],
                            'label': f'{class_name}: {confidence:.2f}',
                            'class_id': class_id
                        })
                        
                elif mode == "production":
                    key, is_valid = is_inside_any_shape(x1, y1, x2, y2, set_shapes, threshold)
                    if key and is_valid:
                        if sort[key] != 'NG':
                            sort[key] = 0
                        draw_info.append({
                            'coords': (x1, y1, x2, y2),
                            'color': colors['green'],
                            'label': f'{class_name}: {confidence:.2f}',
                            'class_id': class_id
                        })
                        result_dict_center[key] = (float(x_center), float(y_center))
                    else:
                        if key:
                            sort[key] = 'NG'
                        draw_info.append({
                            'coords': (x1, y1, x2, y2),
                            'color': colors['red'],
                            'label': f'{class_name}: {confidence:.2f}',
                            'class_id': class_id
                        })
                        
                else:
                    draw_info.append({
                        'coords': (x1, y1, x2, y2),
                        'color': colors['red'],
                        'label': f'{class_name}: {confidence:.2f}',
                        'class_id': class_id
                    })
        
        for info in draw_info:
            x1, y1, x2, y2 = info['coords']
            color = info['color']
            label = info['label']
            
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 8)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(output_img, (x1, y1 - text_height - 12), (x1 + text_width, y1), color, -1)
            cv2.putText(output_img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        sort_values = list(sort.values())
        
        if mode == "bypass":
            if box_outside_shapes or object_outside_shapes or 'NG' in sort_values:
                label_count_str = 'NG'
            else:
                label_count_str = ''.join(str(x) for x in sort_values)
        elif mode == "production":
            if box_outside_shapes or object_outside_shapes or 'NG' in sort_values:
                label_count_str = 'NG'
            else:
                label_count_str = ''.join(str(x) for x in sort_values)
        else:
            if has_class_1 or box_outside_shapes or object_outside_shapes or 'NG' in sort_values:
                label_count_str = 'NG'
            else:
                label_count_str = ''.join(str(x) for x in sort_values)

        if mode == "bypass":
            if box_outside_shapes:
                raise Exception('Object detected outside of Box area')
            elif object_outside_shapes:
                raise Exception('Object detected outside of defined shapes')
            elif 'NG' in sort_values:
                raise Exception('Please recheck the products')
        elif mode == "production":
            if box_outside_shapes:
                raise Exception('Object detected outside of Box area')
            elif object_outside_shapes:
                raise Exception('Object detected outside of defined shapes')
            elif 'NG' in sort_values:
                raise Exception('Please recheck the products')
        else:
            if has_class_1:
                raise Exception('Please recheck the products')
            elif box_outside_shapes:
                raise Exception('Object detected outside of Box area')
            elif object_outside_shapes:
                raise Exception('Object detected outside of defined shapes')
            elif 'NG' in sort_values:
                raise Exception('Please recheck the products')

        if matrix_H is not None:
            common_keys = shapes_dict_center.keys() & result_dict_center.keys()
            shapes_dict_center = {k: shapes_dict_center[k] for k in common_keys}
            shapes_dict_center = {k: shapes_dict_center[k] for k in sorted(shapes_dict_center)}
            result_dict_center = {k: result_dict_center[k] for k in sorted(result_dict_center)}
            shapes_dict_center_mm = transform_dict_points(matrix_H, shapes_dict_center)
            result_dict_center_mm = transform_dict_points(matrix_H, result_dict_center)
            _, avg_offset  = calc_offset_and_average(shapes_dict_center_mm, result_dict_center_mm)
        else:
            avg_offset = (0, 0)
        
        for key, (cx, cy) in shapes_dict_center.items():
            cv2.circle(output_img, (int(cx), int(cy)), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.putText(output_img, f"S{key}", (int(cx) + 5, int(cy) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        for key, (cx, cy) in result_dict_center.items():
            cv2.circle(output_img, (int(cx), int(cy)), radius=5, color=(0, 0, 255), thickness=-1)

            cv2.putText(output_img, str(key), (int(cx) + 5, int(cy) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return RESULT(
            model_name=name_model,
            src=image,
            dst=output_img,
            ret='PASS',
            label_counts=label_count_str,
            timecheck=current_time,
            error=None,
            offset=avg_offset
        )
    except Exception as e:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return RESULT(
            model_name=name_model,
            src=image,
            dst=output_img,
            ret='FAIL',
            label_counts='NG',
            timecheck=current_time,
            error=str(e),
            offset=(0, 0)
        )

def classify_object(name_model, model_ai, image, shapes, threshold_set, threshold_box):
    try:
        if shapes is None or not shapes:
            raise ValueError("shapes parameter cannot be None or empty")
        
        output_image = image.copy()
        
        # Màu sắc
        colors = {
            'OK': (0, 255, 0),    # Xanh lá
            'NG': (0, 0, 255),    # Đỏ
            '0': (255, 0, 255)    # Tím
        }
        
        # Bỏ qua Box, chỉ lấy shapes khác
        valid_shapes = {k: v for k, v in shapes.items() if k != 'Box'}
        
        # Khởi tạo kết quả
        sort = {}
        for key in valid_shapes.keys():
            sort[key] = 0
        
        # Crop tất cả shapes
        crops_data = []
        for key, (x, y, w, h) in valid_shapes.items():
            if x < 0 or y < 0 or w <= 0 or h <= 0 or y + h > image.shape[0] or x + w > image.shape[1]:
                crops_data.append({'key': key, 'crop': None, 'coords': (x, y, w, h), 'valid': False})
                continue
                
            crop = image[y:y+h, x:x+w]
            if crop is None or crop.size == 0:
                crops_data.append({'key': key, 'crop': None, 'coords': (x, y, w, h), 'valid': False})
                continue
            
            crops_data.append({'key': key, 'crop': crop, 'coords': (x, y, w, h), 'valid': True})
        
        # Predict một lần
        valid_crops = [item['crop'] for item in crops_data if item['valid']]
        all_results = model_ai(source=valid_crops) if valid_crops else []
        
        # Xử lý kết quả và vẽ
        result_idx = 0
        for crop_data in crops_data:
            key = crop_data['key']
            x, y, w, h = crop_data['coords']
            crop = crop_data['crop']
            
            # Mặc định NG
            label = 'NG'
            confidence = 0.0
            color = colors['NG']
            
            if not crop_data['valid']:
                sort[key] = 'NG'
            elif result_idx < len(all_results):
                result = all_results[result_idx]
                result_idx += 1
                
                # Detection
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    best_conf = 0
                    best_class = None
                    
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        if conf >= threshold_set and conf > best_conf:
                            best_conf = conf
                            best_class = result.names[int(box.cls[0])]
                    
                    if best_class:
                        confidence = best_conf
                        if best_class.upper() == 'OK':
                            label = 'OK'
                            color = colors['OK']
                            sort[key] = 1
                        elif best_class.upper() == 'NG':
                            label = 'NG'
                            color = colors['NG']
                            sort[key] = 'NG'
                        elif best_class == '0':
                            label = '0'
                            color = colors['0']
                            sort[key] = 0
                        else:
                            sort[key] = 'NG'
                    else:
                        sort[key] = 'NG'
                
                # Classification
                elif hasattr(result, 'probs') and result.probs is not None:
                    confidence = float(result.probs.top1conf.cpu().numpy())
                    class_name = result.names[result.probs.top1]
                    
                    if confidence >= threshold_set:
                        if class_name.upper() == 'OK':
                            label = 'OK'
                            color = colors['OK']
                            sort[key] = 1
                        elif class_name.upper() == 'NG':
                            label = 'NG'
                            color = colors['NG']
                            sort[key] = 'NG'
                        elif class_name == '0':
                            label = '0'
                            color = colors['0']
                            sort[key] = 0
                        else:
                            sort[key] = 'NG'
                    else:
                        sort[key] = 'NG'
                else:
                    sort[key] = 'NG'
            else:
                sort[key] = 'NG'
            
            # Vẽ lên ảnh
            if crop_data['valid'] and crop is not None:
                # Vẽ border
                crop_with_border = crop.copy()
                h_crop, w_crop = crop.shape[:2]
                
                for i in range(10):
                    cv2.rectangle(crop_with_border, (i, i), (w_crop-1-i, h_crop-1-i), color, 1)
                
                # Vẽ label
                text = f"{label}: {confidence:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(crop_with_border, (10, 10), (10 + text_size[0] + 10, 10 + text_size[1] + 10), color, -1)
                cv2.putText(crop_with_border, text, (15, 10 + text_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Dán vào ảnh gốc
                output_image[y:y+h, x:x+w] = crop_with_border
        
        # Kiểm tra có NG không
        for value in sort.values():
            if value == 'NG':
                raise Exception('NG detected')
        
        # Tạo label string
        label_count_str = ''.join(str(sort[key]) for key in sorted(sort.keys()) if sort[key] != 'NG')
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return RESULT(
            model_name=name_model,
            src=image,
            dst=output_image,
            ret='PASS',
            label_counts=label_count_str,
            timecheck=current_time,
            error=None
        )
        
    except Exception as e:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return RESULT(
            model_name=name_model,
            src=image,
            dst=output_image if 'output_image' in locals() else image,
            ret='FAIL',
            label_counts='NG',
            timecheck=current_time,
            error=str(e)
        )

def detect_barcode(name_model, model_ai, image, shapes, threshold):
    try:
        # Import thư viện cần thiết cho barcode
        import cv2
        import numpy as np
        import zxingcpp
        from datetime import datetime
         
        # Sử dụng model AI để detect vị trí barcode
        results = model_ai.predict(source=image, save=False, verbose=False)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        result = results[0]
        
        output_img = image.copy()
        
        name_classes = result.names
        boxes = result.boxes
        masks = result.masks  # Lấy masks từ segmentation
        index_box_classes = boxes.cls
        confidences = boxes.conf
        sort = shapes.copy()
        
        for key in sort:
            sort[key] = 0
        
        # Tạo list để lưu các mã barcode đọc được
        barcode_data_list = []
        
        def rotate_image(image, angle):
            """Xoay ảnh theo góc cho trước"""
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        def preprocess_for_barcode(img):
            """Tiền xử lý ảnh để cải thiện việc đọc barcode"""
            # Chuyển sang grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Có thể thêm các bước tiền xử lý khác nếu cần
            # Ví dụ: blur để giảm noise
            # gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Hoặc cải thiện contrast
            # gray = cv2.equalizeHist(gray)
            
            return gray
        
        def try_read_barcode_with_rotation(cropped_img):
            """Thử đọc barcode với nhiều góc xoay khác nhau"""
            angles = list(range(-180, 181))  # Thử các góc phổ biến
            
            for angle in angles:
                if angle == 0:
                    test_img = cropped_img
                else:
                    test_img = rotate_image(cropped_img, angle)
                
                # Tiền xử lý ảnh trước khi đọc barcode
                processed_img = preprocess_for_barcode(test_img)
                
                try:
                    # Sử dụng zxing để đọc barcode/QR code
                    results = zxingcpp.read_barcodes(processed_img)
                    if results:  # Nếu đọc được barcode
                        return results
                except Exception as e:
                    # Tiếp tục với góc xoay tiếp theo nếu có lỗi
                    continue
            
            return []  # Không đọc được ở góc nào
        
        def get_cropped_from_mask(image, mask):
            """Crop vùng từ mask segmentation"""
            # Tìm contour từ mask
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, None
            
            # Lấy contour lớn nhất
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Tính minAreaRect để có góc xoay
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # Lấy kích thước của rect
            width = int(rect[1][0])
            height = int(rect[1][1])
            
            # Tạo ma trận biến đổi perspective
            src_pts = box.astype("float32")
            dst_pts = np.array([
                [0, height-1],
                [0, 0],
                [width-1, 0],
                [width-1, height-1]
            ], dtype="float32")
            
            # Áp dụng perspective transform
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, M, (width, height))
            
            return warped, box  # Trả về cả ảnh crop và box để vẽ
        
        def get_minarea_rect_from_bbox(x1, y1, x2, y2):
            """Tạo minAreaRect từ bounding box (fallback khi không có mask)"""
            # Tạo 4 điểm góc của bounding box
            points = np.array([
                [x1, y1],
                [x2, y1], 
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32)
            
            # Tính minAreaRect
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            return box
        
        def draw_minarea_rect(image, box, color=(0, 255, 0), thickness=3):
            """Vẽ minAreaRect lên ảnh"""
            cv2.drawContours(image, [box], -1, color, thickness)
            
            # Vẽ thêm các góc để dễ nhìn
            for point in box:
                cv2.circle(image, tuple(point), 5, color, -1)
        
        # Xử lý từng box được detect
        for i, (box, id_box) in enumerate(zip(boxes, index_box_classes)):
            if id_box == 1:  # Giả sử class 0 là barcode
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                key = is_inside_any_shape(x1, y1, x2, y2, shapes, threshold)
                if key is not None:
                    sort[key] = 1
                    
                    minarea_box = None
                    cropped_img = None
                    
                    # Lấy vùng crop từ mask nếu có
                    if masks is not None and i < len(masks.data):
                        # Resize mask về kích thước ảnh gốc
                        mask = masks.data[i].cpu().numpy()
                        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                        
                        # Crop từ mask segmentation và lấy minAreaRect
                        cropped_img, minarea_box = get_cropped_from_mask(image, mask_resized)
                        
                        if cropped_img is None or minarea_box is None:
                            # Fallback về bounding box nếu mask không hoạt động
                            cropped_img = image[y1:y2, x1:x2]
                            minarea_box = get_minarea_rect_from_bbox(x1, y1, x2, y2)
                    else:
                        # Sử dụng bounding box nếu không có mask
                        cropped_img = image[y1:y2, x1:x2]
                        minarea_box = get_minarea_rect_from_bbox(x1, y1, x2, y2)
                    
                    # Thử đọc barcode với nhiều góc xoay (đã bao gồm grayscale)
                    barcodes = try_read_barcode_with_rotation(cropped_img)
                    
                    # Xử lý từng barcode đọc được
                    for result in barcodes:
                        try:
                            barcode_data = result.text
                            barcode_data_list.append(barcode_data)
                        except Exception as e:
                            # Nếu không đọc được text, thử lấy raw bytes
                            try:
                                barcode_data = str(result.bytes, 'utf-8')
                                barcode_data_list.append(barcode_data)
                            except:
                                # Nếu không decode được, lưu dưới dạng hex
                                if hasattr(result, 'bytes') and result.bytes:
                                    barcode_data = result.bytes.hex()
                                    barcode_data_list.append(f"HEX:{barcode_data}")
                                else:
                                    barcode_data_list.append(f"ERROR_READING_BARCODE")
                    
                    # === VẼ MINAREA RECT THAY VÌ BOUNDING BOX ===
                    if minarea_box is not None:
                        draw_minarea_rect(output_img, minarea_box, color=(0, 255, 0), thickness=3)
                        
                        # Thêm label nếu có barcode data
                        if barcode_data_list:
                            # Tìm điểm cao nhất để đặt text
                            top_point = minarea_box[np.argmin(minarea_box[:, 1])]
                            
                            # Vẽ text với background
                            label = f"Barcode: {len(barcode_data_list)} found"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.6
                            thickness = 2
                            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                            
                            # Background cho text
                            text_x = max(0, top_point[0] - text_width // 2)
                            text_y = max(text_height + 10, top_point[1] - 10)
                            
                            cv2.rectangle(output_img, 
                                        (text_x - 5, text_y - text_height - baseline - 5), 
                                        (text_x + text_width + 5, text_y + baseline + 5), 
                                        (0, 255, 0), -1)
                            
                            # Text
                            cv2.putText(output_img, label, 
                                      (text_x, text_y - baseline), 
                                      font, font_scale, (0, 0, 0), thickness)
        
        # Chuyển list thành chuỗi
        label_count_str = str(barcode_data_list)
        
        return RESULT(
            model_name=name_model,
            src=image,
            dst=output_img,
            ret='PASS',
            label_counts=label_count_str,
            timecheck=current_time,
            error=None
        )
    except Exception as e:
        return RESULT(
            model_name=name_model,
            src=image,
            dst=None,
            ret='FAIL',
            label_counts=None,
            timecheck=current_time,
            error=str(e)
        )

def detect_object_(name_model, model_ai, image, shapes, threshold):
    try:
        results = model_ai.predict(source=image, save=False, verbose=False)


        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        result = results[0]


        output_img = image.copy()


        name_classes = result.names
        boxes = result.boxes
        index_box_classes = boxes.cls
        confidences = boxes.conf
        sort = shapes.copy()

        for key in sort:
            sort[key] = 0

        for box, id_box in zip(boxes, index_box_classes):
            if id_box == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                key = is_inside_any_shape(x1, y1, x2, y2, shapes, threshold)
                if key is not None:
                    sort[key] = 1
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 5)


        label_count_str = result = ''.join(str(x) for x in list(sort.values()))


        return RESULT(
            model_name=name_model,
            src=image,
            dst=output_img,
            ret='PASS',
            label_counts=label_count_str,
            timecheck=current_time,
            error=None
        )
    except Exception as e:
        return RESULT(
            model_name=name_model,
            src=image,
            dst=None,
            ret='FAIL',
            label_counts=None,
            timecheck=current_time,
            error=str(e)
        )
        

def load_model(model_path):
    try:
        model = YOLO(model_path)
        model.predict(np.zeros((640, 640, 3), dtype=np.uint8))
    except Exception as ex:
        print(str(ex))
        model = None
    return model 


if __name__ == '__main__':
    pass
    