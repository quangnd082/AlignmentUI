import cv2
import glob
import os
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import A4, A3
from datetime import datetime
import json


class Calibration:
    def __init__(self):
        pass

    @staticmethod
    def create_aruco(squares_x, squares_y, target_square_mm, target_marker_mm, aruco_dict, 
                    paper_size="A4", dpi=600, filename="Settings/CalibrationSettings/charuco_board.pdf"):
        target_square_px = (target_square_mm / 25.4) * dpi
        target_marker_px = (target_marker_mm / 25.4) * dpi
        
        square_px = round(target_square_px)
        marker_px = round(target_marker_px)
        
        actual_square_mm = (square_px * 25.4) / dpi
        actual_marker_mm = (marker_px * 25.4) / dpi
        
        print(f"🎯 Target square: {target_square_mm}mm → Actual: {actual_square_mm:.4f}mm (diff: {actual_square_mm-target_square_mm:.4f}mm)")
        print(f"🎯 Target marker: {target_marker_mm}mm → Actual: {actual_marker_mm:.4f}mm (diff: {actual_marker_mm-target_marker_mm:.4f}mm)")

        dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        board = cv2.aruco.CharucoBoard(
            size=(squares_x, squares_y),
            squareLength=square_px,
            markerLength=marker_px,
            dictionary=dictionary
        )

        width_px = squares_x * square_px
        height_px = squares_y * square_px
        img = board.generateImage((width_px, height_px))

        tmp_png = "Settings/CalibrationSettings/_tmp_charuco.png"
        cv2.imwrite(tmp_png, img)

        marker_width_mm = squares_x * target_square_mm
        marker_height_mm = squares_y * target_square_mm

        page_w, page_h = A3 if paper_size.upper() == "A3" else A4

        x_offset = (page_w - marker_width_mm * mm) / 2
        y_offset = (page_h - marker_height_mm * mm) / 2

        c = canvas.Canvas(filename, pagesize=(page_w, page_h))
        c.drawImage(tmp_png, x_offset, y_offset, marker_width_mm * mm, marker_height_mm * mm)
        c.showPage()
        c.save()

        print(f"✅ Đã tạo Charuco {squares_x}x{squares_y}, kích thước CHÍNH XÁC {marker_width_mm}mm x {marker_height_mm}mm, căn giữa trên {paper_size}")
        print(f"📄 Lưu tại: {filename}")
        
        return img
    
    @staticmethod
    def collect_charuco_corners(image_folder, board, dictionary):
        all_corners = []
        all_ids = []
        image_size = None
        drawn_images = []

        for fname in glob.glob(os.path.join(image_folder, '*.jpg')):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]

            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

            if ids is not None and len(corners) > 0:
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)

                if charuco_ids is not None and len(charuco_ids) > 3:
                    all_corners.append(charuco_corners)
                    all_ids.append(charuco_ids)

                    vis_img = img.copy()
                    cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
                    cv2.aruco.drawDetectedCornersCharuco(vis_img, charuco_corners, charuco_ids, (0, 0, 255))
                    drawn_images.append(vis_img)

        return all_corners, all_ids, image_size, drawn_images


    @staticmethod
    def calibrate_camera_from_charuco(all_corners, all_ids, board, image_size,
                                    save_path_matrix=f'Settings/CalibrationSettings/camera_matrix.npy',
                                    save_path_dist=f'Settings/CalibrationSettings/dist_coeffs.npy'):
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_corners,
            charucoIds=all_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None
        )

        print("Calibration successful:", ret)
        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", dist_coeffs)
        
        np.save(save_path_matrix, camera_matrix)
        np.save(save_path_dist, dist_coeffs)

        return camera_matrix, dist_coeffs
    
    @staticmethod
    def setup_world_coordinate_system(img, board, dictionary,
                                    save_path="Settings/CalibrationSettings/homography_matrix.npz",
                                    output_image_path="Settings/CalibrationSettings/world_coordinate_visualization.jpg"):
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        output_img = img.copy()
        
        _corners, _ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
        
        if _ids is not None and len(_corners) > 0:
            ids, corners = Calibration.sort_aruco_by_position(_ids, _corners)
            
            ids = np.array(ids, dtype=np.int32).reshape(-1, 1)
            
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            
            if charuco_ids is not None and len(charuco_ids) > 3:
                # Lấy kích thước board
                squares_x, squares_y = board.getChessboardSize()
                square_length_mm = board.getSquareLength()
                
                # QUAN TRỌNG: Số corners = (squares_x - 1) × (squares_y - 1)
                corners_per_row = squares_x - 1  # Số corner trên mỗi hàng
                corners_per_col = squares_y - 1  # Số corner trên mỗi cột
                
                # Tìm corner với id == 0 làm gốc tọa độ
                origin_idx = None
                origin_point = None
                origin_row = 0
                origin_col = 0
                
                for i, corner_id in enumerate(charuco_ids.flatten()):
                    if corner_id == 0:
                        origin_idx = i
                        origin_point = charuco_corners[i][0]
                        # Corner ID=0 luôn ở vị trí (0, 0) trong grid corners
                        origin_row = 0
                        origin_col = 0
                        break
                
                if origin_idx is None:
                    print("⚠️ Không tìm thấy corner với ID=0, sử dụng corner đầu tiên làm gốc")
                    origin_idx = 0
                    origin_point = charuco_corners[0][0]
                    origin_id = charuco_ids[0].flatten()[0]
                    # Tính row/col cho corner đầu tiên
                    origin_row = origin_id // corners_per_row
                    origin_col = origin_id % corners_per_row
                
                # Tính tọa độ gốc trong hệ mm
                origin_x_mm = origin_col * square_length_mm
                origin_y_mm = origin_row * square_length_mm
                
                object_points = []
                image_points = []
                
                # Dictionary để lưu mapping từ (row, col) đến index trong charuco_corners
                corner_map = {}
                
                # Vẽ các điểm và tọa độ world
                for i, corner_id in enumerate(charuco_ids.flatten()):
                    # Tính row và col từ corner ID
                    # Corner ID = row * corners_per_row + col
                    row = corner_id // corners_per_row
                    col = corner_id % corners_per_row
                    
                    # Lưu mapping
                    corner_map[(row, col)] = i
                    
                    # Tính tọa độ world với gốc tại corner ID=0
                    x_mm = (col * square_length_mm) - origin_x_mm
                    y_mm = (row * square_length_mm) - origin_y_mm
                    
                    object_points.append([x_mm, y_mm, 0])
                    image_points.append(charuco_corners[i][0])
                    
                    # Vẽ điểm trên ảnh
                    point = tuple(charuco_corners[i][0].astype(int))
                    
                    if corner_id == 0:
                        # Vẽ gốc tọa độ màu đỏ, to hơn
                        cv2.circle(output_img, point, 8, (0, 0, 255), -1)
                        cv2.putText(output_img, "ORIGIN (0,0)", 
                                (point[0] + 10, point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        # Vẽ các điểm khác màu xanh lá
                        cv2.circle(output_img, point, 5, (0, 255, 0), -1)
                    
                    # Hiển thị tọa độ world (x, y) tính bằng mm
                    coord_text = f"({x_mm:.0f},{y_mm:.0f})"
                    cv2.putText(output_img, coord_text, 
                            (point[0] + 10, point[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
                    # Hiển thị ID của corner và vị trí (row,col)
                    id_text = f"ID:{corner_id}[{row},{col}]"
                    cv2.putText(output_img, id_text,
                            (point[0] + 10, point[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
                
                # Vẽ các trục tọa độ từ gốc
                if origin_point is not None:
                    origin_pt = tuple(origin_point.astype(int))
                    
                    # Tìm điểm để vẽ trục X (cùng hàng với origin, cột kế tiếp)
                    # Điểm tiếp theo theo trục X: (origin_row, origin_col + 1)
                    x_next_row = origin_row
                    x_next_col = origin_col + 1
                    
                    if (x_next_row, x_next_col) in corner_map:
                        x_idx = corner_map[(x_next_row, x_next_col)]
                        x_axis_point = tuple(charuco_corners[x_idx][0].astype(int))
                        cv2.arrowedLine(output_img, origin_pt, x_axis_point, 
                                    (0, 0, 255), 3, tipLength=0.1)
                        cv2.putText(output_img, "X", x_axis_point,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        # Debug info
                        x_corner_id = charuco_ids[x_idx].flatten()[0]
                        print(f"📍 Trục X: từ corner ID={0} đến corner ID={x_corner_id}")
                    
                    # Tìm điểm để vẽ trục Y (cùng cột với origin, hàng kế tiếp)
                    # Điểm tiếp theo theo trục Y: (origin_row + 1, origin_col)
                    y_next_row = origin_row + 1
                    y_next_col = origin_col
                    
                    if (y_next_row, y_next_col) in corner_map:
                        y_idx = corner_map[(y_next_row, y_next_col)]
                        y_axis_point = tuple(charuco_corners[y_idx][0].astype(int))
                        cv2.arrowedLine(output_img, origin_pt, y_axis_point,
                                    (0, 255, 0), 3, tipLength=0.1)
                        cv2.putText(output_img, "Y", y_axis_point,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Debug info
                        y_corner_id = charuco_ids[y_idx].flatten()[0]
                        print(f"📍 Trục Y: từ corner ID={0} đến corner ID={y_corner_id}")
                
                object_points = np.array(object_points, dtype=np.float32)
                image_points = np.array(image_points, dtype=np.float32)
                
                # Tính homography matrix
                homography_matrix, _ = cv2.findHomography(image_points, object_points[:, :2])
                
                # Lưu homography matrix
                np.savez(save_path, homography_matrix=homography_matrix)
                
                # Lưu ảnh visualization
                cv2.imwrite(output_image_path, output_img)
                
                # Thêm thông tin lên ảnh
                info_text = f"World Coordinate System - Origin at ID=0"
                cv2.putText(output_img, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(output_img, f"Unit: mm | Board: {squares_x}x{squares_y} squares", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(output_img, f"Corners: {corners_per_row}x{corners_per_col} = {corners_per_row*corners_per_col} total", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                print(f"✅ Thiết lập hệ tọa độ thế giới thành công")
                print(f"📋 Board: {squares_x}×{squares_y} squares → {corners_per_row}×{corners_per_col} corners")
                print(f"🎯 Gốc tọa độ đặt tại corner ID=0")
                print(f"📍 Số corner mỗi hàng: {corners_per_row}")
                print(f"📍 Số corner mỗi cột: {corners_per_col}")
                print("Homography Matrix:\n", homography_matrix)
                print(f"📄 Ma trận lưu tại: {save_path}")
                print(f"🖼️ Ảnh visualization lưu tại: {output_image_path}")
                
                return homography_matrix, output_img
            else:
                print("❌ Không đủ ChArUco corners")
                return None, None
        else:
            print("❌ Không phát hiện được ArUco markers")
            return None, None   
        
    @staticmethod
    def sort_aruco_by_position(ids, corners):
        # Kết hợp corners và ids
        combined = list(zip(ids, corners))

        # Sort theo ID
        combined.sort(key=lambda x: x[0])
        
        # Tách lại
        sorted_ids, sorted_corners = zip(*combined)

        return sorted_ids, sorted_corners