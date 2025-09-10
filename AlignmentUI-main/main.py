import sys
sys.path.append('Libs')
sys.path.append('Libs/cameras/')
import threading
import time
import subprocess
import numpy as np
import cv2
import serial
import serial.tools.list_ports
import json
import os
import copy
import gc
from constant import *
from ultis import *
from Libs import *
from Libs.canvas import Canvas, WindowCanvas
from Libs.cameras import get_camera_devices
from Libs.shape import Shape as MyShape
from Libs.open_camera import Camera
from Libs.Logging import Logger
from Libs.handle_file_json import HandleJsonPBA
from Libs.serial_receiver import SerialReceiver
from Libs.light_controller import LCPController, DCPController
from Libs.server import Server
from Libs.calibration import Calibration
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QActionGroup, QMessageBox, QFileDialog, QHeaderView, QTableWidgetItem, QTableWidget, QLabel
from PyQt5.QtCore import QThread, Qt, pyqtSignal, QStringListModel, QSettings, QByteArray, QTimer, QPointF
from PyQt5.QtGui import QIcon, QImage, QPixmap, QStandardItemModel, QStandardItem, QColor, QBrush, QScreen
from res.UI.mainwindow import Ui_MainWindow
from functools import partial
from types import SimpleNamespace
import re

class MainWindow(QMainWindow):
    showDstSignal = pyqtSignal(Canvas, np.ndarray)
    showResultStatus = pyqtSignal(str)
    checkStarted = pyqtSignal(bool)
    writeLog = pyqtSignal(RESULT)
    setImageTest = pyqtSignal(np.ndarray)
    setImageAruco = pyqtSignal(np.ndarray)
    readComSendData = pyqtSignal(str)
    showEffect = pyqtSignal(int)
    hideEffect = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.setWindowTitle('Trung')
        
        self.init_setup_ui()

        self.init_class_variable()
        self.init_bool_variable()
        self.init_set_property()
        
        self.connect_signal_and_slot()
        
        self.connect_action()
        
        self.set_layout_canvas()
        
        self.upload_model_ai()
        
        self.upload_feature_camera()
        
        self.upload_model_names()
                
        self.refresh_ports()
        
        self.set_arUco_dictionary()
        
        self.loadSettings()
        
        self.load_model_change_teaching()
        
        self.showMaximized()
    
    def init_setup_ui(self):
        self.statusBar().addPermanentWidget(self.ui.progressBar)
        self.ui.progressBar.setVisible(False)
    
    def init_class_variable(self):
        self.camera = Camera()
        self.main_logger = Logger('Main')
        self.handle_file_json = HandleJsonPBA()
        self.com_send_data = SerialReceiver()
        self.server = Server()
        self.light_lcp = LCPController()
        self.light_dcp = DCPController()
        self.log_model = QStandardItemModel()
        self.ui.list_log_view.setModel(self.log_model)
   
    def init_bool_variable(self):
        self.is_open_camera = False
        self.is_connect_server = False
        self.is_open_light = False
        self.is_showing_camera = False
        self.b_start = False
        self.is_resetting = False
        self.is_com_send_data = False
        self.trigger_on = False
        self.started = False
        self.image_capture_test = None
        self.image_capture_aruco = None
        self.light = None
        self.msg_model = None
        self.camera_thread = None
        self.camera_thread_running = False

    def init_set_property(self):
        self.ui.but_stop_auto.setDisabled(True)
        self.ui.but_open_camera_teaching.setProperty("status", "Open")
        self.ui.but_start_camera_teaching.setProperty("status", "Open")
        self.ui.but_open_com_send_data.setProperty("status", "Open")
        self.ui.but_connect_server.setProperty("status", "Open")
        self.ui.but_open_light.setProperty("status", "Open")
    
    def connect_signal_and_slot(self):
        self.ui.but_start_auto.clicked.connect(self.on_click_but_start_auto)
        self.ui.but_stop_auto.clicked.connect(self.on_click_but_stop_auto)
        self.ui.but_add.clicked.connect(self.on_click_but_add_model)
        self.ui.but_del.clicked.connect(self.on_click_but_delete_model)
        self.ui.but_save_teaching.clicked.connect(self.on_click_save_model)
        self.ui.but_open_camera_teaching.clicked.connect(self.on_click_but_open_camera_teaching)
        self.ui.but_open_camera_teaching.clicked.connect(self.on_click_but_close_camera_teaching)
        self.ui.but_open_com_send_data.clicked.connect(self.on_click_but_open_send_data)
        self.ui.but_open_com_send_data.clicked.connect(self.on_click_but_close_send_data)
        self.ui.but_connect_server.clicked.connect(self.on_click_but_connect_server)
        self.ui.but_connect_server.clicked.connect(self.on_click_but_close_server)
        self.ui.but_send_data_tcp.clicked.connect(self.on_click_but_send_data_tcp)
        self.ui.but_start_camera_teaching.clicked.connect(self.on_click_but_start_camera_teaching)
        self.ui.but_capture_teaching.clicked.connect(self.on_click_but_capture_teaching)
        self.ui.but_capture_calib.clicked.connect(self.on_click_but_capture_teaching)
        self.ui.but_open_image_teaching.clicked.connect(self.on_click_but_open_image_teaching)
        self.ui.but_refresh_teaching.clicked.connect(self.refresh_ports)
        self.ui.but_test_teaching.clicked.connect(self.on_click_but_test_teaching)
        self.ui.but_open_light.clicked.connect(self.on_click_but_open_light)
        self.ui.but_open_light.clicked.connect(self.on_click_but_close_light)
        self.ui.but_create_aruco.clicked.connect(self.on_click_but_create_aruco)
        self.ui.but_save_parameters_aruco.clicked.connect(self.on_click_but_save_parameters_aruco)
        self.ui.but_capture_aruco.clicked.connect(self.on_click_but_capture_aruco)
        self.ui.but_clear_images_aruco.clicked.connect(self.on_click_but_clear_images_aruco)
        self.ui.but_detect_marker_aruco.clicked.connect(self.on_click_but_detect_marker)
        self.ui.but_set_center_origin.clicked.connect(self.on_click_but_set_center_origin)
        self.ui.spin_box_channel_value_0.valueChanged.connect(self.update_light_value)
        self.ui.spin_box_channel_value_1.valueChanged.connect(self.update_light_value)
        self.ui.spin_box_channel_value_2.valueChanged.connect(self.update_light_value)
        self.ui.spin_box_channel_value_3.valueChanged.connect(self.update_light_value)
        self.ui.combo_box_model_name_teaching.currentIndexChanged.connect(self.load_model_change_teaching)
        self.server.triggerOn.connect(self.set_trigger_on)
        self.server.server_logger.signalLog.connect(self.add_log_view)
        self.camera.camera_logger.signalLog.connect(self.add_log_view)
        self.main_logger.signalLog.connect(self.add_log_view)
        self.writeLog.connect(self.write_log)
        self.server.sendData.connect(self.read_data_tcp)
        self.showDstSignal.connect(set_canvas)
        self.setImageTest.connect(self.set_image_test)
        self.setImageAruco.connect(self.set_image_aruco)
        self.checkStarted.connect(self.set_ui_auto_start)
        self.showResultStatus.connect(self.update_label_status)
        self.showEffect.connect(self.show_effect)
        self.hideEffect.connect(self.hide_effect)
        self.ui.actionResetLayout.triggered.connect(self.resetLayout)
    
    def connect_action(self):
        self.ui.menuView.addAction(self.ui.dock_widget_image_source.toggleViewAction())
        self.ui.menuView.addAction(self.ui.dock_widget_image_binary.toggleViewAction())
        
        actions = self.ui.menuView.actions()
        if actions:
            ui_action = actions[0]  # Action từ giao diện (ví dụ là action đầu)
            self.ui.menuView.removeAction(ui_action)  # Xóa nó khỏi vị trí ban đầu
            self.ui.menuView.addAction(ui_action)  # Thêm lại xuống cuối
    
    def set_layout_canvas(self):
        self.image_source_canvas = Canvas()
        self.image_binary_canvas = Canvas()
        self.teaching_canvas = Canvas()
        self.auto_canvas = Canvas()
        self.image_input_data_canvas = Canvas()
        self.image_binary_data_canvas = Canvas()
        self.image_aruco_canvas = Canvas()
        self.image_detect_aruco_canvas = Canvas()
        
        self.ui.layout_image_source_canvas.addWidget(WindowCanvas(self.image_source_canvas))
        self.ui.layout_image_binary_canvas.addWidget(WindowCanvas(self.image_binary_canvas))
        self.ui.layout_auto_canvas.addWidget(WindowCanvas(self.auto_canvas))
        self.ui.layout_teaching_canvas.addWidget(WindowCanvas(self.teaching_canvas))
        self.ui.layout_image_output_canvas.addWidget(WindowCanvas(self.image_binary_data_canvas))
        self.ui.layout_image_input_canvas.addWidget(WindowCanvas(self.image_input_data_canvas))
        self.ui.layout_image_capture_aruco.addWidget(WindowCanvas(self.image_aruco_canvas))
        self.ui.layout_image_detect_aruco.addWidget(WindowCanvas(self.image_detect_aruco_canvas))
    
    def on_click_but_start_auto(self):        
        self.ui.combo_box_model_name_teaching.setCurrentIndex(self.ui.combo_box_model_name_auto.currentIndex())
        
        model_name = f'Settings/ModelSettings/{self.ui.combo_box_model_name_auto.currentText()}'
        
        config = self.get_config_auto(model_name)
        
        self.showEffect.emit(70)
        
        threading.Thread(target=self.start_auto, args=(config,), daemon=True).start()
        
    def on_click_but_stop_auto(self):
        self.started = False
        
        self.checkStarted.emit(self.started)
        
        self.close_camera_teaching()
        self.close_light_teaching()
        self.close_server_teaching()
        
        self.stop_process()
        
    
    def on_click_but_reset_auto(self):
        pass
    
    def on_click_but_add_model(self):
        name_model = self.ui.combo_box_model_name_teaching.currentText()
        
        if name_model and self.ui.combo_box_model_name_teaching.findText(name_model) == -1:
            reply = QMessageBox.question(self, 'Question', f'Do you want to add model {name_model}?',
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
            
            self.ui.combo_box_model_name_teaching.addItem(name_model)
            self.ui.combo_box_model_name_auto.addItem(name_model)
            self.ui.combo_box_model_name_data.addItem(name_model)
            
            config = self.get_config_from_ui()
            config_dict = json.loads(json.dumps(config, default=lambda x: vars(x)))
            self.handle_file_json.add(name_model, config_dict)
            
            QMessageBox.information(self, 'Information', f'Model {name_model} has been added successfully',
                                    QMessageBox.StandardButton.Close)
            
            self.ui.combo_box_model_name_teaching.clearEditText()
        elif name_model == '':
            QMessageBox.warning(self, 'Warning', f'Model name not entered yet', QMessageBox.StandardButton.Close)
        else:
            QMessageBox.warning(self, 'Warning', f'Model {name_model} already exists', QMessageBox.StandardButton.Close)

    def on_click_but_delete_model(self):
        name_model = self.ui.combo_box_model_name_teaching.currentText()
        index_name_model = self.ui.combo_box_model_name_teaching.findText(name_model)
        if index_name_model == -1:
            QMessageBox.warning(self, 'Warning', f'Model {name_model} does not exist',
                                QMessageBox.StandardButton.Close)
            return
        elif name_model == 'Default':
            QMessageBox.warning(self, 'Warning', f'Model {name_model} is the default and cannot be deleted',
                                QMessageBox.StandardButton.Close)
            return
        reply = QMessageBox.question(self, 'Delete', f'Do you want to delete {name_model}?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return
        self.handle_file_json.delete(name_model)
        
        index = self.ui.combo_box_model_name_teaching.currentIndex()
        self.ui.combo_box_model_name_auto.removeItem(index)
        self.ui.combo_box_model_name_teaching.removeItem(index)
        
        QMessageBox.information(self, 'Information', f'Model {name_model} has been deleted',
                                QMessageBox.StandardButton.Close)

    def on_click_save_model(self):
        name_model = self.ui.combo_box_model_name_teaching.currentText()
        
        index_name_model = self.ui.combo_box_model_name_teaching.findText(name_model)
        
        if index_name_model == -1:
            QMessageBox.warning(self, 'Warning', f'Model {name_model} does not exist',
                                QMessageBox.StandardButton.Close)
            return
        
        config = self.get_config_from_ui()
        config_dict = json.loads(json.dumps(config, default=lambda x: vars(x)))
        
        reply = QMessageBox.question(self, 'Question', f'Do you want to save the model {name_model}?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.No:
            return
        
        self.handle_file_json.save(name_model, config_dict)
        
        QMessageBox.information(self, 'Information', f'Model {name_model} has been successfully updated',
                                QMessageBox.StandardButton.Close)
    
    def on_click_but_open_camera_teaching(self):
        if self.ui.but_open_camera_teaching.property("status") == "Open":
            if self.open_camera_teaching():
                self.ui.but_open_camera_teaching.setProperty("status", "Close")
                self.ui.but_open_camera_teaching.setText('Close')
            else:
                return
            
        self.style().unpolish(self.ui.but_open_camera_teaching)
        self.style().polish(self.ui.but_open_camera_teaching)
        self.style().unpolish(self.ui.but_start_camera_teaching)
        self.style().polish(self.ui.but_start_camera_teaching)
        
        self.ui.but_open_camera_teaching.clicked.disconnect()
        self.ui.but_open_camera_teaching.clicked.connect(self.on_click_but_close_camera_teaching)

    def on_click_but_close_camera_teaching(self):
        if self.ui.but_open_camera_teaching.property("status") == "Close":
            self.close_camera_teaching()
            self.ui.but_open_camera_teaching.setText('Open camera')
            self.ui.but_open_camera_teaching.setProperty("status", "Open")
            self.ui.but_start_camera_teaching.setProperty("status", "Open")
            self.ui.but_start_camera_teaching.setText('Start')
        
        self.style().unpolish(self.ui.but_open_camera_teaching)
        self.style().polish(self.ui.but_open_camera_teaching)
        self.style().unpolish(self.ui.but_start_camera_teaching)
        self.style().polish(self.ui.but_start_camera_teaching)
        
        self.ui.but_open_camera_teaching.clicked.disconnect()
        self.ui.but_open_camera_teaching.clicked.connect(self.on_click_but_open_camera_teaching)
    
    def on_click_but_start_camera_teaching(self):
        if self.ui.but_start_camera_teaching.property("status") == "Open":
            if not self.is_open_camera:
                self.main_logger.warning('Please open camera first')
                return
                
            self.ui.but_start_camera_teaching.setProperty("status", "Close")
            self.ui.but_start_camera_teaching.setText('Stop')
            self.start_camera_teaching()
        else:
            self.ui.but_start_camera_teaching.setText('Start')
            self.ui.but_start_camera_teaching.setProperty("status", "Open")
            self.stop_camera_teaching()
                
        self.style().unpolish(self.ui.but_start_camera_teaching)
        self.style().polish(self.ui.but_start_camera_teaching)
    
    def on_click_but_capture_teaching(self):
        try:
            image_capture = None
            if self.is_showing_camera:
                if self.mat is not None:
                    image_capture = self.mat.copy()
                else:
                    QMessageBox.warning(self, 'Warning', 'No image currently displayed to capture')
                    return
                    
            elif self.is_open_camera:
                image_capture = self.camera.get_frame()
                if image_capture is None:
                    QMessageBox.warning(self, 'Warning', 'Failed to capture image from camera')
                    return                
            else:
                QMessageBox.information(self, 'Info', 'Please open camera or load an image first')
                return
            
            if image_capture is None or image_capture.size == 0:
                QMessageBox.critical(self, 'Error', 'Failed to capture valid image')
                return
            
            config = self.get_config_from_ui()
            if config.calibration.use_calib:
                camera_file = 'Settings/CalibrationSettings/camera_matrix.npy'
                dist_file = 'Settings/CalibrationSettings/dist_coeffs.npy'

                camera_matrix = dist_coeffs = None

                if os.path.exists(camera_file):
                    camera_matrix = np.load(camera_file)
                else:
                    QMessageBox.critical(self, 'Calibration Error', 'Camera matrix file not found. Please check calibration.')
                    return

                if os.path.exists(dist_file):
                    dist_coeffs = np.load(dist_file)
                else:
                    QMessageBox.critical(self, 'Calibration Error', 'Distortion coefficients file not found. Please check calibration.')
                    return
                
                if camera_matrix is None or dist_coeffs is None:
                    QMessageBox.critical(self, 'Calibration Error', 'Invalid calibration matrices. Please check calibration.')
                    return
                
                try:
                    image_capture = undistort_image(image_capture, camera_matrix, dist_coeffs)
                    if image_capture is None:
                        QMessageBox.critical(self, 'Calibration Error', 'Failed to undistort image. Please check calibration matrices.')
                        return
                except Exception as undistort_ex:
                    QMessageBox.critical(self, 'Calibration Error', f'Undistortion failed: {str(undistort_ex)}')
                    return
                
            today_folder = datetime.now().strftime('%Y_%m_%d')
            model_name = self.ui.combo_box_model_name_teaching.currentText()
            
            if not model_name:
                QMessageBox.warning(self, 'Warning', 'Please select a model name first')
                return
            
            self.main_logger.log_image(
                model_name,
                image_capture,
                image_folder=f'Images/ImageCapture/{today_folder}/{model_name}'
            )
            
            set_canvas(self.teaching_canvas, image_capture)
            self.setImageTest.emit(image_capture)
            
        except Exception as ex:
            QMessageBox.critical(self, 'Error', f'Capture failed: {str(ex)}')
    
    def on_click_but_open_image_teaching(self):
        if self.ui.but_start_camera_teaching.property("status") == "Close":
            self.is_showing_camera = False
            self.ui.but_start_camera_teaching.setProperty("status", "Open")
            self.ui.but_start_camera_teaching.setText('Start')
            self.style().unpolish(self.ui.but_start_camera_teaching)
            self.style().polish(self.ui.but_start_camera_teaching)
        
        options = QFileDialog.Options()
        file_open_image, _ = QFileDialog.getOpenFileName(self, "Choose image", "", 
                                                  "Images (*.png *.jpg *.jpeg *.bmp *.gif)", 
                                                  options=options)
        if file_open_image:
            image = cv2.imread(file_open_image)
            set_canvas(self.teaching_canvas, image)
            self.setImageTest.emit(image)
    
    def on_click_but_test_teaching(self):
        try:
            if self.image_capture_test is None:
                QMessageBox.warning(self, 'Warning', 'No test image available. Please capture an image first.')
                return
                
            config = self.get_config_from_ui()
            
            name_model = self.ui.combo_box_model_name_teaching.currentText()
            name_model_ai = self.ui.combo_box_model_ai.currentText()
            name_classify_ai = self.ui.combo_box_classify_ai.currentText()
            use_classify = self.ui.check_box_use_classify.isChecked()
            
            model_ai = YOLO(f'res/ModelAI/{name_model_ai}')
            classify_ai = YOLO(f'res/ModelAI/{name_classify_ai}')
            
            threshold_set = self.ui.spin_box_threshold_set.value() / 100
            threshold_box = self.ui.spin_box_threshold_box.value() / 100
            conf = self.ui.spin_box_confidence.value()
            iou = self.ui.spin_box_iou.value()
            max_det = self.ui.spin_box_max_det.value()
            agnostic_nms = self.ui.check_box_agnostic_nms.isChecked()
            
            image_to_process = self.image_capture_test
            matrix_H = None
            mode = self.get_detection_mode(config)
            
            if config.calibration.use_calib:
                homography_file = 'Settings/CalibrationSettings/homography_matrix.npz'
                
                try:
                    if os.path.exists(homography_file):
                        homography_data = np.load(homography_file)
                        matrix_H = homography_data.get('homography_matrix')
                        
                        if matrix_H is None:
                            raise Exception('Invalid homography matrix')
                    else:
                        raise Exception('Homography matrix file not found')
                        
                except Exception as e:
                    raise Exception(f'Failed to load homography matrix: {str(e)}')
            
            if use_classify:
                result = classify_object(name_model, classify_ai, image_to_process, 
                                    config.shapes, threshold_set, threshold_box)
            else:
                result = detect_object(name_model, model_ai, image_to_process, 
                                    config.shapes, threshold_set, threshold_box, 
                                    conf, iou, max_det, agnostic_nms, matrix_H, mode)
            
            avg_offset_x, avg_offset_y = result.offset
            dx = dy = 0
            
            if config.calibration.use_calib and result.label_counts != 'NG' and not re.fullmatch(r"0+", result.label_counts):
                
                center_x_new, center_y_new = self.process_center(
                    avg_offset_x, avg_offset_y,
                    self.ui.check_box_swap_xy.isChecked(),
                    self.ui.check_box_negative_x.isChecked(),
                    self.ui.check_box_negative_y.isChecked()
                )
                
                self.ui.spin_box_center_x.setValue(center_x_new)
                self.ui.spin_box_center_y.setValue(center_y_new)
                
                org_x = config.calibration.center_origin_x
                org_y = config.calibration.center_origin_y
                
                scale_x = config.calibration.scale_x
                scale_y = config.calibration.scale_y
                
                dx = (center_x_new - org_x) * scale_x
                dy = (center_y_new - org_y) * scale_y   
                
                self.ui.label_dx.setText(str(dx))
                self.ui.label_dy.setText(str(dy))
            
            today_folder = datetime.now().strftime('%Y_%m_%d')
            self.main_logger.log_image(
                name_model,
                result.dst,
                image_folder=f'Images/ImageTest/{today_folder}/{name_model}'
            )
            
            set_canvas(self.teaching_canvas, result.dst)
            
            output_str = f"{result.label_counts}: {dx}, {dy}"
            print(output_str)
            
        except Exception as ex:
            error_msg = f"Test failed: {str(ex)}"
            print(error_msg)
            QMessageBox.critical(self, 'Test Error', error_msg)
    
    def on_click_but_connect_server(self):
        if self.ui.but_connect_server.property("status") == "Open":
            self.ui.but_connect_server.setProperty("status", "Close")
            self.ui.but_connect_server.setText('Close camera')
            self.connect_server_teaching()
            
        self.style().unpolish(self.ui.but_connect_server)
        self.style().polish(self.ui.but_connect_server)
        
        self.ui.but_connect_server.clicked.disconnect()
        self.ui.but_connect_server.clicked.connect(self.on_click_but_close_server)
        
    def on_click_but_close_server(self):        
        if self.ui.but_connect_server.property("status") == "Close":
            self.ui.but_connect_server.setText('Open')
            self.ui.but_connect_server.setProperty("status", "Open")
            self.close_server_teaching()
            
        self.style().unpolish(self.ui.but_connect_server)
        self.style().polish(self.ui.but_connect_server)
        
        self.ui.but_connect_server.clicked.disconnect()
        self.ui.but_connect_server.clicked.connect(self.on_click_but_connect_server)

    def on_click_but_open_send_data(self):
        if self.ui.but_open_com_send_data.property("status") == "Open":
            if self.open_send_data_teaching():
                self.ui.but_open_com_send_data.setProperty("status", "Close")
                self.ui.but_open_com_send_data.setText('Close')
            else:
                return
            
        self.style().unpolish(self.ui.but_open_com_send_data)
        self.style().polish(self.ui.but_open_com_send_data)
        
        self.ui.but_open_com_send_data.clicked.disconnect()
        self.ui.but_open_com_send_data.clicked.connect(self.on_click_but_close_send_data)
    
    def on_click_but_close_send_data(self):
        if self.ui.but_open_com_send_data.property("status") == "Close":
            if self.close_send_data_teaching():
                self.ui.but_open_com_send_data.setText('Open')
                self.ui.but_open_com_send_data.setProperty("status", "Open")
            else:
                return
            
        self.style().unpolish(self.ui.but_open_com_send_data)
        self.style().polish(self.ui.but_open_com_send_data)
        
        self.ui.but_open_com_send_data.clicked.disconnect()
        self.ui.but_open_com_send_data.clicked.connect(self.on_click_but_open_send_data)
    
    def on_click_but_open_light(self):
        if self.ui.but_open_light.property("status") == "Open":
            if self.open_light_teaching():
                self.ui.but_open_light.setProperty("status", "Close")
                self.ui.but_open_light.setText('Close')
            else:
                return
        
        channel_0 = self.ui.spin_box_channel_value_0.value()
        channel_1 = self.ui.spin_box_channel_value_1.value()
        channel_2 = self.ui.spin_box_channel_value_2.value()
        channel_3 = self.ui.spin_box_channel_value_3.value()
        
        self.set_light_value(channel_0, channel_1, channel_2, channel_3)

        self.style().unpolish(self.ui.but_open_light)
        self.style().polish(self.ui.but_open_light)
        
        self.ui.but_open_light.clicked.disconnect()
        self.ui.but_open_light.clicked.connect(self.on_click_but_close_light)
    
    def on_click_but_close_light(self):
        if self.ui.but_open_light.property("status") == "Close":
            if self.close_light_teaching():
                self.ui.but_open_light.setText('Open')
                self.ui.but_open_light.setProperty("status", "Open")
            else:
                return
            
        self.style().unpolish(self.ui.but_open_light)
        self.style().polish(self.ui.but_open_light)

        self.ui.but_open_light.clicked.disconnect()
        self.ui.but_open_light.clicked.connect(self.on_click_but_open_light)
    
    def on_click_but_create_aruco(self):
        squares_x = self.ui.spin_box_square_x.value()
        squares_y = self.ui.spin_box_square_y.value()
        square_length = self.ui.spin_box_square_length.value()
        marker_length = self.ui.spin_box_marker_length.value()
        aruco_dict = aruco_dict_mapping[self.ui.combo_box_aruco_dict.currentText()]
        
        reply = QMessageBox.question(self, 'Question', 'Are you want to create Aruco?', 
                                     QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            
            if squares_x <= 1 or squares_y <= 1 or square_length <= 1 or marker_length <= 0:
                QMessageBox.warning(self, 'Warning', 'Please check setting parameters')
                return
            
            elif square_length <= marker_length:
                QMessageBox.warning(self, 'Warning', 'Square length cannot be less than Marker length')
                return
            
            image_aruco = Calibration.create_aruco(squares_x, squares_y, square_length, marker_length, aruco_dict)
            set_canvas(self.image_aruco_canvas, image_aruco)
    
    def on_click_but_save_parameters_aruco(self):
        try:
            config = {
                'squares_x': self.ui.spin_box_square_x.value(),
                'squares_y': self.ui.spin_box_square_y.value(),
                'square_length': self.ui.spin_box_square_length.value(),
                'marker_length': self.ui.spin_box_marker_length.value(),
                'aruco_dict': aruco_dict_mapping[self.ui.combo_box_aruco_dict.currentText()],
            }
            reply = QMessageBox.question(self, 'Question', 'Are You want to save parameters?', 
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.handle_file_json.add('CalibrationSettings', config, 'Settings')
                QMessageBox.information(self, 'Information', 'Parameters has been saved')
                
        except Exception as e:
            QMessageBox.warning(self, 'Warning', str(e))
            print(e)
    
    def on_click_but_clear_images_aruco(self):
        reply = QMessageBox.question(self, 'Question', 'Are You want to clear all images?', 
                                         QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.handle_file_json.delete('ImagesCalib', 'Images')
            QMessageBox.information(self, 'Information', 'Successfully clear all images')
    
    def on_click_but_detect_marker(self):
        try:
            self.handle_file_json.delete('Destination', 'Images/ImagesCalib')
            squares_x = self.ui.spin_box_square_x.value()
            squares_y = self.ui.spin_box_square_y.value()
            square_length = self.ui.spin_box_square_length.value()
            marker_length = self.ui.spin_box_marker_length.value()
            aruco_dict = aruco_dict_mapping[self.ui.combo_box_aruco_dict.currentText()]
            
            dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
            
            board = cv2.aruco.CharucoBoard(
                size=(squares_x, squares_y),
                squareLength=square_length,
                markerLength=marker_length,
                dictionary=dictionary
            )
            
            image_folder = 'Images/ImagesCalib/Source'
            
            reply = QMessageBox.question(self, 'Question', 'Are You want to detect markers?', 
                                            QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                all_corners, all_ids, image_size, drawn_images = Calibration.collect_charuco_corners(
                    image_folder=image_folder, 
                    board=board, 
                    dictionary=dictionary
                )
                
                if len(drawn_images) > 0:
                    for dst in drawn_images:
                        self.main_logger.log_image(
                            'Calib', 
                            dst,
                            image_folder=f'Images/ImagesCalib/Destination'
                        )
                    
                    print(f'Get successfully corners, ids of {len(drawn_images)} picture')
                    
                    QMessageBox.information(self, 'Information', f'Get successfully corners, ids of {len(drawn_images)} picture')
                    
                    camera_matrix, dist_coeffs = Calibration.calibrate_camera_from_charuco(
                        all_corners=all_corners, 
                        all_ids=all_ids, 
                        board=board, 
                        image_size=image_size
                    )
                    
                    img_undistort = undistort_image(self.image_capture_aruco, camera_matrix, dist_coeffs)
                    
                    homography_matrix, image_world_point = Calibration.setup_world_coordinate_system(
                        img=img_undistort,
                        board=board,
                        dictionary=dictionary
                    )
                    
                    set_canvas(self.image_detect_aruco_canvas, image_world_point)
                    
                    
        except Exception as e:
            QMessageBox.warning(self, 'Warning', str(e))
            print(e)
            
    def on_click_but_capture_aruco(self):
        try:
            if self.is_showing_camera:
                image_capture = self.mat.copy()
            elif self.is_open_camera:
                image_capture = self.camera.get_frame()
            else:
                QMessageBox.warning(self, 'Warning', 'Please open camera first')
                return
            
            set_canvas(self.image_aruco_canvas, image_capture)
            self.setImageAruco.emit(image_capture)
            self.main_logger.log_image(
                'Calib', 
                image_capture,
                image_folder=f'Images/ImagesCalib/Source'
            )
        except Exception as e:
            print(e)
            return
    
    def on_click_but_set_center_origin(self): 
        reply = QMessageBox.question(self, 'Question', 'Do you want to set center origin ?', 
                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            center_x = self.ui.spin_box_center_x.value()
            center_y = self.ui.spin_box_center_y.value()
    
            self.ui.label_center_origin_x.setText(str(center_x))
            self.ui.label_center_origin_y.setText(str(center_y))
            QMessageBox.information(self, 'Information', 'Center origin has been successfully updated')
    
    def set_config_auto(self, config):
        self.set_shapes(self.auto_canvas, config.shapes)
        
        self.light_value = {
            'channel_0': config.hardware.lighting.channel_value_0,
            'channel_1': config.hardware.lighting.channel_value_1,
            'channel_2': config.hardware.lighting.channel_value_2,
            'channel_3': config.hardware.lighting.channel_value_3,
            'delay': config.hardware.lighting.delay
        }
        
        self.name_model = config.name_model
        self.model_ai = load_model(f'res/ModelAI/{config.model_ai.model_name}')
        self.classify_ai = load_model(f'res/ModelAI/{config.model_ai.classify_name}')
        
        self.shapes = config.shapes
        
        self.threshold_set = config.model_ai.threshold_set / 100
        self.threshold_box = config.model_ai.threshold_box / 100
        
        self.conf = config.model_ai.conf
        self.iou = config.model_ai.iou
        self.max_det = config.model_ai.max_det
        self.agnostic_nms = config.model_ai.agnostic_nms
        self.load_calibration_data(config)
        self.hideEffect.emit(200)

    def thread_loop_auto(self, config):
        self.set_config_auto(config)
        
        step = Step.READ_TRIGGER
        result = None
        mat_check = None
        error = None
        loop_count = 0
        org_x = config.calibration.center_origin_x
        org_y = config.calibration.center_origin_y
        
        scale_x = config.calibration.scale_x
        scale_y = config.calibration.scale_y
        
        mode = self.get_detection_mode(config)
        
        while True:
            try:
                if step == Step.READ_TRIGGER:
                    if self.trigger_on:
                        self.main_logger.info(step)
                        step = Step.ON_LIGHTING
            ### ----------------------------------------- ###
            #Thêm vào khi muốn change model tự động
                # elif step == Step.LOAD_CONFIG:
                #     if config.name_model == self.msg_model:
                #         pass
                #     else:
                #         index = self.ui.combo_box_model_name_auto.findText(self.msg_model)
                        
                #         if index != -1:
                #             self.ui.combo_box_model_name_auto.setCurrentIndex(index)
                #         else:
                #             raise Exception('Model undefined, please try again')
                        
                #         config = self.get_config_auto(f'Settings/ModelSettings/{self.msg_model}')
                        
                #         self.set_config_auto(config)
                        
                #     step = Step.PREPROCESS
                
            ### ----------------------------------------- ###
            
            ### ----------------------------------------- ###
            #Thêm vào khi có đèn
            
                elif step == Step.ON_LIGHTING:
                    self.showResultStatus.emit(StepResult.WAIT.value)
                    self.main_logger.info(step)
                    self.set_light_value(**self.light_value)
                    step = Step.PREPROCESS
                
            ### ----------------------------------------- ###

                elif step == Step.PREPROCESS:
                    self.main_logger.info(step)
                    time.sleep(config.hardware.camera.delay / 1000)
                    mat_check = self.camera.get_frame()
                    if mat_check is None:
                        raise Exception('Failed to get frame')
                    step = Step.OFF_LIGHTING

                elif step == Step.OFF_LIGHTING:
                    self.main_logger.info(step)
                    self.set_light_value()
                    step = Step.UNDISTORT_CAMERA
                
                elif step == step.UNDISTORT_CAMERA:
                    if config.calibration.use_calib:
                        if not self.calibration_loaded:
                            raise Exception('Calibration data not loaded. Please check calibration files.')
                    
                        mat_check = undistort_image(mat_check, self.camera_matrix, self.dist_coeffs)
                    
                    step = Step.VISION_DETECTION
                    
                            
                elif step == Step.VISION_DETECTION:
                    self.main_logger.info(step)
                    result = detect_object(self.name_model, self.model_ai, mat_check, 
                                           self.shapes, self.threshold_set, self.threshold_box, 
                                           self.conf, self.iou, self.max_det, self.agnostic_nms, self.matrix_H, mode)
                    if config.model_ai.use_classify:
                        if result.label_counts != "NG":
                            pass
                        else:
                            result = classify_object(self.name_model, self.classify_ai, mat_check, self.shapes, self.threshold_set, self.threshold_box)
                    if config.calibration.use_calib and result.label_counts != "NG" and not re.fullmatch(r"0+", result.label_counts):           
                        avg_center_x, avg_center_y = result.offset
                        center_x_new, center_y_new = self.process_center(avg_center_x, avg_center_y, 
                                                                        config.calibration.swap_xy, 
                                                                        config.calibration.negative_x, 
                                                                        config.calibration.negative_y
                                                                    )
                        dx = (center_x_new - org_x) * scale_x
                        dy = (center_y_new - org_y) * scale_y
                    else:
                        dx, dy = 0, 0
                        
                    self.showDstSignal.emit(self.auto_canvas, result.dst)
                        
                    output_str = f"{result.label_counts}: {dx}, {dy}"
                    
                    step = Step.OUTPUT
                        
                elif step == Step.OUTPUT:
                    self.main_logger.info(step)
                    self.showResultStatus.emit(result.ret)
                    self.server.send_data_to_all_clients(output_str)
                    step = Step.WRITE_LOG
                
                elif step == Step.WRITE_LOG:
                    try:
                        self.main_logger.info(step)
                        self.writeLog.emit(result)
                        step = Step.RELEASE
                    except Exception as ex:
                        self.main_logger.error(str(ex))
                        step = Step.RELEASE
                
                elif step == Step.RELEASE:
                    self.main_logger.info(step)
                    mat_check = None
                    result = None
                    dx = 0
                    dy = 0
                    center_x, center_y = 0, 0
                    self.set_trigger_off()
                    loop_count += 1
                    if loop_count % 10 == 0:
                        gc.collect()
                    step = Step.READ_TRIGGER
                
                elif step == Step.ERROR:
                    self.main_logger.error(step)
                    if error:
                        self.showResultStatus.emit(StepResult.FAIL.value)
                    step = Step.WRITE_LOG

            except Exception as ex:
                error = str(ex)
                self.main_logger.error(f"Error in step {step}: {error}")
                step = Step.ERROR
            
            if not self.b_start:
                self.set_shapes(self.auto_canvas, {})
                gc.collect()
                break

            time.sleep(0.01)
    
    def load_calibration_data(self, config):
        self.matrix_H = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_loaded = False
        
        if config.calibration.use_calib:
            calib_files = {
                'homography': 'Settings/CalibrationSettings/homography_matrix.npz',
                'camera': 'Settings/CalibrationSettings/camera_matrix.npy',
                'dist': 'Settings/CalibrationSettings/dist_coeffs.npy'
            }
            
            try:
                homography_data = np.load(calib_files['homography'])
                self.matrix_H = homography_data.get('homography_matrix')
                
                self.camera_matrix = np.load(calib_files['camera'])
                self.dist_coeffs = np.load(calib_files['dist'])
                
                if self.matrix_H is None or self.camera_matrix is None or self.dist_coeffs is None:
                    raise Exception('Invalid calibration data')
                
                self.calibration_loaded = True
                self.main_logger.info("Calibration data loaded successfully")
                
            except (FileNotFoundError, KeyError) as e:
                self.main_logger.error(f"Failed to load calibration: {e}")
                self.calibration_loaded = False
    
    def start_auto(self, config):
        self.ui.but_start_auto.setDisabled(True)
        
        self.open_camera_teaching(config)
        self.connect_server_teaching(config)
        self.open_light_teaching(config)

        if self.is_open_camera and self.is_connect_server and self.is_open_light:
            self.started = True
            self.start_process(config)
            self.checkStarted.emit(self.started)
            
        else:
            self.ui.but_start_auto.setDisabled(False)
            
            self.close_camera_teaching()
            self.close_server_teaching()
            self.close_light_teaching()
            
            self.hideEffect.emit(200)
    
    def set_ui_auto_start(self, started):
        if started:
            self.set_disable_auto(True)
            self.ui.but_start_auto.setDisabled(True)
            self.ui.but_stop_auto.setDisabled(False)
        else:
            self.set_disable_auto(False)
            self.ui.but_start_auto.setDisabled(False)
            self.ui.but_stop_auto.setDisabled(True)
            
    
    def start_process(self, config):
        time.sleep(0.2)
        threading.Thread(target=self.thread_loop_auto, daemon=True, args=(config,)).start()
        self.b_start = True
    
    def stop_process(self):
        self.b_start = False
        self.set_trigger_off()
    
    def set_image_test(self, image_capture):
        self.image_capture_test = image_capture
    
    def set_image_aruco(self, image_capture):
        self.image_capture_aruco = image_capture
    
    def open_camera_teaching(self, config=None):
        # Nếu camera đã mở, đóng nó trước
        if self.is_open_camera:
            self.close_camera_teaching()
            time.sleep(0.2)  # Đợi một chút để đảm bảo camera được đóng hoàn toàn
        
        if config is None:
            config = self.get_config_from_ui()
        
        camera_config = config.hardware.camera
        id_camera = camera_config.id
        feature_camera = f'res/CameraFeature/{camera_config.feature}'
        
        if feature_camera == 'res/CameraFeature/Default':
            feature_camera = ''
        
        try:
            success = self.camera.open_camera(camera_config.name, config={
                'id': id_camera,
                'feature': ''
            })
            
            if not success:
                self.main_logger.error('Failed to open camera')
                return False
            
            self.is_open_camera = True
            self.ui.but_start_camera_teaching.setDisabled(False)
            self.main_logger.info('Camera opened successfully')
            return True
            
        except Exception as ex:
            self.main_logger.error(f'Error opening camera: {ex}')
            return False

    def close_camera_teaching(self):
        if not self.is_open_camera:
            return True
        
        # Dừng thread hiển thị camera trước
        self.stop_camera_teaching()
        
        # Đợi thread kết thúc hoàn toàn
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread_running = False
            self.camera_thread.join(timeout=2.0)  # Đợi tối đa 2 giây
        
        # Đóng camera
        try:
            self.camera.close_camera()
        except Exception as ex:
            self.main_logger.error(f'Error closing camera: {ex}')
        
        # Reset trạng thái
        self.is_open_camera = False
        self.is_showing_camera = False
        self.camera_thread = None
        self.camera_thread_running = False
        
        self.ui.but_start_camera_teaching.setDisabled(True)
        return True
    
    def loop_live_camera(self, camera):
        try:
            while self.camera_thread_running and self.is_showing_camera:
                if not self.is_open_camera:
                    break
                    
                frame = camera.get_frame()
                self.mat = frame
                if frame is not None:
                    # Tạo copy để tránh reference issues
                    frame_copy = frame.copy()
                    self.showDstSignal.emit(self.teaching_canvas, frame_copy)
                    # Giải phóng frame gốc
                    del frame
                else:
                    break
                    
                time.sleep(0.04)
        except Exception as ex:
            self.main_logger.error(f'Error in camera loop: {ex}')
        finally:
            # Cleanup
            if hasattr(self, 'mat'):
                del self.mat
            self.camera_thread_running = False
    
    def start_camera_teaching(self):
        try:
            # Dừng thread cũ nếu còn chạy
            if self.camera_thread and self.camera_thread.is_alive():
                self.stop_camera_teaching()
                time.sleep(0.1)  # Đợi một chút
            
            # Kiểm tra camera có mở không
            if not self.is_open_camera:
                self.main_logger.error('Camera is not opened')
                return
            
            self.is_showing_camera = True
            self.camera_thread_running = True
            
            # Tạo thread mới
            self.camera_thread = threading.Thread(
                target=self.loop_live_camera, 
                args=(self.camera,), 
                daemon=True
            )
            self.camera_thread.start()
            
            self.main_logger.info('Camera is starting')
            
        except Exception as ex:
            self.main_logger.error(f'Failed to start camera. Error: {ex}')
            self.is_showing_camera = False
            self.camera_thread_running = False
    
    def stop_camera_teaching(self):
        try:
            # Dừng thread hiển thị
            self.camera_thread_running = False
            self.is_showing_camera = False
            
            # Đợi thread kết thúc
            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.join(timeout=1.0)
            
            self.main_logger.info('Camera was stopped')
            
        except Exception as ex:
            self.main_logger.error(f'Failed to stop camera. Error: {ex}')
            
    def connect_server_teaching(self, config=None):
        if self.is_connect_server:
            return True
        
        if config is None:
            config = self.get_config_from_ui()
        
        server_config = config.hardware.server
        self.server.start_server(server_config.ip, int(server_config.port))
        
        if not self.server.is_connected:
            return False
        
        self.is_connect_server = True
        return True

    def close_server_teaching(self):
        if not self.is_connect_server:
            return True
        
        self.server.stop_server()
        self.is_connect_server = False
        return True

    def open_send_data_teaching(self, config=None):
        if self.is_com_send_data:
            return True
        
        if config is None:
            config = self.get_config_from_ui()
        
        com = config.hardware.send_data.comport
        baud = int(config.hardware.send_data.baudrate)
        
        self.com_send_data.connect(com, baud)
        self.is_com_send_data = True
        return True

    def close_send_data_teaching(self):
        if not self.is_com_send_data:
            return True
        
        self.com_send_data.disconnect()
        self.is_com_send_data = False
        return True

    def open_light_teaching(self, config=None):
        if self.is_open_light:
            return True
        
        if config is None:
            config = self.get_config_from_ui()
        
        lighting_config = config.hardware.lighting
        com = lighting_config.comport
        baud = int(lighting_config.baudrate)
        controller = lighting_config.controller

        if controller == 'LCPController':
            self.light = self.light_lcp
        else:
            self.light = self.light_dcp

        try:
            self.light.light_logger.signalLog.disconnect(self.add_log_view)
        except:
            pass
            
        self.light.light_logger.signalLog.connect(self.add_log_view)
        
        if self.light.open(com, baud):
            self.is_open_light = True
            return True
        else:
            return False

    def close_light_teaching(self):
        if not self.is_open_light:
            return True
        
        if self.light:
            try:
                self.light.light_logger.signalLog.disconnect(self.add_log_view)
            except:
                pass
                      
            if self.light.close():
                self.is_open_light = False
                self.light = None
                return True
        else:
            return False
    
    def set_light_value(self, channel_0=0 , channel_1=0, channel_2=0, channel_3=0, delay=0):
        self.light.set_light_value(0, channel_0)
        self.light.set_light_value(1, channel_1)
        self.light.set_light_value(2, channel_2)
        self.light.set_light_value(3, channel_3)
        delay = delay / 1000
        time.sleep(delay)
        
    def update_light_value(self):
        if hasattr(self, 'light') and self.ui.but_open_light.property("status") == "Close":
            channel_0 = self.ui.spin_box_channel_value_0.value()
            channel_1 = self.ui.spin_box_channel_value_1.value()
            channel_2 = self.ui.spin_box_channel_value_2.value()
            channel_3 = self.ui.spin_box_channel_value_3.value()
            
            self.light.set_light_value(0, channel_0)
            self.light.set_light_value(1, channel_1)
            self.light.set_light_value(2, channel_2)
            self.light.set_light_value(3, channel_3)
        
    def read_data_tcp(self, data):
        self.ui.list_widget_message_tcp.addItem(data)

    def on_click_but_send_data_tcp(self):
        message = self.ui.plain_text_edit_send_data.toPlainText()
        self.server.send_data_to_all_clients(message)
        self.ui.plain_text_edit_send_data.clear()

    def process_center(self, x, y, swap_xy: bool, negative_x: bool, negative_y: bool):
        if swap_xy:
            x, y = y, x
        if negative_x:
            x = -x
        if negative_y:
            y = -y
        return x, y
    
    def get_config_from_ui(self):
        config = SimpleNamespace(
            name_model = self.ui.combo_box_model_name_teaching.currentText(),
            
            shapes=self.get_shapes(self.teaching_canvas),
            
            model_ai=SimpleNamespace(
                model_name=self.ui.combo_box_model_ai.currentText(),
                classify_name=self.ui.combo_box_classify_ai.currentText(),
                use_classify=self.ui.check_box_use_classify.isChecked(),
                agnostic_nms=self.ui.check_box_agnostic_nms.isChecked(),
                threshold_set=self.ui.spin_box_threshold_set.value(),
                threshold_box=self.ui.spin_box_threshold_box.value(),
                conf=self.ui.spin_box_confidence.value(),
                iou=self.ui.spin_box_iou.value(),
                max_det=self.ui.spin_box_max_det.value(),
            ),
            
            calibration=SimpleNamespace(
                use_calib=self.ui.check_box_use_calib.isChecked(),
                center_origin_x=float(self.ui.label_center_origin_x.text()),
                center_origin_y=float(self.ui.label_center_origin_y.text()),
                scale_x=self.ui.spin_box_scale_x.value(),
                scale_y=self.ui.spin_box_scale_y.value(),
                swap_xy=self.ui.check_box_swap_xy.isChecked(),
                negative_x=self.ui.check_box_negative_x.isChecked(),
                negative_y=self.ui.check_box_negative_y.isChecked(),
            ),
                        
            hardware=SimpleNamespace(
                server=SimpleNamespace(
                    ip=self.ui.line_edit_ip_server.text(),
                    port=self.ui.line_edit_port_server.text()
                ),
                camera=SimpleNamespace(
                    id=self.ui.combo_box_id_camera.currentText(),
                    feature=self.ui.combo_box_feature_camera.currentText(),
                    name=self.ui.combo_box_name_camera.currentText(),
                    delay=self.ui.spin_box_time_delay_camera.value()
                ),
                lighting=SimpleNamespace(
                    comport=self.ui.combo_box_com_controller_light.currentText(),
                    baudrate=self.ui.combo_box_baudrate_controller_light.currentText(),
                    channel_value_0=self.ui.spin_box_channel_value_0.value(),
                    channel_value_1=self.ui.spin_box_channel_value_1.value(),
                    channel_value_2=self.ui.spin_box_channel_value_2.value(),
                    channel_value_3=self.ui.spin_box_channel_value_3.value(),
                    controller=self.ui.combo_box_light_controller.currentText(),
                    delay=self.ui.spin_box_delay_controller.value()
                ),
                system=SimpleNamespace(
                    log_dir=self.ui.line_edit_log_dir.text(),
                    log_size=self.ui.spin_box_log_size.value(),
                    database_path=self.ui.line_edit_database_path.text(),
                    production_mode=self.ui.radio_production_mode.isChecked(),
                    no_ng_mode=self.ui.radio_no_ng_mode.isChecked(),
                    bypass_mode=self.ui.radio_bypass_mode.isChecked()
                ),
                io=SimpleNamespace(
                    comport=self.ui.combo_box_com_controller_io.currentText(),
                    baudrate=self.ui.combo_box_baudrate_controller_io.currentText(),
                    delay=self.ui.line_edit_time_delay_io.text()
                ),
                scanner=SimpleNamespace(
                    comport=self.ui.combo_box_com_controller_scanner.currentText(),
                    baudrate=self.ui.combo_box_baudrate_controller_scanner.currentText()
                ),
                send_data=SimpleNamespace(
                    comport=self.ui.combo_box_com_controller_send_data.currentText(),
                    baudrate=self.ui.combo_box_baudrate_controller_send_data.currentText()
                )
            )
        )

        return config

    def set_config_teaching(self, config):
        # self.ui.combo_box_model_name_teaching.setCurrentText(config.get("name_model", ""))
        
        self.set_shapes(self.teaching_canvas, config.get("shapes", {}))
        
        model_ai = config.get("model_ai", {})
        self.ui.combo_box_model_ai.setCurrentText(model_ai.get("model_name", ""))
        self.ui.combo_box_classify_ai.setCurrentText(model_ai.get("classify_name", ""))
        self.ui.check_box_use_classify.setChecked(model_ai.get("use_classify", False))
        self.ui.check_box_agnostic_nms.setChecked(model_ai.get("agnostic_nms", False))
        self.ui.spin_box_threshold_set.setValue(model_ai.get("threshold_set", 50))
        self.ui.spin_box_threshold_box.setValue(model_ai.get("threshold_box", 50))
        self.ui.spin_box_confidence.setValue(model_ai.get("conf", 0.1))
        self.ui.spin_box_iou.setValue(model_ai.get("iou", 1))
        self.ui.spin_box_max_det.setValue(model_ai.get("max_det", 100))
        
        calibration = config.get("calibration", {})
        self.ui.check_box_use_calib.setChecked(calibration.get("use_calib", False))
        self.ui.label_center_origin_x.setText(str(calibration.get("center_origin_x", 0.0)))
        self.ui.label_center_origin_y.setText(str(calibration.get("center_origin_y", 0.0)))
        self.ui.spin_box_scale_x.setValue(calibration.get("scale_x", 1.0))
        self.ui.spin_box_scale_y.setValue(calibration.get("scale_y", 1.0))
        self.ui.check_box_swap_xy.setChecked(calibration.get("swap_xy", False))
        self.ui.check_box_negative_x.setChecked(calibration.get("negative_x", False))
        self.ui.check_box_negative_y.setChecked(calibration.get("negative_y", False))

        # Cập nhật phần cứng
        hardware = config.get("hardware", {})

        # Server
        server_config = hardware.get("server", {})
        self.ui.line_edit_ip_server.setText(server_config.get("ip", ""))
        self.ui.line_edit_port_server.setText(server_config.get("port", ""))

        # Camera
        camera_config = hardware.get("camera", {})
        self.ui.combo_box_id_camera.setCurrentText(camera_config.get("id", ""))
        self.ui.combo_box_feature_camera.setCurrentText(camera_config.get("feature", ""))
        self.ui.combo_box_name_camera.setCurrentText(camera_config.get("name", ""))
        self.ui.spin_box_time_delay_camera.setValue(camera_config.get("delay", 0))

        # Lighting
        lighting_config = hardware.get("lighting", {})
        self.ui.combo_box_com_controller_light.setCurrentText(lighting_config.get("comport", ""))
        self.ui.combo_box_baudrate_controller_light.setCurrentText(lighting_config.get("baudrate", ""))
        self.ui.spin_box_channel_value_0.setValue(lighting_config.get("channel_value_0", 0))
        self.ui.spin_box_channel_value_1.setValue(lighting_config.get("channel_value_1", 0))
        self.ui.spin_box_channel_value_2.setValue(lighting_config.get("channel_value_2", 0))
        self.ui.spin_box_channel_value_3.setValue(lighting_config.get("channel_value_3", 0))
        self.ui.combo_box_light_controller.setCurrentText(lighting_config.get("controller", ""))
        self.ui.spin_box_delay_controller.setValue(lighting_config.get("delay", 0))

        # System
        system_config = hardware.get("system", {})
        self.ui.line_edit_log_dir.setText(system_config.get("log_dir", ""))
        self.ui.spin_box_log_size.setValue(system_config.get("log_size", 0))
        self.ui.line_edit_database_path.setText(system_config.get("database_path", ""))
        
        # Set mode radio buttons
        self.ui.radio_production_mode.setChecked(system_config.get("production_mode", True))  # Default to production mode
        self.ui.radio_no_ng_mode.setChecked(system_config.get("no_ng_mode", False))
        self.ui.radio_bypass_mode.setChecked(system_config.get("bypass_mode", False))

        # IO
        io_config = hardware.get("io", {})
        self.ui.combo_box_com_controller_io.setCurrentText(io_config.get("comport", ""))
        self.ui.combo_box_baudrate_controller_io.setCurrentText(io_config.get("baudrate", ""))
        self.ui.line_edit_time_delay_io.setText(io_config.get("delay", ""))

        # Scanner
        scanner_config = hardware.get("scanner", {})
        self.ui.combo_box_com_controller_scanner.setCurrentText(scanner_config.get("comport", ""))
        self.ui.combo_box_baudrate_controller_scanner.setCurrentText(scanner_config.get("baudrate", ""))

        # Send Data
        send_data_config = hardware.get("send_data", {})
        self.ui.combo_box_com_controller_send_data.setCurrentText(send_data_config.get("comport", ""))
        self.ui.combo_box_baudrate_controller_send_data.setCurrentText(send_data_config.get("baudrate", ""))

    def get_config_auto(self, model_name):
        config_data = self.handle_file_json.load(model_name)

        model_ai_config = config_data.get('model_ai', {})
        calib_config = config_data.get("calibration", {})
        hardware = config_data.get("hardware", {})

        # Các phần cứng
        server_config = hardware.get("server", {})
        camera_config = hardware.get("camera", {})
        lighting_config = hardware.get("lighting", {})
        system_config = hardware.get("system", {})
        io_config = hardware.get("io", {})
        scanner_config = hardware.get("scanner", {})
        send_data_config = hardware.get("send_data", {})

        config = SimpleNamespace(
            name_model=config_data.get("name_model", ""),
            shapes=config_data.get("shapes", {}),

            model_ai=SimpleNamespace(
                model_name=model_ai_config.get("model_name", ""),
                classify_name=model_ai_config.get("classify_name", ""),
                use_classify=model_ai_config.get("use_classify", False),
                agnostic_nms=model_ai_config.get("agnostic_nms", False),
                threshold_set=model_ai_config.get("threshold_set", ""),
                threshold_box=model_ai_config.get("threshold_box", ""),
                conf=model_ai_config.get("conf", 0.1),
                iou=model_ai_config.get("iou", 1),
                max_det=model_ai_config.get("max_det", 100),
            ),
            
            calibration=SimpleNamespace(
                use_calib=calib_config.get("use_calib", False),
                center_origin_x=calib_config.get("center_origin_x", 0.0),
                center_origin_y=calib_config.get("center_origin_y", 0.0),
                scale_x=calib_config.get("scale_x", 1.0),
                scale_y=calib_config.get("scale_y", 1.0),
                swap_xy=calib_config.get("swap_xy", False),
                negative_x=calib_config.get("negative_x", False),
                negative_y=calib_config.get("negative_y", False),
            ),

            hardware=SimpleNamespace(                
                server=SimpleNamespace(
                    ip=server_config.get("ip", ""),
                    port=server_config.get("port", "")
                ),
                camera=SimpleNamespace(
                    id=camera_config.get("id", ""),
                    feature=camera_config.get("feature", ""),
                    name=camera_config.get("name", ""),
                    delay=camera_config.get("delay", 0)
                ),
                lighting=SimpleNamespace(
                    comport=lighting_config.get("comport", ""),
                    baudrate=lighting_config.get("baudrate", ""),
                    channel_value_0=lighting_config.get("channel_value_0", 0),
                    channel_value_1=lighting_config.get("channel_value_1", 0),
                    channel_value_2=lighting_config.get("channel_value_2", 0),
                    channel_value_3=lighting_config.get("channel_value_3", 0),
                    controller=lighting_config.get("controller", ""),
                    delay=lighting_config.get("delay", 0)
                ),
                system=SimpleNamespace(
                    log_dir=system_config.get("log_dir", ""),
                    log_size=system_config.get("log_size", 0),
                    database_path=system_config.get("database_path", ""),
                    production_mode=system_config.get("production_mode", True),  # Default to production mode
                    no_ng_mode=system_config.get("no_ng_mode", False),
                    bypass_mode=system_config.get("bypass_mode", False)
                ),
                io=SimpleNamespace(
                    comport=io_config.get("comport", ""),
                    baudrate=io_config.get("baudrate", ""),
                    delay=io_config.get("delay", "")
                ),
                scanner=SimpleNamespace(
                    comport=scanner_config.get("comport", ""),
                    baudrate=scanner_config.get("baudrate", "")
                ),
                send_data=SimpleNamespace(
                    comport=send_data_config.get("comport", ""),
                    baudrate=send_data_config.get("baudrate", "")
                )
            )
        )

        return config

    def get_detection_mode(self, config):
        system_config = config.hardware.system
        if system_config.production_mode:
            return "production"
        elif system_config.no_ng_mode:
            return "no_ng"
        elif system_config.bypass_mode:
            return "bypass"
        else:
            return "production"

    def load_model_change_teaching(self):
        name_model = self.ui.combo_box_model_name_teaching.currentText()
        if name_model and self.ui.combo_box_model_name_teaching.findText(name_model) == -1:
            QMessageBox.warning(self, 'Warning', 'The model name is incorrect', QMessageBox.StandardButton.Close)
            return
        config = self.handle_file_json.load(file_path=f'Settings/ModelSettings/{name_model}')
        self.set_config_teaching(config)
    
    def get_shapes(self, canvas: Canvas):
        shapes: list[MyShape] = canvas.shapes
        shape_config = {}
        
        for s in shapes:
            shape_config[s.label] = s.cvBox
        # print(shape_config)
        return shape_config
    
    def set_shapes(self, canvas: Canvas, shape_config):
        canvas.clear()
        for label in shape_config:
            s = self.new_shape(label, shape_config[label])
            canvas.append_shape(s)  # Thay đổi dòng này
            
    def new_shape(self, label, box):
        s = MyShape(label)
        x, y, w, h = box
        s.points = [
            QPointF(x, y),
            QPointF(x + w, y),
            QPointF(x + w, y + h),
            QPointF(x, y + h)            
        ]
        return s
    
    def update_label_status(self, status):
        self.ui.label_result.setText(status)
        self.ui.label_result.setProperty("status", status)
        self.ui.label_result.style().polish(self.ui.label_result)
    
    def write_log(self, result: RESULT):
        try:
            if result is not None:
                today_folder = datetime.now().strftime('%Y_%m_%d')
                
                path_input = self.main_logger.log_image(
                    'Source', 
                    result.src,
                    f'Log_Vision/Log_Image/{today_folder}/{result.model_name}/src',
                    result.ret
                )
                
                path_output = self.main_logger.log_image(
                    result.model_name, 
                    result.dst,
                    f'Log_Vision/Log_Image/{today_folder}/{result.model_name}/dst',
                    result.ret
                )
        except Exception as e:
            print(e)
    
    def add_log_view(self, message):
        item = QStandardItem(message)
        
        if "ERROR" in message:
            item.setBackground(QColor("#FF6347"))  # Nền đỏ cam
            item.setForeground(QColor("#FFFFFF"))  # Chữ trắng

        elif "CRITICAL" in message:
            item.setBackground(QColor("#8B0000"))  # Nền đỏ đậm
            item.setForeground(QColor("#FFFFFF"))  # Chữ trắng

        self.log_model.appendRow(item)
        if self.log_model.rowCount() > 50:
            self.log_model.removeRows(0, 42)
        
        self.ui.list_log_view.scrollToBottom()
    
    def upload_model_ai(self):
        model_dir = "res/ModelAI"
        model_names = [name for name in os.listdir(model_dir) if name.endswith((".pt", ".pth"))]
        classify_names = [name for name in os.listdir(model_dir) if name.endswith((".pt", ".pth"))]
        
        self.ui.combo_box_model_ai.clear()
        self.ui.combo_box_model_ai.addItems(model_names)
        self.ui.combo_box_classify_ai.clear()
        self.ui.combo_box_classify_ai.addItems(classify_names)
        
        
    def upload_model_names(self):
        model_dir = "Settings/ModelSettings"
        model_names = [name for name in os.listdir(model_dir)]
        self.ui.combo_box_model_name_teaching.clear()
        self.ui.combo_box_model_name_teaching.addItems(model_names)
        self.ui.combo_box_model_name_auto.clear()
        self.ui.combo_box_model_name_auto.addItems(model_names)
        self.ui.combo_box_model_name_data.addItems(model_names)
        
    def upload_feature_camera(self):
        feature_dir = "res/CameraFeature"
        feature_names = [name for name in os.listdir(feature_dir) if name.endswith((".ini"))]
        self.ui.combo_box_feature_camera.clear()
        self.ui.combo_box_feature_camera.addItems(feature_names)
    
    def refresh_ports(self):
        ports = serial.tools.list_ports.comports()
        
        self.ui.combo_box_com_controller_light.clear()
        self.ui.combo_box_com_controller_io.clear()
        self.ui.combo_box_com_controller_scanner.clear()
        self.ui.combo_box_com_controller_send_data.clear()
        self.ui.combo_box_com_controller_light.addItems([port.device for port in ports])
        self.ui.combo_box_com_controller_io.addItems([port.device for port in ports])
        self.ui.combo_box_com_controller_scanner.addItems([port.device for port in ports])
        self.ui.combo_box_com_controller_send_data.addItems([port.device for port in ports])
        
        self.ui.combo_box_baudrate_controller_light.clear()
        self.ui.combo_box_baudrate_controller_io.clear()
        self.ui.combo_box_baudrate_controller_scanner.clear()
        self.ui.combo_box_baudrate_controller_send_data.clear()
        self.ui.combo_box_baudrate_controller_light.addItems(list(map(str, serial.Serial.BAUDRATES)))
        self.ui.combo_box_baudrate_controller_io.addItems(list(map(str, serial.Serial.BAUDRATES)))
        self.ui.combo_box_baudrate_controller_scanner.addItems(list(map(str, serial.Serial.BAUDRATES)))
        self.ui.combo_box_baudrate_controller_send_data.addItems(list(map(str, serial.Serial.BAUDRATES)))
        
        devices = get_camera_devices()
        self.ui.combo_box_id_camera.clear()
        self.ui.combo_box_id_camera.addItems(list(devices.keys()))
    
    def set_disable_auto(self, ret):
        self.ui.but_open_camera_teaching.setDisabled(ret)
        self.ui.but_connect_server.setDisabled(ret)
        self.ui.but_open_io_controller.setDisabled(ret)
        self.ui.but_connect_scanner.setDisabled(ret)
        self.ui.but_open_com_send_data.setDisabled(ret)
        
        self.ui.combo_box_feature_camera.setDisabled(ret)
        self.ui.combo_box_id_camera.setDisabled(ret)
        self.ui.combo_box_name_camera.setDisabled(ret)
        self.ui.combo_box_com_controller_io.setDisabled(ret)
        self.ui.combo_box_com_controller_scanner.setDisabled(ret)
        self.ui.combo_box_com_controller_light.setDisabled(ret)
        self.ui.combo_box_com_controller_send_data.setDisabled(ret)
        self.ui.combo_box_baudrate_controller_light.setDisabled(ret)
        self.ui.combo_box_baudrate_controller_io.setDisabled(ret)
        self.ui.combo_box_baudrate_controller_scanner.setDisabled(ret)
        self.ui.combo_box_baudrate_controller_send_data.setDisabled(ret)
        self.ui.combo_box_model_ai.setDisabled(ret)
        self.ui.combo_box_classify_ai.setDisabled(ret)
        self.ui.check_box_use_classify.setDisabled(ret)
        self.ui.check_box_agnostic_nms.setDisabled(ret)
        
        self.ui.line_edit_ip_server.setDisabled(ret)
        self.ui.line_edit_port_server.setDisabled(ret)
        self.ui.spin_box_threshold_set.setDisabled(ret)
        self.ui.spin_box_threshold_box.setDisabled(ret)
        self.ui.spin_box_confidence.setDisabled(ret)
        self.ui.spin_box_iou.setDisabled(ret)
        self.ui.spin_box_max_det.setDisabled(ret)
        self.ui.line_edit_time_delay_io.setDisabled(ret)

    def set_arUco_dictionary(self):
        for dict_name in aruco_dict_mapping.keys():
            self.ui.combo_box_aruco_dict.addItem(dict_name)
    
    def set_trigger_on(self, msg):
        self.msg_model = msg
        self.trigger_on = True
    
    def set_trigger_off(self):
        self.msg_model = None
        self.trigger_on = False

    def show_effect(self, dt=3):
        self.ui.progressBar.setValue(0)
        self.ui.progressBar.setVisible(True)
        for i in range(1, 100):
            QTimer.singleShot(dt*i, lambda value=i: self.ui.progressBar.setValue(value))

    def hide_effect(self, timeout=500):
        QTimer.singleShot(10, partial(self.ui.progressBar.setValue, 100))
        QTimer.singleShot(timeout, partial(self.ui.progressBar.setVisible, False))
    
    def saveSettings(self):
        """Lưu trạng thái layout vào file settings.ini"""
        settings = QSettings("Settings/layout_settings.ini", QSettings.IniFormat)
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.setValue("combo_box_index", self.ui.combo_box_model_name_teaching.currentIndex())
        settings.setValue("combo_box_index_auto", self.ui.combo_box_model_name_auto.currentIndex())

    def loadSettings(self):
        settings = QSettings("Settings/layout_settings.ini", QSettings.IniFormat)

        geometry = settings.value("geometry", QByteArray())
        windowState = settings.value("windowState", QByteArray())

        if isinstance(geometry, QByteArray) and not geometry.isEmpty():
            self.restoreGeometry(geometry)

        if isinstance(windowState, QByteArray) and not windowState.isEmpty():
            self.restoreState(windowState)
            
        self.ui.combo_box_model_name_teaching.setCurrentIndex(int(settings.value("combo_box_index", 0)))
        self.ui.combo_box_model_name_auto.setCurrentIndex(int(settings.value("combo_box_index_auto", 0)))
        
        config = self.handle_file_json.load('Settings/CalibrationSettings/')
        
        if config is not None:
            self.ui.spin_box_square_x.setValue(config['squares_x'])
            self.ui.spin_box_square_y.setValue(config['squares_y'])
            self.ui.spin_box_square_length.setValue(config['square_length'])
            self.ui.spin_box_marker_length.setValue(config['marker_length'])
            self.ui.combo_box_aruco_dict.setCurrentIndex(config['aruco_dict'])

    def resetLayout(self):
        confirm = QMessageBox.question(self, "Reset Layout", 
                                       "Are you sure you want to reset the layout to default?",
                                       QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.is_resetting = True
            settings = QSettings("Settings/layout_settings.ini", QSettings.IniFormat)
            settings.clear()  # Xóa tất cả dữ liệu đã lưu
            QMessageBox.information(self, "Notification", "Layout has been reset. The application will restart")
            self.restartApp()
            
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Question', 'Are You want to quit?', 
                                    QMessageBox.Yes|QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Cleanup tất cả resources
            self.cleanup_resources()
            
            if not self.is_resetting:
                self.saveSettings()
            else:
                self.restartApp()
            event.accept()
        else:
            event.ignore()

    def cleanup_resources(self):
        """Giải phóng tất cả resources"""
        # Dừng threads
        if self.is_showing_camera:
            self.stop_camera_teaching()
        
        if self.is_open_camera:
            self.close_camera_teaching()
        
        if self.is_open_light:
            self.close_light_teaching()
        
        # Giải phóng images
        if hasattr(self, 'mat'):
            del self.mat
        if hasattr(self, 'image_capture_test'):
            del self.image_capture_test
        if hasattr(self, 'image_capture_aruco'):
            del self.image_capture_aruco
        
        # Disconnect tất cả signals
        try:
            self.server.triggerOn.disconnect()
            self.server.sendData.disconnect()
            # ... disconnect other signals ...
        except:
            pass
        
        gc.collect()
            
    def restartApp(self):
        python = sys.executable
        subprocess.Popen([python] + sys.argv)
        sys.exit(0)
        
        
def show():
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Set high DPI scaling policy
    app.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    try:
        with open('res/Style/style.qss', 'r', encoding='utf-8') as file:
            style_sheet = file.read()
            app.setStyleSheet(style_sheet)
    except FileNotFoundError:
        print("Style file not found, using default styling")
    except Exception as e:
        print(f"Error loading QSS: {e}")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    show()
