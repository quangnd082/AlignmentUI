import os
import sys

# Handle both direct execution and import scenarios
if __name__ == "__main__":
    # When running directly, use absolute imports with path modification
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from Libs.shape import *
    from Libs import resources
    from Libs.ui_utils import *
    from Libs.utils import load_label
else:
    # When imported as module, use relative imports
    from .shape import *
    from . import resources
    from .ui_utils import *
    from .utils import load_label

from functools import partial
from datetime import datetime  # Thêm import cho datetime

# cursor
CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_POINT = Qt.PointingHandCursor
CURSOR_DRAW = Qt.CrossCursor
CURSOR_DRAW_POLYGON = Qt.SizeAllCursor
CURSOR_MOVE = Qt.ClosedHandCursor
CURSOR_GRAB = Qt.OpenHandCursor

class Canvas(QLabel):
    mouseMoveSignal = pyqtSignal(QPointF)
    newShapeSignal = pyqtSignal(int)
    editShapeSignal = pyqtSignal(str)
    deleteShapeSignal = pyqtSignal(int)
    moveShapeSignal = pyqtSignal(int)
    drawShapeSignal = pyqtSignal(QRectF)
    changeShapeSignal = pyqtSignal(int)
    selectedShapeSignal = pyqtSignal(int)
    zoomSignal = pyqtSignal(float)
    actionSignal = pyqtSignal(str)
    applyConfigSignal = pyqtSignal()
    
    def __init__(self, parent=None, bcontext_menu=True, benable_drawing=True):
        super().__init__(parent)
        self.setObjectName("Canvas")
        self.bcontext_menu = bcontext_menu
        self.picture = QPixmap(640, 480)
        # self.picture = QPixmap(1280,1020)
        self.painter = QPainter()
        self.scale = 1
        self.org = QPointF()
        self.moving = False
        self.edit = False
        self.drawing = False
        self.highlight = False
        self.wheel = False
        self.current_pos = QPointF()
        self.current = None
        self.win_start_pos = QPointF()
        self.start_pos = QPointF()
        self.start_pos_moving = QPointF()
        
        self.line1 = [QPointF(),QPointF()]
        self.line2 = [QPointF(),QPointF()]

        self.text_pixel_color = "BGR:"
        self.shapes = []
        # self.dict_shapes = {}
        self.idVisible = None
        self.idSelected = None
        self.idCorner = None
        self.benable_drawing = benable_drawing
        self.label_path="res/canvas/classes.txt"
        self.labels = load_label(self.label_path)
        self.last_label = ""
        self.boxEditLabel = BoxEditLabel("Enter shape name",self)
        
        # Thêm biến để track grid vừa được tạo
        self.last_created_grid = []  # Lưu danh sách indices của grid vừa tạo
        self.all_grids = []  # Lưu tất cả các grids đã tạo để tracking
        
        # Initialize shapes list
        self.shapes = []
        
        #========
        self.contextMenu = QMenu()
        action = partial(newAction,self)

        copy = action("copy",self.copyShape,"","copy","copy shape")

        lock = action("Lock", self.change_lock,"","lock","Lock/Unlock shape")
        lock_all = action("Lock All", self.change_lock_all,"","lock","Lock/Unlock all shapes")

        # hide = action("Hide", self.change_hide,"","hide","Hide/Show shape")
        hide_all = action("Hide All", self.change_hide_all,"","lock","Hide/Show all shapes")

        edit = action("edit",self.editShape,"","edit","edit shape")
        delete = action("delete",self.deleteShape,"","delete","delete shape")
        delete_all = action("delete all",self.delete_all,"","","delete shape")

        # Add Create Grid action
        create_grid = action("Create Grid", self.createGrid, "", "grid", "Create grid of boxes")
        
        # Add Select All action
        select_all = action("Select All", self.selectAll, "", "select_all", "Select all shapes")

        # Add Crop Image action
        crop_image = action("Crop Image", self.cropImage, "", "crop", "Crop image by bounding boxes")

        self.actions = struct(
            copy    = copy,
            edit    = edit,
            delete  = delete,
            delete_all = delete_all,
            lock=lock, lock_all=lock_all,
            hide_all=hide_all,
            create_grid=create_grid,
            select_all=select_all,
            crop_image=crop_image
        )

        addActions(self.contextMenu,[lock, lock_all])
        addActions(self.contextMenu,[hide_all])
        addActions(self.contextMenu,[create_grid, select_all])
        addActions(self.contextMenu,[crop_image])  # Thêm crop action vào menu
        self.contextMenu.addSeparator()
        addActions(self.contextMenu,[edit,copy,delete,delete_all])

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.popUpMenu)

        # Kích hoạt Lock All từ đầu để tất cả shapes mới đều bị lock
        self.change_lock_all()

        # 
        style = "QLabel{background-color:rgba(128, 128, 128, 150); color:white; font:bold 12px}"

        self.tool_zoom = QWidget(self)
        self.label_pos = newLabel("", style=style)
        self.label_rect = newLabel("", style=style)
        self.label_color = newLabel("", style=style)

        self.label_pos.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.zoom_buttons = newDialogButton(self.tool_zoom, ["Draw rectangle", "Zoom in", "Zoom out", "Fit window", "Full screen"], 
                        [self.active_edit, 
                         lambda: self.zoom_manual(1.2), 
                         lambda: self.zoom_manual(0.8), 
                         lambda: self.fit_window(), 
                         self.on_show_full_screen],
                        icons=["draw", "zoom_in", "zoom_out", "fit_window", "full_screen"],
                        orient=Qt.Horizontal).buttons()
        #=======
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)
        # ==============

    def cropImage(self):
        """Crop tất cả vùng có bounding box và lưu vào folder crop"""
        if self.picture is None:
            QMessageBox.warning(self, "Warning", "No image loaded!")
            return
        
        # Kiểm tra xem có bounding boxes không
        visible_shapes = [shape for shape in self.shapes if not shape.hide]
        if len(visible_shapes) == 0:
            QMessageBox.warning(self, "Warning", "No bounding boxes found!")
            return

        # Tạo folder crop nếu chưa có
        crop_folder = "Images/ImageCrop"
        if not os.path.exists(crop_folder):
            try:
                os.makedirs(crop_folder)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot create crop folder: {str(e)}")
                return

        # Convert QPixmap to QImage để crop
        image = self.picture.toImage()
        
        success_count = 0
        error_count = 0
        
        # Tạo timestamp để tránh duplicate tên file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, shape in enumerate(visible_shapes):
            try:
                # Lấy bounding box coordinates
                points = shape.points
                if len(points) < 4:
                    continue
                    
                # Tính toán vùng crop
                min_x = max(0, int(min(p.x() for p in points)))
                max_x = min(image.width(), int(max(p.x() for p in points)))
                min_y = max(0, int(min(p.y() for p in points)))
                max_y = min(image.height(), int(max(p.y() for p in points)))
                
                # Kiểm tra kích thước hợp lệ
                width = max_x - min_x
                height = max_y - min_y
                
                if width <= 0 or height <= 0:
                    print(f"Invalid dimensions for shape {i}: {width}x{height}")
                    error_count += 1
                    continue
                
                # Crop image
                cropped = image.copy(min_x, min_y, width, height)
                
                if cropped.isNull():
                    print(f"Failed to crop shape {i}")
                    error_count += 1
                    continue
                
                # Tạo tên file dựa trên label của shape
                label = shape.label if hasattr(shape, 'label') and shape.label else f"shape_{i}"
                # Làm sạch label để tránh ký tự không hợp lệ trong tên file
                safe_label = "".join(c for c in label if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_label = safe_label.replace(' ', '_')
                
                filename = f"{safe_label}_{timestamp}_{i:03d}.png"
                filepath = os.path.join(crop_folder, filename)
                
                # Lưu file
                if cropped.save(filepath, "PNG"):
                    success_count += 1
                    print(f"Saved: {filepath} ({width}x{height})")
                else:
                    print(f"Failed to save: {filepath}")
                    error_count += 1
                    
            except Exception as e:
                print(f"Error processing shape {i}: {str(e)}")
                error_count += 1
                continue

        # Hiển thị kết quả
        if success_count > 0:
            message = f"Successfully cropped {success_count} images to '{crop_folder}' folder"
            if error_count > 0:
                message += f"\n{error_count} images failed to crop"
            QMessageBox.information(self, "Crop Complete", message)
        else:
            QMessageBox.warning(self, "Crop Failed", f"Failed to crop any images. {error_count} errors occurred.")

    def createGrid(self):
        """Tạo lưới các box chia đều trong vùng box được chọn với preview overlay"""
        if self.picture is None:
            QMessageBox.warning(self, "Warning", "No image loaded!")
            return
        
        if self.idSelected is None:
            QMessageBox.warning(self, "Warning", "Please select a box first!")
            return
        
        # Lấy shape được chọn
        selected_shape = self[self.idSelected]
        selected_index = self.idSelected
        
        # Dialog để nhập số hàng, cột, overlap và spacing
        dialog = QDialog(self)
        dialog.setWindowTitle("Create Grid in Selected Box - Live Preview")
        dialog.setModal(False)  # Cho phép tương tác với canvas
        dialog.resize(400, 320)
        
        layout = QVBoxLayout(dialog)
        
        # Info label
        info_label = QLabel(f"Creating grid inside: {selected_shape.label}")
        info_label.setStyleSheet("font-weight: bold; color: blue;")
        layout.addWidget(info_label)
        
        # Input cho số hàng
        row_layout = QHBoxLayout()
        row_layout.addWidget(QLabel("Rows:"))
        row_input = QSpinBox()
        row_input.setMinimum(1)
        row_input.setMaximum(20)
        row_input.setValue(2)  # Default 2 hàng
        row_layout.addWidget(row_input)
        layout.addLayout(row_layout)
        
        # Input cho số cột
        col_layout = QHBoxLayout()
        col_layout.addWidget(QLabel("Columns:"))
        col_input = QSpinBox()
        col_input.setMinimum(1)
        col_input.setMaximum(20)
        col_input.setValue(2)  # Default 2 cột
        col_layout.addWidget(col_input)
        layout.addLayout(col_layout)
        
        # Tab widget để chọn giữa Overlap và Spacing
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Tab Overlap
        overlap_tab = QWidget()
        overlap_layout = QVBoxLayout(overlap_tab)
        
        overlap_input_layout = QHBoxLayout()
        overlap_input_layout.addWidget(QLabel("Overlap (%):"))
        overlap_input = QSpinBox()
        overlap_input.setMinimum(0)
        overlap_input.setMaximum(90)  # Tối đa 90% overlap
        overlap_input.setValue(0)  # Default không overlap
        overlap_input.setSuffix("%")
        overlap_input_layout.addWidget(overlap_input)
        overlap_layout.addLayout(overlap_input_layout)
        
        overlap_help = QLabel("0% = no overlap, 50% = boxes overlap by half")
        overlap_help.setStyleSheet("color: gray; font-size: 10px;")
        overlap_layout.addWidget(overlap_help)
        
        tab_widget.addTab(overlap_tab, "Overlap")
        
        # Tab Spacing
        spacing_tab = QWidget()
        spacing_layout = QVBoxLayout(spacing_tab)
        
        spacing_input_layout = QHBoxLayout()
        spacing_input_layout.addWidget(QLabel("Spacing (%):"))
        spacing_input = QSpinBox()
        spacing_input.setMinimum(0)
        spacing_input.setMaximum(999)  # Tối đa 999% spacing - rất lớn cho mọi use case
        spacing_input.setValue(0)  # Default không spacing
        spacing_input.setSuffix("%")
        spacing_input_layout.addWidget(spacing_input)
        spacing_layout.addLayout(spacing_input_layout)
        
        spacing_help = QLabel("0% = no spacing, 100% = space equals box size, 500% = very large spacing")
        spacing_help.setStyleSheet("color: gray; font-size: 10px;")
        spacing_layout.addWidget(spacing_help)
        
        tab_widget.addTab(spacing_tab, "Spacing")
        
        # Preview info
        preview_label = QLabel("Preview: Change values to see preview on canvas")
        preview_label.setStyleSheet("color: green; font-weight: bold;")
        layout.addWidget(preview_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("Apply Grid")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        # Biến để track preview shapes - chỉ overlay trên shape gốc
        preview_shapes = []
        
        # Ẩn shape gốc khi preview (nhưng không xóa)
        original_hide_state = selected_shape.hide
        
        def update_preview():
            """Cập nhật preview overlay"""
            nonlocal preview_shapes
            
            # Xóa preview shapes cũ
            for shape in preview_shapes:
                if shape in self.shapes:
                    self.shapes.remove(shape)
            preview_shapes.clear()
            
            # Ẩn shape gốc để hiển thị preview overlay
            selected_shape.hide = True
            
            # Tạo preview shapes mới
            rows = row_input.value()
            cols = col_input.value()
            
            # Kiểm tra tab nào đang được chọn
            if tab_widget.currentIndex() == 0:  # Overlap tab
                overlap_percent = overlap_input.value()
                spacing_percent = 0
            else:  # Spacing tab
                overlap_percent = 0
                spacing_percent = spacing_input.value()
            
            preview_shapes = self._createPreviewGridBoxes(selected_shape, rows, cols, overlap_percent, spacing_percent)
            
            # Cập nhật preview label
            mode = "overlap" if tab_widget.currentIndex() == 0 else "spacing"
            value = overlap_percent if mode == "overlap" else spacing_percent
            preview_label.setText(f"Preview: {rows}x{cols} grid with {value}% {mode} ({len(preview_shapes)} boxes)")
            
            # Force repaint canvas
            self.update()
        
        def restore_original():
            """Khôi phục trạng thái gốc - xóa preview và hiện lại shape gốc"""
            nonlocal preview_shapes
            
            # Xóa tất cả preview shapes
            for shape in preview_shapes:
                if shape in self.shapes:
                    self.shapes.remove(shape)
            preview_shapes.clear()
            
            # Hiện lại shape gốc
            selected_shape.hide = original_hide_state
            
            # Reset selection
            self.idSelected = selected_index
            self.idVisible = None
            self.idCorner = None
            
            self.update()
        
        def apply_grid():
            """Áp dụng grid - thay thế shape gốc bằng grid"""
            nonlocal preview_shapes
            
            if len(preview_shapes) == 0:
                restore_original()
                dialog.reject()
                return
            
            # Convert preview shapes thành real shapes với tên đúng
            rows = row_input.value()
            cols = col_input.value()
            base_label = selected_shape.label
            
            created_indices = []
            index = 0
            
            for shape in preview_shapes:
                # Tạo tên đúng cho grid (không có "_copy")
                shape.label = f"{base_label}{index}"
                shape.preview = False  # Remove preview flag
                
                # Áp dụng trạng thái lock hiện tại
                if hasattr(self.actions, 'lock_all'):
                    shape.lock = (self.actions.lock_all.text() == "UnLock All")
                else:
                    shape.lock = True
                
                # Emit signals cho real shapes
                if shape in self.shapes:
                    shape_index = self.shapes.index(shape)
                    created_indices.append(shape_index)
                    self.newShapeSignal.emit(shape_index)
                    self.append_new_label(shape.label)
                
                index += 1
            
            # Lưu grid info với thông tin spacing/overlap
            self.last_created_grid = created_indices
            mode = "overlap" if tab_widget.currentIndex() == 0 else "spacing"
            value = overlap_input.value() if mode == "overlap" else spacing_input.value()
            
            grid_info = {
                'base_label': base_label,
                'indices': created_indices.copy(),
                'mode': mode,
                'value': value
            }
            self.all_grids.append(grid_info)
            
            # Xóa shape gốc khỏi danh sách
            if selected_shape in self.shapes:
                original_index = self.shapes.index(selected_shape)
                self.deleteShapeSignal.emit(original_index)
                self.shapes.remove(selected_shape)
                
                # Cập nhật indices sau khi xóa
                self.last_created_grid = [idx - 1 if idx > original_index else idx for idx in self.last_created_grid]
                for grid in self.all_grids:
                    grid['indices'] = [idx - 1 if idx > original_index else idx for idx in grid['indices']]
            
            self.idVisible = self.idSelected = self.idCorner = None
            dialog.accept()
        
        # Connect signals để update preview khi thay đổi giá trị
        row_input.valueChanged.connect(update_preview)
        col_input.valueChanged.connect(update_preview)
        overlap_input.valueChanged.connect(update_preview)
        spacing_input.valueChanged.connect(update_preview)
        tab_widget.currentChanged.connect(update_preview)
        
        # Connect buttons
        ok_button.clicked.connect(apply_grid)
        cancel_button.clicked.connect(lambda: (restore_original(), dialog.reject()))
        
        # Connect dialog close event
        def closeEvent(event):
            restore_original()
            event.accept()
        dialog.closeEvent = closeEvent
        
        # Hiển thị preview ban đầu
        update_preview()
        
        # Hiển thị dialog
        dialog.show()

    def _createPreviewGridBoxes(self, base_shape, rows, cols, overlap_percent=0, spacing_percent=0):
        """Tạo preview boxes overlay trên shape gốc với hỗ trợ cả overlap và spacing"""
        if self.picture is None:
            return []
        
        # Lấy bounds của shape gốc
        points = base_shape.points
        min_x = min(p.x() for p in points)
        max_x = max(p.x() for p in points)
        min_y = min(p.y() for p in points)
        max_y = max(p.y() for p in points)
        
        # Kích thước vùng được chọn
        box_width = max_x - min_x
        box_height = max_y - min_y
        
        # Tính toán kích thước và bước nhảy
        if spacing_percent > 0:
            # SPACING MODE: Boxes cách nhau một khoảng spacing_percent
            spacing_ratio = spacing_percent / 100.0
            
            if rows == 1:
                cell_height = box_height
                step_y = 0
            else:
                # Với spacing: total_height = rows * cell_height + (rows-1) * spacing
                # spacing = cell_height * spacing_ratio
                # box_height = rows * cell_height + (rows-1) * cell_height * spacing_ratio
                # box_height = cell_height * (rows + (rows-1) * spacing_ratio)
                cell_height = box_height / (rows + (rows - 1) * spacing_ratio)
                step_y = cell_height * (1 + spacing_ratio)
            
            if cols == 1:
                cell_width = box_width
                step_x = 0
            else:
                cell_width = box_width / (cols + (cols - 1) * spacing_ratio)
                step_x = cell_width * (1 + spacing_ratio)
        else:
            # OVERLAP MODE: Boxes chồng lấp nhau overlap_percent
            overlap_ratio = overlap_percent / 100.0
            
            if rows == 1:
                cell_height = box_height
                step_y = 0
            else:
                cell_height = box_height / (rows - overlap_ratio * (rows - 1))
                step_y = cell_height * (1 - overlap_ratio)
            
            if cols == 1:
                cell_width = box_width
                step_x = 0
            else:
                cell_width = box_width / (cols - overlap_ratio * (cols - 1))
                step_x = cell_width * (1 - overlap_ratio)
        
        # Tên gốc của box được chọn
        base_label = base_shape.label
        
        # Tạo các preview box với tên đúng (không có "_copy")
        preview_shapes = []
        index = 0
        
        for row in range(rows):
            for col in range(cols):
                # Tính toán vị trí
                x1 = min_x + col * step_x
                y1 = min_y + row * step_y
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                # Đảm bảo không vượt quá biên của box gốc (chỉ áp dụng cho overlap mode)
                if spacing_percent == 0:  # Overlap mode
                    x2 = min(x2, max_x)
                    y2 = min(y2, max_y)
                
                # Kiểm tra kích thước hợp lệ
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Tạo rectangle
                rect = QRectF(QPointF(x1, y1), QPointF(x2, y2))
                
                # Tạo label đúng cho box con
                label = f"{base_label}{index}"
                index += 1
                
                # Tạo preview shape
                shape = Shape(label)
                ret, points = shape.get_points(rect)
                if ret:
                    shape.points = points
                    shape.preview = True  # Đánh dấu là preview shape
                    shape.lock = False  # Preview shapes không bị lock
                    
                    # Thêm vào danh sách shapes để hiển thị
                    self.shapes.append(shape)
                    preview_shapes.append(shape)
        
        return preview_shapes

    def selectAll(self):
        """Chọn tất cả các shape trên canvas"""
        if len(self.shapes) == 0:
            QMessageBox.information(self, "Select All", "No shapes to select!")
            return
        
        # Bỏ chọn tất cả trước
        self.cancel_selected()
        
        # Chọn tất cả shapes (không bị hide)
        selected_count = 0
        for i, shape in enumerate(self.shapes):
            if not shape.hide:
                shape.selected = True
                selected_count += 1
                if self.idSelected is None:  # Set idSelected là shape đầu tiên được chọn
                    self.idSelected = i

    def active_edit(self):
        self.edit = True

    def show_grid(self, b_show=True):
        shape:Shape = None
        for shape in self:
            if "P" in shape.label:
                if b_show:
                    shape.hide = False
                else:
                    shape.hide = True

    def change_hide(self):
        index = self.idSelected
        s:Shape = None

        if index is not None:
            s:Shape = self[index]
            if s.hide:
                s.hide = False
            else:
                s.hide = True
    
    def change_hide_all(self):
        s:Shape = None
        self.cancel_selected()
        if self.actions.hide_all.text() == "Hide All":
            self.actions.hide_all.setText("Show All")
            for s in self:
                s.hide = True
        else:
            self.actions.hide_all.setText("Hide All")
            for s in self:
                s.hide = False

    def change_lock(self):
        index = self.idSelected
        s:Shape = None

        if index is not None:
            s:Shape = self[index]
            if s.lock:
                s.lock = False
            else:
                s.lock = True
    
    def change_lock_all(self):
        s:Shape = None
        
        if self.actions.lock_all.text() == "Lock All":
            self.actions.lock_all.setText("UnLock All")
            for s in self:
                s.lock = True
        else:
            self.actions.lock_all.setText("Lock All")
            for s in self:
                s.lock = False
    

    def set_benable_drawing(self,enable):
        self.benable_drawing = enable
        if self.benable_drawing : 
            self.actions.disable_drawing.setText("Disable drawing")
        else:
            self.actions.disable_drawing.setText("Enable drawing")
            
    def setEnabledActions(self,enable):
        self.actions.copy.setEnabled(enable)
        self.actions.edit.setEnabled(enable)
        self.actions.delete.setEnabled(enable)
        self.actions.delete_all.setEnabled(enable)
        self.actions.lock.setEnabled(enable)
        self.actions.create_grid.setEnabled(enable)  # create_grid cần shape được chọn
        # select_all luôn enabled nếu có shapes
        pass

    def popUpMenu(self):
        if self.idSelected is None:
            self.setEnabledActions(False)
            # create_grid cần shape được chọn nên disable khi không có selection
            self.actions.create_grid.setEnabled(False)
        else:
            self.setEnabledActions(True)
            s:Shape = self[self.idSelected]
            if s.lock:
                self.actions.lock.setText("UnLock")
            else:
                self.actions.lock.setText("Lock")

        # select_all luôn enabled nếu có shapes
        self.actions.select_all.setEnabled(len(self.shapes) > 0)
        
        # crop_image luôn enabled nếu có image và shapes
        self.actions.crop_image.setEnabled(self.picture is not None and len(self.shapes) > 0)

        if self.bcontext_menu:
            self.contextMenu.exec_(QCursor.pos())

    # ... (phần còn lại của code giữ nguyên như trong file gốc)
    
    def emitAction(self,name):
        self.actionSignal.emit(name)

    def focus_cursor(self):
        cur_pos = self.mapFromGlobal(QCursor().pos())
        return self.transformPos(cur_pos)

    def offset_center(self):
        dx = self.width() - self.picture.width()*self.scale
        dy = self.height()- self.picture.height()*self.scale
        pos = QPointF(dx/2,dy/2)
        self.org = pos
        return pos
    
    _b_full_screen = False
    _old_parent = None
    _geometry = None
    def on_show_full_screen(self):
        if not self._b_full_screen:
            self.show_full_screen()
        else:
            self.cancel_full_screen()

    def show_full_screen(self):
        self._b_full_screen = True
        self._old_parent = self.parent()
        self._geometry = self.saveGeometry()
        self.setParent(None)
        self.showFullScreen()
        self.zoom_buttons[4].setIcon(newIcon("full_screen_off"))
        self.zoom_buttons[4].setToolTip("Full screen off")

    def cancel_full_screen(self):
        self._b_full_screen = False
        if self._old_parent is not None:
            self.setParent(self._old_parent)
            if hasattr(self._old_parent, 'setCentralWidget'):
                self._old_parent.setCentralWidget(self)
        if hasattr(self, 'zoom_buttons') and len(self.zoom_buttons) > 4:
            self.zoom_buttons[4].setIcon(newIcon("full_screen"))
            self.zoom_buttons[4].setToolTip("Full screen")

    def fit_window(self):
        if self.picture is None:
            return
        self.scale = self.scaleFitWindow()
        self.org = self.offset_center()

    def scaleFitWindow(self):
        e = 2.
        w1 = self.width() - 2
        h1 = self.height() - 2
        a1 = w1/h1
        w2 = self.picture.width()
        h2 = self.picture.height()
        a2 = w2/h2
        return w1/w2 if a2 >= a1 else h1/h2

    def zoom_origin(self):
        self.scale = 1
        self.org = QPointF()

    def zoom_manual(self,s):
        self.scale *= s
        self.zoomSignal.emit(self.scale)
        return
    
    def zoom_focus_cursor(self, s):
        old_scale = self.scale
        p1 = self.current_pos
        self.scale *= s
        # focus cursor pos
        self.org -= p1*self.scale-p1*old_scale
        
    def zoom_by_wheel(self,s):
        self.zoom_focus_cursor(s)
        # self.repaint()
        self.zoomSignal.emit(self.scale)
        return
    
    def transformPos(self,pos):
        '''
        convert main pos -> cv pos
        '''
        return (pos - self.org)/(self.scale + 1e-5)
    
    def move_org(self,point):
        self.org += point

    def update_center(self,pos):
        pass
    
    def draw_rect(self,pos1,pos2):
        self.current_rect = QRectF(pos1,pos2)
        
    def shape_to_cvRect(self,shape):
        p1 = shape.points[0]
        p2 = shape.points[2]
        x , y = p1.x() , p1.y()
        x2 , y2 = p2.x() , p2.y()
        # 
        x = max(x,0)
        y = max(y,0)
        x2 = min(x2,self.picture.width())
        y2 = min(y2,self.picture.height())
        # 
        w , h = int(x2- x) , int(y2 - y)
        x , y = int(x) , int(y)
        # 
        return (x,y,w,h) 
        
    def editShape(self):
        if self.idSelected is not None:
            label = self.boxEditLabel.popUp(self.last_label,self.labels,bMove=True)
            if label:
                self[self.idSelected].label = label
                self.last_label = label
                self.append_new_label(label)

    def copyShape(self):
        if self.idSelected is not None:
            shape = self[self.idSelected].copy()
            self.shapes.append(shape)
            # self.dict_shapes[shape.label] = shape
            i = self.idSelected
            # self.releae_shape_selected(i)
            self.idSelected = i + 1

    def undo(self):
        if len(self.shapes) > 0:
            self.shapes.remove(self[-1])

    def deleteShape(self):
        if self.idSelected is not None:
            # self.emitAction("remove")
            shape = self[self.idSelected]
            self.deleteShapeSignal.emit(self.idSelected)
            self.shapes.remove(shape)
            # del(self.dict_shapes[shape.label])

            self.idVisible = self.idSelected = self.idCorner = None

    def delete_all(self):
        for i in range(len(self)):
            self.deleteShapeSignal.emit(len(self)-1)
            self.shapes.remove(self.shapes[-1])
        self.idVisible = self.idSelected = self.idCorner = None

    def moveShape(self,i,v):
        if self.picture is None:
            return
        self[i].move(v)
        self.moveShapeSignal.emit(i)

    def append_new_label(self,label):
        if label not in self.labels:
            self.labels.append(label)
            self.labels = [lb.strip("\r\n") for lb in self.labels]
            string = "\n".join(self.labels)
            if os.path.exists(self.label_path):
                with open(self.label_path, "w") as ff:
                    ff.write(string)
                    ff.close()
        pass

    def newShape(self,r,label):
        labels = [s.label for s in self.shapes]
        if label in labels:
            QMessageBox.warning(self, "WARNING", "Shape already exists")
            return
        # n = len(self)
        # label = "Shape-%d"%n
        shape = Shape(label)
        ret, points = shape.get_points(r)
        if ret:
            shape.points = points
            
            # Áp dụng trạng thái lock hiện tại từ Lock All button
            if hasattr(self.actions, 'lock_all'):
                # Nếu button text là "UnLock All" nghĩa là đang trong trạng thái locked
                shape.lock = (self.actions.lock_all.text() == "UnLock All")
            else:
                # Fallback: mặc định lock nếu chưa có actions
                shape.lock = True
                
            self.shapes.append(shape)
            # self.dict_shapes[label] = shape
            self.newShapeSignal.emit(len(self)-1)
            self.last_label = label
            self.append_new_label(label)
        return shape
    
    def format_shape(self,shape):
        label = shape.label
        r = self.shape_to_cvRect(shape)
        id = self.shapes.index(shape)
        return {
            "label" : label,
            "box" : r,
            "id" : id
        }

    def pos_in_shape(self,pos,shape):
        pass

    def visibleShape(self,pos):
        n = len(self)
        ids_shape_contain_pos = []
        distances = []
        for i in range(n):
            self[i].visible = False
            d = self[i].dis_to(pos)
            if d > 0:
                ids_shape_contain_pos.append(i)
                distances.append(d)

        if len(distances) > 0:
            index = np.argmin(distances)
            self.idVisible = ids_shape_contain_pos[index]
            self[self.idVisible].visible = True
            # self.visi.emit(self.idSelected)
        else:
            self.idVisible = None
        return self.idVisible
    
    def findShapeGrid(self, shape_index):
        """Tìm grid mà shape thuộc về"""
        for grid in self.all_grids:
            if shape_index in grid['indices']:
                return grid['indices']
        return None

    def selectedShape(self, pos):
        ids_shape_contain_pos = []
        distances = []
        
        s:Shape = None
        for i, s in enumerate(self.shapes):
            if not s.hide:
                d = s.dis_to(pos)
                if d > 0:
                    ids_shape_contain_pos.append(i)
                    distances.append(d)

        if len(distances) > 0:
            index = np.argmin(distances)
            selected_index = ids_shape_contain_pos[index]
            
            # Kiểm tra xem đang có multiple selection không
            currently_selected = [i for i, shape in enumerate(self.shapes) if shape.selected]
            
            # Tìm grid mà shape được click thuộc về
            clicked_shape_grid = self.findShapeGrid(selected_index)
            
            if len(currently_selected) > 1:
                # Nếu đang có multiple selection
                if selected_index in currently_selected:
                    # Click vào shape đang được chọn trong multi-selection → giữ nguyên selection
                    self.idSelected = selected_index
                    # KHÔNG thay đổi selection hiện tại
                    return  # Return sớm để không thực hiện logic bên dưới
                else:
                    # Click vào shape không được chọn
                    if clicked_shape_grid:
                        # Nếu thuộc grid → clear all và chọn grid
                        for shape in self.shapes:
                            shape.selected = False
                        for idx in clicked_shape_grid:
                            if 0 <= idx < len(self.shapes):
                                self.shapes[idx].selected = True
                        self.idSelected = selected_index
                    else:
                        # Nếu là shape đơn lẻ → clear all và chọn shape đó
                        for shape in self.shapes:
                            shape.selected = False
                        self[selected_index].selected = True
                        self.idSelected = selected_index
            elif clicked_shape_grid:
                # Clear existing selections first
                for shape in self.shapes:
                    shape.selected = False
                # Chọn tất cả boxes trong grid
                for idx in clicked_shape_grid:
                    if 0 <= idx < len(self.shapes):
                        self.shapes[idx].selected = True
                self.idSelected = selected_index
            else:
                # Clear existing selections first
                for shape in self.shapes:
                    shape.selected = False
                # Chỉ chọn box đó
                self[selected_index].selected = True
                self.idSelected = selected_index
                
            self.selectedShapeSignal.emit(self.idSelected)
        else:
            self.idSelected = None

    def highlightCorner(self,pos,epsilon=10):
        if self.idSelected is None:
            return False
        try:
            i = self.idSelected
            return self[i].get_corner(pos,epsilon)        
        except Exception as ex:
            print("{}".format(ex))   
            return False 

    def cancel_edit(self):
        self.edit = False
        self.drawing = False
        self.moving = False

    def cancel_selected(self):
        n = len(self)
        for i in range(n):
            self[i].selected = False
            self[i].corner = None
            self[i].visible = False
        self.idSelected = None
        
    def paintEvent(self, event):
        r: QRect = self.geometry()
        self.label_pos.setGeometry(0, r.height() - 30, r.width(), 30)

        w = 450
        h = 65
        self.tool_zoom.setGeometry(int((r.width() - w)/2) + 50, 30, w, h)

        if self.picture is None:
            return super(Canvas,self).paintEvent(event)
        
        p:QPainter = self.painter
        p.begin(self)
        lw = max(int(Shape.THICKNESS/(self.scale + 1e-5)), 1)
        p.setPen(QPen(Qt.green, lw))
        p.translate(self.org)
        p.scale(self.scale, self.scale)

        if self.picture:
            p.drawPixmap(0, 0, self.picture)
        
        shape:Shape = None
        for shape in self.shapes: 
            shape.paint(p, self.scale)
                
        if self.edit :
            # draw center
            pos = self.current_pos
            self.line1 = [QPointF(0,pos.y()),QPointF(self.picture.width(),pos.y())]
            self.line2 = [QPointF(pos.x(),0),QPointF(pos.x(),self.picture.height())]
            p.drawLine(self.line1[0],self.line1[1])
            p.drawLine(self.line2[0],self.line2[1])
        
        if self.drawing :    # draw rect
            if self.current is not None:
                p.drawRect(self.current)

        self.update()
        p.end()

        return super().paintEvent(event)
    
    def wheelEvent(self, ev):
        if self.picture is None:
            return super(Canvas,self).wheelEvent(ev)
        delta = ev.angleDelta()
        h_delta = delta.x()
        v_delta = delta.y()
        mods = ev.modifiers()
        # if Qt.ControlModifier == int(mods) and v_delta:
        if v_delta:
            self.zoom_by_wheel(1+v_delta/120*.2)
        # else:
        #     pos = QPointF(0.,v_delta/8.)
        #     self.move_org(pos)
        #     pass
        
        ev.accept()

    def mousePressEvent(self,ev):
        if self.picture is None :
            return super(Canvas,self).mousePressEvent(ev)
        # pos = self.transformPos(ev.pos())
        self.start_pos = self.transformPos(ev.pos())
        if ev.button() == Qt.LeftButton:
            if self.edit:
                if self.idSelected is not None:
                    self[self.idSelected].selected = False
                    self.idSelected = None
                self.drawing = True
            else:
                self.moving = True
                if not self.highlight:
                    # Kiểm tra xem có click vào shape nào không
                    old_selected = self.idSelected
                    self.selectedShape(self.start_pos)
                    
                    # Nếu không click vào shape nào, hủy tất cả selections
                    if self.idSelected is None:
                        self.cancel_selected()
    
    def mouseReleaseEvent(self,ev):
        if self.picture is None :
            return super(Canvas,self).mouseReleaseEvent(ev)
        # pos = self.transformPos(ev.pos())
        self.move_shape = False
        if ev.button() == Qt.LeftButton:
            if self.drawing: 
                r = self.current
                if r is not None and r.width() > Shape.MIN_WIDTH and r.height() > Shape.MIN_WIDTH:
                    label = self.boxEditLabel.popUp(self.last_label,self.labels,bMove=False)
                    if label:
                        self.newShape(r,label)
                self.current = None
       
            self.cancel_edit()
    
    def mouseMoveEvent(self, ev):
        if self.picture is None :
            return super(Canvas,self).mouseMoveEvent(ev)

        self.current_pos:QPointF = self.transformPos(ev.pos())

        image = self.picture.toImage()
        try:
            pos:QPoint = self.current_pos.toPoint()
            if self.picture.width() > pos.x() >= 0 and self.picture.height() > pos.y() >= 0:
                pixel:QColor = image.pixelColor(pos)
                h, s, v, _ = pixel.getHsv()
                r, g, b, _ = pixel.getRgb()
                x, y = pos.x(), pos.y()
                self.text_pixel_color = "POS: [%d, %d], BGR: [%d, %d, %d], HSV: [%d, %d, %d]" % (x, y, b, g, r, h, s, v)
                self.label_pos.setText(self.text_pixel_color)
        except Exception as ex:
            pass

        self.mouseMoveSignal.emit(self.current_pos)
        if self.drawing:
            self.current = QRectF(self.start_pos,self.current_pos)
            self.drawShapeSignal.emit(self.current)
            # self.override_cursor(CURSOR_MOVE)
        if not self.moving:
            self.highlight = self.highlightCorner(self.current_pos,epsilon=40)
            if self.highlight:
                # self.override_cursor(CURSOR_DRAW)
                pass
        if self.moving:
            v = self.current_pos - self.start_pos
            index = self.idSelected
            
            # Đếm số shapes được chọn
            selected_shapes = [i for i, shape in enumerate(self.shapes) if shape.selected]
            
            if len(selected_shapes) > 1:
                # PRIORITY 1: Nếu có nhiều shapes được chọn, di chuyển tất cả (bao gồm cả Select All)
                for idx in selected_shapes:
                    if not self.shapes[idx].lock:
                        self.shapes[idx].move(v)
                        self.moveShapeSignal.emit(idx)
            elif index is not None and 0 <= index < len(self.shapes):
                # PRIORITY 2: Nếu chỉ có 1 shape được chọn
                shape_grid = self.findShapeGrid(index)
                if shape_grid and len(shape_grid) > 1:
                    # Nếu thuộc grid và grid có nhiều hơn 1 shape, di chuyển tất cả boxes trong grid
                    for idx in shape_grid:
                        if 0 <= idx < len(self.shapes) and not self.shapes[idx].lock:
                            self.shapes[idx].move(v)
                            self.moveShapeSignal.emit(idx)
                elif not self[index].lock:
                    # Nếu không thuộc grid hoặc grid chỉ có 1 shape, di chuyển bình thường
                    s = self[index]
                    if self.highlight:
                        s.change(v)
                    else:
                        s.move(v)
            else:
                self.move_org(v*self.scale)

            self.start_pos = self.transformPos(ev.pos())
            if self.idSelected is not None:
                self.changeShapeSignal.emit(self.idSelected)
            # self.override_cursor(CURSOR_MOVE)

        if self.visibleShape(self.current_pos) is None and not self.highlight and not self.drawing:
            self.restore_cursor()
        elif not self.highlight and not self.drawing and not self.moving:
            pass
            # self.override_cursor(CURSOR_GRAB)
        if self.edit :
            pass
            # self.restore_cursor()

    def currentCursor(self):
        cursor = QApplication.overrideCursor()
        if cursor is not None:
            cursor = cursor.shape()
        return cursor
    
    def overrideCursor(self, cursor):
        self._cursor = cursor
        if self.currentCursor() is None:
            QApplication.setOverrideCursor(cursor)
        else:
            QApplication.changeOverrideCursor(cursor)

    def resizeEvent(self,ev):
        if self.picture is None:
            return super(Canvas,self).resizeEvent(ev)
        self.fit_window()
        pass

    def keyPressEvent(self, ev):
        key = ev.key()
        step = 5
        # if key == Qt.Key_1:
        #     self.parent.io_signal = not self.parent.io_signal
        if key == Qt.Key_W:
            if self.benable_drawing:
                self.edit = True

        elif key == Qt.Key_Escape:
            self.cancel_edit()
            self.cancel_selected()
            if self._b_full_screen:
              self.cancel_full_screen()

        elif key == Qt.Key_A and ev.modifiers() == Qt.ControlModifier:
            # Ctrl+A để chọn tất cả
            self.selectAll()

        elif key == Qt.Key_Delete:
            self.deleteShape()
        
        elif key == Qt.Key_Return:
            self.fit_window()
        
        elif key == Qt.Key_Plus:
            s = 1.2
            self.zoom_focus_cursor(s)

        elif key == Qt.Key_Minus:
            s = 0.8
            self.zoom_focus_cursor(s)

        i = self.idSelected
        if i is not None :
            if key == Qt.Key_Right:
                v = QPointF(step,0)
                self.moveShape(i,v)
            elif key == Qt.Key_Left:
                v = QPointF(-step,0)
                self.moveShape(i,v)
            elif key == Qt.Key_Up:
                v = QPointF(0,-step)
                self.move_org(v)
            elif key == Qt.Key_Down:
                v = QPointF(0,step)
                self.move_org(v)
        
    def load_pixmap(self,pixmap,fit=False):
        self.picture = pixmap
        if fit:
            self.fit_window()
        self.zoomSignal.emit(self.scale)
        self.repaint()

    def apply_current_lock_state_to_all(self):
        """Áp dụng trạng thái lock cho tất cả shapes - mặc định luôn lock"""
        for shape in self.shapes:
            shape.lock = True

    def add_loaded_shape(self, shape):
        """Method để thêm shape từ file load"""
        # Mặc định lock tất cả shapes được load
        shape.lock = True
        
        self.shapes.append(shape)
        self.newShapeSignal.emit(len(self.shapes) - 1)
        return shape

    def load_shapes_from_data(self, shapes_data):
        """Load multiple shapes từ data và apply lock state"""
        for shape_data in shapes_data:
            # Tạo shape từ data (bạn cần customize theo format của bạn)
            shape = Shape(shape_data.get('label', 'Unknown'))
            # Set points, properties khác...
            # shape.points = shape_data.get('points', [])
            
            # Mặc định lock tất cả shapes được load
            shape.lock = True
                
            self.shapes.append(shape)
        
        # Emit signals và update
        for i in range(len(shapes_data)):
            self.newShapeSignal.emit(len(self.shapes) - len(shapes_data) + i)

    def current_cursor(self):
        cursor = QApplication.overrideCursor()
        if cursor is not None:
            cursor = cursor.shape()
        return cursor

    def override_cursor(self, cursor):
        self._cursor = cursor
        if self.current_cursor() is None:
            QApplication.setOverrideCursor(cursor)
        else:
            QApplication.changeOverrideCursor(cursor)

    def restore_cursor(self):
        QApplication.restoreOverrideCursor()

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, key):
        # if isinstance(key,int):
         return self.shapes[key]
        # elif isinstance(key,str):
        #     return self.dict_shapes[key]

    def __setitem__(self, key, value):
        # Áp dụng trạng thái lock hiện tại khi set
        if hasattr(value, 'lock'):
            if hasattr(self.actions, 'lock_all'):
                value.lock = (self.actions.lock_all.text() == "UnLock All")
            else:
                value.lock = True
        self.shapes[key] = value

    def append_shape(self, shape):
        """Method an toàn để thêm shape - áp dụng lock state hiện tại"""
        if hasattr(shape, 'lock'):
            # Áp dụng trạng thái lock hiện tại
            if hasattr(self.actions, 'lock_all'):
                shape.lock = (self.actions.lock_all.text() == "UnLock All")
            else:
                shape.lock = True
        self.shapes.append(shape)
        return len(self.shapes) - 1

    def force_lock_all_shapes(self):
        """Force lock tất cả shapes hiện có - gọi sau khi load file"""
        for shape in self.shapes:
            if hasattr(shape, 'lock'):
                shape.lock = True
        # print(f"Forced lock on {len(self.shapes)} shapes")

    def clear_pixmap(self):
        self.picture = None
        
    def clear(self):
        self.shapes.clear()
        self.idSelected = None
        self.idVisible = None
        self.idCorner = None
        self.last_created_grid = []
        self.all_grids = []


class WindowCanvas(QMainWindow):
    def __init__(self, canvas=None, parent=None):
        super().__init__(parent=parent)
        self.setCentralWidget(canvas)
        self.setObjectName("WindowCanvas")


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    # wd = QMainWindow()
    
    canvas = Canvas()
    canvas.load_pixmap(QPixmap(640,480))

    # wd.setCentralWidget(canvas)
    canvas.showMaximized()

    sys.exit(app.exec_())