import sys
import random
import shutil
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QLineEdit, QTextEdit, QFileDialog, QProgressBar,
    QGridLayout, QMessageBox, QSplashScreen
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt5.QtGui import QFont, QPixmap, QPainter, QColor

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ================= SPLASH SCREEN =================
def show_splash(app):
    pixmap = QPixmap(520, 300)
    pixmap.fill(QColor("#121212"))

    painter = QPainter(pixmap)
    painter.setPen(QColor("#2979FF"))
    painter.setFont(QFont("Segoe UI", 26, QFont.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "AI Dataset Splitter")

    painter.setFont(QFont("Segoe UI", 11))
    painter.setPen(QColor("#AAAAAA"))
    painter.drawText(
        pixmap.rect().adjusted(0, 110, 0, 0),
        Qt.AlignCenter,
        "Loading modules..."
    )
    painter.end()

    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()
    return splash


# ================= WORKER =================
class DatasetWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(self, input_dir, output_dir, train_ratio):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self._running = True

        self.total_files = 0
        self.current = 0

    def stop(self):
        self._running = False

    def run(self):
        try:
            self.log.emit("🔍 Scanning input folder...")

            images = [
                p.name for p in self.input_dir.iterdir()
                if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")
            ]

            if not images:
                raise RuntimeError("No images found in input folder")

            random.shuffle(images)
            split_idx = int(len(images) * self.train_ratio)

            train_files = images[:split_idx]
            val_files = images[split_idx:]

            self.total_files = len(train_files) + len(val_files)
            self.current = 0

            classes_file = self.input_dir / "classes.txt"
            if not classes_file.exists():
                raise FileNotFoundError("classes.txt not found in input folder")

            self.process_split(train_files, "Train")
            if not self._running:
                self.cancelled.emit()
                return

            self.process_split(val_files, "Validation")
            if not self._running:
                self.cancelled.emit()
                return

            self.finished.emit()
            self.log.emit("✅ Dataset conversion completed")

        except Exception as e:
            self.log.emit(f"❌ ERROR: {e}")

    def process_split(self, files, split_name):
        img_dir = self.output_dir / split_name / "images" / split_name
        lbl_dir = self.output_dir / split_name / "labels" / split_name
        txt_path = self.output_dir / split_name / f"{split_name}.txt"

        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        with open(txt_path, "w", encoding="utf-8") as f:
            for img in files:
                if not self._running:
                    self.log.emit("⛔ Task cancelled by user")
                    return

                src_img = self.input_dir / img
                src_lbl = self.input_dir / img.rsplit(".", 1)[0] + ".txt"

                self.current += 1
                self.progress.emit(self.current, self.total_files)

                if not src_img.exists() or not src_lbl.exists():
                    self.log.emit(f"⚠️ Missing pair ({self.current}/{self.total_files}): {img}")
                    continue

                shutil.copy2(src_img, img_dir / img)
                shutil.copy2(src_lbl, lbl_dir / src_lbl.name)

                f.write(f"data/images/{split_name}/{img}\n")

                self.log.emit(f"📄 Processing {self.current} / {self.total_files} : {img}")


# ================= UI =================
class DatasetUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Dataset Splitter")
        self.setMinimumSize(900, 600)

        self.worker = None
        self.settings = QSettings("AICompany", "DatasetSplitter")

        self.init_ui()
        self.apply_dark_theme()
        self.load_settings()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QGridLayout(central)

        title = QLabel("AI Dataset Splitter")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))

        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.ratio_edit = QLineEdit("0.8")

        btn_input = QPushButton("📂 Input Folder")
        btn_output = QPushButton("📁 Output Folder")
        self.btn_convert = QPushButton("🚀 Convert")
        self.btn_cancel = QPushButton("⛔ Cancel")
        self.btn_cancel.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setFormat("%p%  (%v / %m files)")
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        btn_input.clicked.connect(self.choose_input)
        btn_output.clicked.connect(self.choose_output)
        self.btn_convert.clicked.connect(self.start_convert)
        self.btn_cancel.clicked.connect(self.cancel_task)

        layout.addWidget(title, 0, 0, 1, 3)

        layout.addWidget(QLabel("Input Folder"), 1, 0)
        layout.addWidget(self.input_edit, 1, 1)
        layout.addWidget(btn_input, 1, 2)

        layout.addWidget(QLabel("Output Folder"), 2, 0)
        layout.addWidget(self.output_edit, 2, 1)
        layout.addWidget(btn_output, 2, 2)

        layout.addWidget(QLabel("Train Ratio"), 3, 0)
        layout.addWidget(self.ratio_edit, 3, 1)

        layout.addWidget(self.btn_convert, 4, 0, 1, 2)
        layout.addWidget(self.btn_cancel, 4, 2)

        layout.addWidget(self.progress, 5, 0, 1, 3)
        layout.addWidget(self.log_box, 6, 0, 1, 3)

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #E0E0E0;
                font-size: 13px;
            }
            QLineEdit, QTextEdit {
                background-color: #1E1E1E;
                border: 1px solid #333;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton {
                background-color: #2979FF;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #448AFF;
            }
            QPushButton:disabled {
                background-color: #555;
            }
            QProgressBar {
                border-radius: 6px;
                background: #2A2A2A;
            }
            QProgressBar::chunk {
                background-color: #00E676;
            }
        """)

    def load_settings(self):
        self.input_edit.setText(self.settings.value("input_dir", ""))
        self.output_edit.setText(self.settings.value("output_dir", ""))
        self.ratio_edit.setText(self.settings.value("ratio", "0.8"))

    def save_settings(self):
        self.settings.setValue("input_dir", self.input_edit.text())
        self.settings.setValue("output_dir", self.output_edit.text())
        self.settings.setValue("ratio", self.ratio_edit.text())

    def choose_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_edit.setText(folder)

    def choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_edit.setText(folder)

    def start_convert(self):
        if not self.input_edit.text() or not self.output_edit.text():
            QMessageBox.warning(self, "Error", "Please select input and output folders")
            return

        self.save_settings()
        self.log_box.clear()
        self.progress.setValue(0)

        self.worker = DatasetWorker(
            self.input_edit.text(),
            self.output_edit.text(),
            float(self.ratio_edit.text())
        )

        self.worker.log.connect(self.log_box.append)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.cancelled.connect(self.on_cancelled)

        self.btn_convert.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self.worker.start()

    def update_progress(self, current, total):
        self.progress.setMaximum(total)
        self.progress.setValue(current)

    def cancel_task(self):
        if self.worker:
            self.worker.stop()
            self.log_box.append("⛔ Cancelling...")

    def on_finished(self):
        self.btn_convert.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        QMessageBox.information(self, "Done", "Dataset conversion completed")

    def on_cancelled(self):
        self.btn_convert.setEnabled(True)
        self.btn_cancel.setEnabled(False)


# ================= MAIN =================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    splash = show_splash(app)

    win = DatasetUI()
    win.show()

    splash.finish(win)
    sys.exit(app.exec_())
