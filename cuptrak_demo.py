import sys
import os
import cv2
import json
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QFileDialog, QMessageBox, QSizePolicy, QFrame
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer

CAMERA_INDEXES = [1, 2, 3, 4]
SAVE_PATH_FILE = "save_path.txt"

# --- Circle Button ---
class CircleButton(QPushButton):
    def __init__(self, text, color="#6699CC", diameter=100):
        super().__init__(text)
        self.diameter = diameter
        self.setFixedSize(self.diameter, self.diameter)
        self.setFont(QFont("Arial", 14, QFont.Bold))
        self.setStyleSheet(f"""
            QPushButton {{
                border: 3px solid #444;
                border-radius: {self.diameter // 2}px;
                background-color: {color};
                color: white;
            }}
            QPushButton:hover {{
                background-color: {self._darker_color(color, 0.85)};
            }}
            QPushButton:pressed {{
                background-color: {self._darker_color(color, 0.7)};
            }}
        """)

    def _darker_color(self, hex_color, factor=0.8):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = max(0, min(255, int(r * factor)))
        g = max(0, min(255, int(g * factor)))
        b = max(0, min(255, int(b * factor)))
        return f'#{r:02x}{g:02x}{b:02x}'

# --- Counter Box ---
class CounterBox(QFrame):
    def __init__(self, label_text, bg_color="#f9f9f9"):
        super().__init__()
        self.setStyleSheet(f"""
            QFrame {{
                border: 3px solid #666;
                border-radius: 10px;
                padding: 15px;
                background-color: {bg_color};
            }}
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        label = QLabel(label_text)
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Arial", 18, QFont.Bold))
        self.count_label = QLabel("0")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setFont(QFont("Arial", 32, QFont.Bold))

        layout.addWidget(label)
        layout.addWidget(self.count_label)
        self.setLayout(layout)

    def set_count(self, val):
        self.count_label.setText(str(val))

# --- Video Feed Widget ---
class VideoFeedWidget(QLabel):
    def __init__(self, cam_index):
        super().__init__()
        self.cam_index = cam_index
        self.cap = None
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black; border: 5px solid black; color: white;")
        self.setText(f"Camera {cam_index} not started")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def start(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            self.setText(f"Camera {self.cam_index} disconnected")
            self.cap = None

    def stop(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            self.setText(f"Camera {self.cam_index} stopped")
            self.clear()

    def read_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.setPixmap(pixmap)
                return frame
            else:
                self.setText(f"Camera {self.cam_index} frame error")
        else:
            self.setText(f"Camera {self.cam_index} stopped")
        return None

class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Texa Camera Software")
        self.setStyleSheet("background-color: #722F37; color: white;")

        self.good_count = 0
        self.bad_count = 0
        self.total_count = 0

        self.save_path = self.load_or_select_save_path()

        self.video_widgets = []
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.capturing = False

        self.init_ui()
        self.showMaximized()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Top labels
        top_layout = QVBoxLayout()
        texa_label = QLabel("TEXA")
        texa_label.setAlignment(Qt.AlignCenter)
        texa_label.setFont(QFont("Arial", 36, QFont.Bold))

        project_label = QLabel("VisioCup project")
        project_label.setAlignment(Qt.AlignCenter)
        project_label.setFont(QFont("Arial", 24))

        top_layout.addWidget(texa_label)
        top_layout.addWidget(project_label)
        main_layout.addLayout(top_layout)

        # Middle: Left buttons, camera grid, right buttons
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(20)

        # Left buttons
        left_btn_layout = QVBoxLayout()
        left_btn_layout.setSpacing(20)
        self.btn_start = CircleButton("Start", "#28a745")
        self.btn_stop = CircleButton("Stop", "#d9534f")
        self.btn_restart = CircleButton("Restart", "#007bff")
        left_btn_layout.addWidget(self.btn_stop)
        left_btn_layout.addWidget(self.btn_start)
        left_btn_layout.addWidget(self.btn_restart)
        left_btn_layout.addStretch()
        middle_layout.addLayout(left_btn_layout, 1)

        # Camera grid 2x2
        camera_grid = QGridLayout()
        camera_grid.setSpacing(15)
        for i, cam_index in enumerate(CAMERA_INDEXES):
            vw = VideoFeedWidget(cam_index)
            self.video_widgets.append(vw)
            camera_grid.addWidget(vw, i // 2, i % 2)
        middle_layout.addLayout(camera_grid, 6)

        # Right buttons
        right_btn_layout = QVBoxLayout()
        right_btn_layout.setSpacing(20)
        self.btn_auto = CircleButton("Auto", "#0275d8")
        self.btn_manual = CircleButton("Manual", "#5bc0de")
        right_btn_layout.addWidget(self.btn_auto)
        right_btn_layout.addWidget(self.btn_manual)
        right_btn_layout.addStretch()
        middle_layout.addLayout(right_btn_layout, 1)

        main_layout.addLayout(middle_layout)

        # Bottom counters
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(40)
        self.counter_good = CounterBox("Good", "#d4edda")
        self.counter_bad = CounterBox("Bad (Parts)", "#f8d7da")
        self.counter_total = CounterBox("Total", "#d1ecf1")
        bottom_layout.addWidget(self.counter_good)
        bottom_layout.addWidget(self.counter_bad)
        bottom_layout.addWidget(self.counter_total)
        main_layout.addLayout(bottom_layout)

        # Trigger button below cameras
        self.trigger_button = CircleButton("Trigger", "#ffc107", diameter=100)
        main_layout.addWidget(self.trigger_button, alignment=Qt.AlignCenter)

        self.setLayout(main_layout)

        # Connect button signals
        self.btn_start.clicked.connect(self.start_cameras)
        self.btn_stop.clicked.connect(self.stop_cameras)
        self.btn_restart.clicked.connect(self.restart_cameras)
        self.btn_auto.clicked.connect(lambda: self.set_mode("Auto"))
        self.btn_manual.clicked.connect(lambda: self.set_mode("Manual"))
        self.trigger_button.clicked.connect(self.trigger_snapshots)

    def load_or_select_save_path(self):
        if os.path.exists(SAVE_PATH_FILE):
            with open(SAVE_PATH_FILE, 'r') as f:
                path = f.read().strip()
                if os.path.isdir(path):
                    return path
        path = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if path:
            with open(SAVE_PATH_FILE, 'w') as f:
                f.write(path)
        else:
            QMessageBox.warning(self, "Warning", "No save folder selected. Exiting.")
            sys.exit()
        return path

    def start_cameras(self):
        if not self.capturing:
            for vw in self.video_widgets:
                vw.start()
            self.timer.start(30)
            self.capturing = True
            print("Cameras started")

    def stop_cameras(self):
        if self.capturing:
            self.timer.stop()
            for vw in self.video_widgets:
                vw.stop()
            self.capturing = False
            print("Cameras stopped")

    def restart_cameras(self):
        print("Restarting cameras...")
        self.stop_cameras()
        self.start_cameras()

    def set_mode(self, mode):
        print(f"Mode set to {mode}")
        # Implement your mode-specific logic here

    def update_frames(self):
        for vw in self.video_widgets:
            vw.read_frame()

    def trigger_snapshots(self):
        if not self.capturing:
            QMessageBox.warning(self, "Warning", "Cameras are not started!")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, vw in enumerate(self.video_widgets):
            frame = vw.read_frame()
            if frame is not None:
                cam_folder = os.path.join(self.save_path, f"camera{i+1}")
                os.makedirs(cam_folder, exist_ok=True)
                img_path = os.path.join(cam_folder, f"{timestamp}.bmp")
                cv2.imwrite(img_path, frame)
                print(f"Saved snapshot: {img_path}")

        # You can update counters here if you want
        self.total_count += 1
        self.counter_total.set_count(self.total_count)
        # For demo, increment good and bad randomly or leave as zero
        self.counter_good.set_count(self.good_count)
        self.counter_bad.set_count(self.bad_count)

        QMessageBox.information(self, "Done", "Snapshots captured.")

    def closeEvent(self, event):
        self.stop_cameras()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
