import sys
import cv2
import os
import json
import joblib
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QFileDialog, QFrame
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPolygon
from PyQt5.QtCore import Qt, QTimer, QPoint
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class CameraFeed(QLabel):
    def __init__(self, cam_index, roi_points=None):
        super().__init__()
        self.cap = cv2.VideoCapture(cam_index)
        self.setFixedSize(320, 240)
        self.setStyleSheet("border: 2px solid #ccc; border-radius: 8px;")
        self.roi_points = roi_points

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.roi_points is not None:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(self.roi_points, dtype=np.int32)], 255)
            frame = cv2.bitwise_and(frame, frame, mask=mask)

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)


class DiamondButton(QPushButton):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        size = min(self.width(), self.height()) - 10
        center = self.rect().center()
        points = [
            QPoint(center.x(), center.y() - size // 2),
            QPoint(center.x() + size // 2, center.y()),
            QPoint(center.x(), center.y() + size // 2),
            QPoint(center.x() - size // 2, center.y()),
        ]
        polygon = QPolygon(points)
        painter.setBrush(QColor("#FFD700"))
        painter.drawPolygon(polygon)
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, self.text())


class QualiScanApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QUALI SCAN")
        self.setStyleSheet("background-color: #f5f5f5;")

        # Load ROI if exists
        self.roi_points = None
        if os.path.exists("roi.json"):
            with open("roi.json", "r") as f:
                self.roi_points = json.load(f)

        # Cameras
        self.cameras = [CameraFeed(i, self.roi_points) for i in [1, 2, 3, 4]]

        # Counters
        self.good_count, self.bad_count, self.total_count = 0, 0, 0

        # Layout
        self.init_ui()

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)

    def init_ui(self):
        # Top bar
        top_bar = QHBoxLayout()
        texa_label = QLabel("TEXA")
        texa_label.setFont(QFont("Arial", 20, QFont.Bold))
        texa_label.setStyleSheet("color: #333;")
        title_label = QLabel("QUALI SCAN")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        top_bar.addWidget(texa_label)
        top_bar.addStretch()
        top_bar.addWidget(title_label)
        top_bar.addStretch()

        # Camera grid
        cam_grid = QGridLayout()
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for pos, cam in zip(positions, self.cameras):
            cam_grid.addWidget(cam, *pos)

        # Left buttons
        left_panel = QVBoxLayout()
        start_btn = QPushButton("Start")
        stop_btn = QPushButton("Stop")
        restart_btn = QPushButton("Restart")
        mode_btn = QPushButton("Auto / Manual")
        for btn, color in [(start_btn, "#4CAF50"), (stop_btn, "#F44336"), (restart_btn, "#FF9800")]:
            btn.setStyleSheet(f"background-color: {color}; color: white; padding: 10px; border-radius: 10px;")
            left_panel.addWidget(btn)
        mode_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 10px; border-radius: 10px;")
        left_panel.addWidget(mode_btn)
        left_panel.addStretch()

        start_btn.clicked.connect(self.start_detection)
        stop_btn.clicked.connect(self.stop_detection)

        # Right counters
        right_panel = QVBoxLayout()
        self.good_label = self.make_circle_label("Good: 0", "#4CAF50")
        self.bad_label = self.make_circle_label("Bad: 0", "#F44336")
        self.total_label = self.make_circle_label("Total: 0", "#2196F3")
        for lbl in [self.good_label, self.bad_label, self.total_label]:
            right_panel.addWidget(lbl)
        right_panel.addStretch()

        # Bottom bar
        bottom_bar = QHBoxLayout()
        train_btn = QPushButton("Train Model")
        train_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 10px; border-radius: 10px;")
        trigger_btn = DiamondButton("Trigger")
        trigger_btn.setFixedSize(100, 100)
        bottom_bar.addWidget(train_btn)
        bottom_bar.addWidget(trigger_btn)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_bar)
        content_layout = QHBoxLayout()
        content_layout.addLayout(left_panel)
        content_layout.addLayout(cam_grid)
        content_layout.addLayout(right_panel)
        main_layout.addLayout(content_layout)
        main_layout.addLayout(bottom_bar)

        self.setLayout(main_layout)

    def make_circle_label(self, text, color):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(f"background-color: {color}; color: white; border-radius: 50px; padding: 20px;")
        return lbl

    def update_frames(self):
        for cam in self.cameras:
            pixmap = cam.get_frame()
            if pixmap:
                cam.setPixmap(pixmap)

    def start_detection(self):
        self.timer.start(30)

    def stop_detection(self):
        self.timer.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QualiScanApp()
    window.show()
    sys.exit(app.exec_())
