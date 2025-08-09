import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class CameraPlaceholder(QFrame):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 150)
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(2)
        self.setStyleSheet("background-color: #2e2e2e; color: white;")
        label = QLabel(text, self)
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Arial", 14, QFont.Bold))
        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)


class CircleButton(QPushButton):
    def __init__(self, text, color="#6699CC"):
        super().__init__(text)
        size = 150  # Bigger circle size
        self.setFixedSize(size, size)
        self.setFont(QFont("Arial", 16, QFont.Bold))
        self.setStyleSheet(f"""
            QPushButton {{
                border: 3px solid #444;
                border-radius: {size // 2}px;
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


class CounterBox(QFrame):
    def __init__(self, label_text, bg_color="#f9f9f9"):
        super().__init__()
        self.setStyleSheet(f"""
            QFrame {{
                border: 2px solid #666666;
                border-radius: 8px;
                padding: 10px;
                background-color: {bg_color};
            }}
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(5)

        label = QLabel(label_text)
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Arial", 18))

        self.count_label = QLabel("0")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setFont(QFont("Arial", 36, QFont.Bold))

        layout.addWidget(label)
        layout.addWidget(self.count_label)

        self.setLayout(layout)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisioCup Project")
        self.resize(1000, 800)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Top Labels
        top_labels_layout = QVBoxLayout()
        top_labels_layout.setSpacing(0)

        texa_label = QLabel("TEXA")
        texa_label.setAlignment(Qt.AlignCenter)
        texa_label.setFont(QFont("Arial", 32, QFont.Bold))
        top_labels_layout.addWidget(texa_label)

        project_label = QLabel("VisioCup Project")
        project_label.setAlignment(Qt.AlignCenter)
        project_label.setFont(QFont("Arial", 24))
        top_labels_layout.addWidget(project_label)

        main_layout.addLayout(top_labels_layout)

        # Middle Layout with left buttons, cameras, right buttons
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(20)

        # Left buttons (circle, colored, changed text/color)
        texts_left = ["Stop", "start", "Reset"]  # Changed Start -> Go, Restart -> Reset
        colors_left = ["#d9534f", "#28a745", "#007bff"]  # red, green, blue

        left_buttons_layout = QVBoxLayout()
        left_buttons_layout.setSpacing(30)
        for text, color in zip(texts_left, colors_left):
            btn = CircleButton(text, color)
            left_buttons_layout.addWidget(btn)
        left_buttons_layout.addStretch()
        middle_layout.addLayout(left_buttons_layout, stretch=1)

        # Camera grid (2x2)
        camera_grid = QGridLayout()
        camera_grid.setSpacing(20)
        for i in range(4):
            cam = CameraPlaceholder(f"Camera {i + 1}")
            row = i // 2
            col = i % 2
            camera_grid.addWidget(cam, row, col)
        middle_layout.addLayout(camera_grid, stretch=6)

        # Right buttons (circle, colored)
        colors_right = ["#0275d8", "#5bc0de"]  # blue, light blue
        right_buttons_layout = QVBoxLayout()
        right_buttons_layout.setSpacing(30)
        for text, color in zip(["Auto", "Manual"], colors_right):
            btn = CircleButton(text, color)
            right_buttons_layout.addWidget(btn)
        right_buttons_layout.addStretch()
        middle_layout.addLayout(right_buttons_layout, stretch=1)

        main_layout.addLayout(middle_layout)

        # Bottom counters with colored boxes
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(80)

        bottom_layout.addWidget(CounterBox("Good", bg_color="#d4edda"))       # light green background
        bottom_layout.addWidget(CounterBox("Bad (Parts)", bg_color="#f8d7da"))  # light red background
        bottom_layout.addWidget(CounterBox("Total", bg_color="#d1ecf1"))       # light blue background

        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
