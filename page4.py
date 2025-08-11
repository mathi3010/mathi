import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QCheckBox, QGridLayout, QFrame, QMessageBox
)
from PyQt5.QtCore import Qt, QSize, QPoint, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPainter, QPolygon, QColor, QFont


class DiamondButton(QPushButton):
    def __init__(self, text='', parent=None):
        super().__init__(text, parent)
        self.setFixedSize(100, 100)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("color: white; font-weight: bold; font-size: 18px;")
        self._anim = QPropertyAnimation(self, b"windowOpacity")
        self._anim.setDuration(2000)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.75)
        self._anim.setLoopCount(-1)
        self._anim.setEasingCurve(QEasingCurve.InOutQuad)
        self._anim.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        points = QPolygon([
            QPoint(w//2, 0),
            QPoint(w, h//2),
            QPoint(w//2, h),
            QPoint(0, h//2),
        ])

        painter.setBrush(QColor(58, 134, 255))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(points)

        painter.setPen(QColor('white'))
        font = QFont('Segoe UI', 16, QFont.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, self.text())


class PageOne(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(50, 100, 50, 100)
        layout.setSpacing(50)

        company_label = QLabel("TEXA INNOVATES")
        company_label.setAlignment(Qt.AlignCenter)
        company_label.setStyleSheet("color: #3a86ff; font-weight: bold; font-size: 36px;")

        project_label = QLabel("CHROMA PLAST")
        project_label.setAlignment(Qt.AlignCenter)
        project_label.setStyleSheet("color: #eeeeee; font-weight: 600; font-size: 30px;")

        inspection_label = QLabel("Ice Cup Label Inspection")
        inspection_label.setAlignment(Qt.AlignCenter)
        inspection_label.setStyleSheet("color: #559dff; font-weight: 500; font-size: 26px;")

        self.btn_next = QPushButton("Next")
        self.btn_next.setCursor(Qt.PointingHandCursor)
        self.btn_next.setFixedSize(140, 50)
        self.btn_next.setStyleSheet("""
            QPushButton {
                background-color: #3a86ff;
                color: white;
                font-weight: bold;
                font-size: 18px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #559dff;
            }
        """)

        layout.addWidget(company_label)
        layout.addWidget(project_label)
        layout.addWidget(inspection_label)
        layout.addStretch()
        layout.addWidget(self.btn_next, alignment=Qt.AlignCenter)

        self.setLayout(layout)
        self.setStyleSheet("background-color: #121212;")


class PageTwo(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(25)

        name_label = QLabel("Operator Name:")
        name_label.setStyleSheet("color: #eeeeee; font-size: 16px; font-weight: 600;")
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter operator name")
        self.name_edit.setStyleSheet("padding: 8px; font-size: 16px;")

        id_label = QLabel("Operator ID:")
        id_label.setStyleSheet("color: #eeeeee; font-size: 16px; font-weight: 600;")
        self.id_edit = QLineEdit()
        self.id_edit.setPlaceholderText("Enter operator ID")
        self.id_edit.setStyleSheet("padding: 8px; font-size: 16px;")

        brand_label = QLabel("Brand for Label:")
        brand_label.setStyleSheet("color: #eeeeee; font-size: 16px; font-weight: 600;")
        self.brand_edit = QLineEdit()
        self.brand_edit.setPlaceholderText("Enter brand")
        self.brand_edit.setStyleSheet("padding: 8px; font-size: 16px;")

        category_label = QLabel("Category:")
        category_label.setStyleSheet("color: #eeeeee; font-size: 16px; font-weight: 600;")
        self.category_edit = QLineEdit()
        self.category_edit.setPlaceholderText("Enter category")
        self.category_edit.setStyleSheet("padding: 8px; font-size: 16px;")

        btn_layout = QHBoxLayout()
        self.btn_prev = QPushButton("Previous")
        self.btn_prev.setCursor(Qt.PointingHandCursor)
        self.btn_prev.setFixedSize(120, 40)
        self.btn_prev.setStyleSheet("""
            QPushButton {
                background-color: #393e46;
                color: #eeeeee;
                font-weight: 600;
                font-size: 16px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #4e5361;
            }
        """)

        self.btn_next = QPushButton("Next")
        self.btn_next.setCursor(Qt.PointingHandCursor)
        self.btn_next.setFixedSize(120, 40)
        self.btn_next.setStyleSheet("""
            QPushButton {
                background-color: #3a86ff;
                color: white;
                font-weight: 700;
                font-size: 16px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #559dff;
            }
        """)

        btn_layout.addWidget(self.btn_prev)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_next)

        layout.addWidget(name_label)
        layout.addWidget(self.name_edit)
        layout.addWidget(id_label)
        layout.addWidget(self.id_edit)
        layout.addWidget(brand_label)
        layout.addWidget(self.brand_edit)
        layout.addWidget(category_label)
        layout.addWidget(self.category_edit)
        layout.addStretch()
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.setStyleSheet("background-color: #121212;")


class PageThree(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet("background-color: #121212;")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)

        texa_label = QLabel("TEXA INNOVATES")
        texa_label.setStyleSheet("color: #3a86ff; font-weight: bold; font-size: 22px;")
        top_bar.addWidget(texa_label, alignment=Qt.AlignLeft | Qt.AlignVCenter)

        title_label = QLabel("CHROMA PLAST - Ice Cup Label Inspection")
        title_label.setStyleSheet("color: #eeeeee; font-weight: 700; font-size: 20px;")
        top_bar.addWidget(title_label, alignment=Qt.AlignCenter)

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_restart = QPushButton("Restart")
        self.btn_train = QPushButton("Train")

        for btn in (self.btn_start, self.btn_stop, self.btn_restart, self.btn_train):
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedSize(90, 35)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3a86ff;
                    color: white;
                    font-weight: bold;
                    border-radius: 6px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #559dff;
                }
            """)
        right_buttons = QVBoxLayout()
        right_buttons.setSpacing(15)
        right_buttons.addWidget(self.btn_start)
        right_buttons.addWidget(self.btn_stop)
        right_buttons.addWidget(self.btn_restart)
        right_buttons.addWidget(self.btn_train)
        right_buttons.addStretch()
        top_bar.addLayout(right_buttons)

        main_layout.addLayout(top_bar)

        cams_layout = QGridLayout()
        cams_layout.setSpacing(30)
        self.cams = []
        cam_style = """
            QFrame {
                background-color: #222831;
                border: 3px solid #3a86ff;
                border-radius: 15px;
            }
        """
        cam_size = QSize(250, 160)
        for i in range(4):
            frame = QFrame()
            frame.setFixedSize(cam_size)
            frame.setStyleSheet(cam_style)
            label = QLabel(f"Camera {i+1}")
            label.setStyleSheet("color: #cccccc; font-weight: 700; font-size: 18px;")
            label.setAlignment(Qt.AlignCenter)
            layout = QVBoxLayout(frame)
            layout.addStretch()
            layout.addWidget(label)
            layout.addStretch()
            self.cams.append(frame)
            row, col = divmod(i, 2)
            cams_layout.addWidget(frame, row, col)

        main_layout.addLayout(cams_layout)

        main_layout.addStretch()

        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(0, 0, 0, 0)
        bottom_bar.setSpacing(50)

        self.checkbox_auto = QCheckBox("Auto Mode")
        self.checkbox_auto.setChecked(True)
        self.checkbox_auto.setStyleSheet("""
            QCheckBox {
                color: #eeeeee;
                font-size: 20px;
                font-weight: 700;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 28px;
                height: 28px;
            }
            QCheckBox::indicator:checked {
                background-color: #3a86ff;
                border-radius: 7px;
                border: 2px solid #559dff;
            }
            QCheckBox::indicator:unchecked {
                background-color: #333;
                border-radius: 7px;
                border: 2px solid #555;
            }
        """)
        bottom_bar.addWidget(self.checkbox_auto, alignment=Qt.AlignLeft | Qt.AlignVCenter)

        self.btn_trigger = DiamondButton("Trigger")
        bottom_bar.addWidget(self.btn_trigger, alignment=Qt.AlignCenter)

        counters_layout = QVBoxLayout()
        counters_layout.setSpacing(20)
        counter_style = """
            QLabel {
                color: #3a86ff;
                font-weight: 700;
                font-size: 24px;
                border: 3px solid #3a86ff;
                border-radius: 15px;
                padding: 20px 30px;
                min-width: 130px;
                background-color: #121212;
                qproperty-alignment: AlignCenter;
            }
        """
        self.lbl_good = QLabel("Good\n0")
        self.lbl_good.setStyleSheet(counter_style)
        self.lbl_bad = QLabel("Bad\n0")
        self.lbl_bad.setStyleSheet(counter_style)
        self.lbl_total = QLabel("Total\n0")
        self.lbl_total.setStyleSheet(counter_style)

        counters_layout.addWidget(self.lbl_good)
        counters_layout.addWidget(self.lbl_bad)
        counters_layout.addWidget(self.lbl_total)
        counters_layout.addStretch()
        bottom_bar.addLayout(counters_layout)

        main_layout.addLayout(bottom_bar)

        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("Previous")
        self.btn_prev.setFixedSize(130, 45)
        self.btn_prev.setCursor(Qt.PointingHandCursor)
        self.btn_prev.setStyleSheet("""
            QPushButton {
                background-color: #393e46;
                color: #eeeeee;
                font-size: 17px;
                font-weight: 700;
                border-radius: 10px;
                border: 2px solid #3a86ff;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #4e5361;
            }
        """)
        nav_layout.addWidget(self.btn_prev, alignment=Qt.AlignLeft)
        nav_layout.addStretch()
        main_layout.addLayout(nav_layout)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ice Cup Label Inspection")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("background-color: #121212;")
        self._init_ui()

    def _init_ui(self):
        self.stack = QStackedWidget()
        self.page1 = PageOne()
        self.page2 = PageTwo()
        self.page3 = PageThree()

        self.stack.addWidget(self.page1)
        self.stack.addWidget(self.page2)
        self.stack.addWidget(self.page3)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stack)
        self.setLayout(main_layout)

        self.page1.btn_next.clicked.connect(self.goto_page2)
        self.page2.btn_prev.clicked.connect(self.goto_page1)
        self.page2.btn_next.clicked.connect(self.goto_page3)
        self.page3.btn_prev.clicked.connect(self.goto_page2)

        self.page3.btn_start.clicked.connect(lambda: print("Start pressed"))
        self.page3.btn_stop.clicked.connect(lambda: print("Stop pressed"))
        self.page3.btn_restart.clicked.connect(lambda: print("Restart pressed"))
        self.page3.btn_train.clicked.connect(lambda: print("Train pressed"))
        self.page3.btn_trigger.clicked.connect(lambda: print("Trigger pressed"))
        self.page3.checkbox_auto.stateChanged.connect(lambda state: print(f"Auto mode: {state}"))

    def goto_page1(self):
        self.stack.setCurrentWidget(self.page1)

    def goto_page2(self):
        self.stack.setCurrentWidget(self.page2)

    def goto_page3(self):
        if not self.page2.name_edit.text().strip():
            QMessageBox.warning(self, "Input Error", "Please enter Operator Name before continuing.")
            return
        self.stack.setCurrentWidget(self.page3)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
