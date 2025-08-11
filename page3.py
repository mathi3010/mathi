import sys
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QPolygon, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QLineEdit, QComboBox, QGridLayout, QRadioButton, QButtonGroup,
    QSizePolicy, QMessageBox
)

class DiamondButton(QPushButton):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setMinimumSize(80, 80)
        self.setMaximumSize(100, 100)
        self.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor("#4CAF50"))
        painter.setPen(Qt.NoPen)

        w, h = self.width(), self.height()
        points = QPolygon([
           
            self.mapFromParent(QPoint(w // 2, 0)),
            self.mapFromParent(QPoint(w, h // 2)),
            self.mapFromParent(QPoint(w // 2, h)),
            self.mapFromParent(QPoint(0, h // 2))


        ])
        painter.drawPolygon(points)

        painter.setPen(Qt.white)
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, self.text())

class PageOne(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 60, 40, 60)
        layout.setSpacing(20)

        label1 = QLabel("TEXA INNOVATES")
        label1.setStyleSheet("color: #81C784; font-size: 28px; font-weight: 900;")
        label1.setAlignment(Qt.AlignCenter)

        label2 = QLabel("CHROMA PLAST")
        label2.setStyleSheet("color: #A5D6A7; font-size: 22px; font-weight: 700;")
        label2.setAlignment(Qt.AlignCenter)

        label3 = QLabel("Ice Cup Label Inspection")
        label3.setStyleSheet("color: #C8E6C9; font-size: 18px; font-weight: 600;")
        label3.setAlignment(Qt.AlignCenter)

        layout.addStretch(1)
        layout.addWidget(label1)
        layout.addWidget(label2)
        layout.addWidget(label3)
        layout.addStretch(3)

        self.next_btn = QPushButton("Next →")
        self.next_btn.setCursor(Qt.PointingHandCursor)
        self.next_btn.setStyleSheet("""
            QPushButton {
                background-color: #388E3C;
                color: white;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 16px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #2E7D32;
            }
        """)
        layout.addWidget(self.next_btn, alignment=Qt.AlignRight)

        self.setLayout(layout)

class PageTwo(QWidget):
    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)

        title = QLabel("Operator Information")
        title.setStyleSheet("color: #A5D6A7; font-size: 22px; font-weight: 700;")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        form_layout = QGridLayout()
        form_layout.setSpacing(15)

        name_label = QLabel("Operator Name:")
        name_label.setStyleSheet("color: #C8E6C9; font-weight: 600;")
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter your name")
        self.name_edit.setStyleSheet(self._input_style())

        id_label = QLabel("Operator ID:")
        id_label.setStyleSheet("color: #C8E6C9; font-weight: 600;")
        self.id_edit = QLineEdit()
        self.id_edit.setPlaceholderText("Enter your ID")
        self.id_edit.setStyleSheet(self._input_style())

        brand_label = QLabel("Brand for Label:")
        brand_label.setStyleSheet("color: #C8E6C9; font-weight: 600;")
        self.brand_combo = QComboBox()
        self.brand_combo.addItems(["Select Brand", "Brand A", "Brand B", "Brand C"])
        self.brand_combo.setStyleSheet(self._combo_style())

        cat_label = QLabel("Category:")
        cat_label.setStyleSheet("color: #C8E6C9; font-weight: 600;")
        self.cat_combo = QComboBox()
        self.cat_combo.addItems(["Select Category", "Category 1", "Category 2", "Category 3"])
        self.cat_combo.setStyleSheet(self._combo_style())

        form_layout.addWidget(name_label, 0, 0)
        form_layout.addWidget(self.name_edit, 0, 1)
        form_layout.addWidget(id_label, 1, 0)
        form_layout.addWidget(self.id_edit, 1, 1)
        form_layout.addWidget(brand_label, 2, 0)
        form_layout.addWidget(self.brand_combo, 2, 1)
        form_layout.addWidget(cat_label, 3, 0)
        form_layout.addWidget(self.cat_combo, 3, 1)

        main_layout.addLayout(form_layout)

        btn_layout = QHBoxLayout()
        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.setCursor(Qt.PointingHandCursor)
        self.prev_btn.setStyleSheet(self._nav_btn_style())

        self.next_btn = QPushButton("Next →")
        self.next_btn.setCursor(Qt.PointingHandCursor)
        self.next_btn.setStyleSheet(self._nav_btn_style())

        btn_layout.addWidget(self.prev_btn)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.next_btn)

        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

    def _input_style(self):
        return """
            QLineEdit {
                background-color: #2E7D32;
                border: 1.5px solid #4CAF50;
                border-radius: 5px;
                padding: 8px;
                color: #E8F5E9;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #81C784;
            }
        """

    def _combo_style(self):
        return """
            QComboBox {
                background-color: #2E7D32;
                border: 1.5px solid #4CAF50;
                border-radius: 5px;
                padding: 6px;
                color: #E8F5E9;
                font-size: 14px;
            }
            QComboBox QAbstractItemView {
                background-color: #1B5E20;
                color: #C8E6C9;
                selection-background-color: #388E3C;
            }
        """

    def _nav_btn_style(self):
        return """
            QPushButton {
                background-color: #388E3C;
                color: white;
                padding: 10px 24px;
                font-weight: 600;
                font-size: 15px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #2E7D32;
            }
        """

class PageThree(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 20, 30, 20)
        main_layout.setSpacing(15)

        title = QLabel("Quali Scan")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #A5D6A7; font-size: 24px; font-weight: 800;")
        main_layout.addWidget(title)

        cam_grid = QGridLayout()
        cam_grid.setSpacing(15)
        self.cam_labels = []

        for i in range(4):
            cam_label = QLabel(f"Camera {i+1}\n[Placeholder]")
            cam_label.setAlignment(Qt.AlignCenter)
            cam_label.setFixedSize(160, 120)
            cam_label.setStyleSheet("""
                background-color: #1B5E20;
                border: 2px dashed #4CAF50;
                color: #A5D6A7;
                font-size: 14px;
                font-weight: 600;
            """)
            self.cam_labels.append(cam_label)
            cam_grid.addWidget(cam_label, i // 2, i % 2)

        main_layout.addLayout(cam_grid)

        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet("color: #C8E6C9; font-weight: 700; font-size: 14px;")
        mode_layout.addWidget(mode_label)

        self.auto_radio = QRadioButton("Auto")
        self.manual_radio = QRadioButton("Manual")
        self.auto_radio.setChecked(True)

        for rb in (self.auto_radio, self.manual_radio):
            rb.setStyleSheet("""
                color: #C8E6C9;
                font-weight: 600;
                font-size: 14px;
            """)

        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.auto_radio)
        self.mode_group.addButton(self.manual_radio)

        mode_layout.addWidget(self.auto_radio)
        mode_layout.addWidget(self.manual_radio)
        mode_layout.addStretch(1)

        main_layout.addLayout(mode_layout)

        control_layout = QHBoxLayout()
        btn_style = """
            QPushButton {
                background-color: #388E3C;
                color: white;
                padding: 10px 22px;
                font-weight: 700;
                font-size: 15px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #2E7D32;
            }
        """

        self.start_btn = QPushButton("Start")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setStyleSheet(btn_style)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setStyleSheet(btn_style)

        self.restart_btn = QPushButton("Restart")
        self.restart_btn.setCursor(Qt.PointingHandCursor)
        self.restart_btn.setStyleSheet(btn_style)

        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.restart_btn)
        control_layout.addStretch(1)
        main_layout.addLayout(control_layout)

        counters_layout = QHBoxLayout()
        self.good_count_lbl = QLabel("Good: 0")
        self.bad_count_lbl = QLabel("Bad: 0")
        self.total_count_lbl = QLabel("Total: 0")

        for lbl in (self.good_count_lbl, self.bad_count_lbl, self.total_count_lbl):
            lbl.setStyleSheet("color: #C8E6C9; font-size: 16px; font-weight: 700; padding: 6px;")
            counters_layout.addWidget(lbl)

        counters_layout.addStretch(1)
        main_layout.addLayout(counters_layout)

        trigger_layout = QHBoxLayout()
        self.trigger_btn = DiamondButton("Trigger")
        trigger_layout.addStretch(1)
        trigger_layout.addWidget(self.trigger_btn)
        trigger_layout.addStretch(1)
        main_layout.addLayout(trigger_layout)

        train_predict_layout = QHBoxLayout()
        self.train_btn = QPushButton("Train")
        self.train_btn.setCursor(Qt.PointingHandCursor)
        self.train_btn.setStyleSheet(btn_style)
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.setCursor(Qt.PointingHandCursor)
        self.predict_btn.setStyleSheet(btn_style)

        train_predict_layout.addWidget(self.train_btn)
        train_predict_layout.addWidget(self.predict_btn)
        train_predict_layout.addStretch(1)
        main_layout.addLayout(train_predict_layout)

        self.setLayout(main_layout)

        self.start_btn.clicked.connect(self.start_scanning)
        self.stop_btn.clicked.connect(self.stop_scanning)
        self.restart_btn.clicked.connect(self.restart_scanning)
        self.train_btn.clicked.connect(self.train_model)
        self.predict_btn.clicked.connect(self.predict_labels)
        self.trigger_btn.clicked.connect(self.trigger_action)

        self.good_count = 0
        self.bad_count = 0
        self.total_count = 0

    def start_scanning(self):
        QMessageBox.information(self, "Start", "Start scanning initiated.\nAdd your camera logic here.")

    def stop_scanning(self):
        QMessageBox.information(self, "Stop", "Stop scanning.\nAdd your logic here.")

    def restart_scanning(self):
        QMessageBox.information(self, "Restart", "Restart scanning.\nAdd your logic here.")
        self.good_count = 0
        self.bad_count = 0
        self.total_count = 0
        self.update_counters()

    def train_model(self):
        QMessageBox.information(self, "Train", "Training model.\nAdd your training code here.")

    def predict_labels(self):
        QMessageBox.information(self, "Predict", "Prediction started.\nAdd your prediction code here.")

    def trigger_action(self):
        QMessageBox.information(self, "Trigger", "Diamond Trigger pressed!\nAdd your special trigger logic here.")

    def update_counters(self):
        self.good_count_lbl.setText(f"Good: {self.good_count}")
        self.bad_count_lbl.setText(f"Bad: {self.bad_count}")
        self.total_count_lbl.setText(f"Total: {self.total_count}")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ice Cup Label Inspection - TEXA INNOVATES")
        self.setMinimumSize(700, 550)
        self.setStyleSheet(self._dark_style())

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

        self.page1.next_btn.clicked.connect(self.go_to_page2)
        self.page2.prev_btn.clicked.connect(self.go_to_page1)
        self.page2.next_btn.clicked.connect(self.go_to_page3)

    def _dark_style(self):
        return """
            QWidget {
                background-color: #121212;
                color: #E0E0E0;
                font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            }
            QPushButton {
                font-weight: 600;
            }
            QLabel {
                font-weight: 500;
            }
        """

    def go_to_page1(self):
        self.stack.setCurrentWidget(self.page1)

    def go_to_page2(self):
        print("Going to Page 2")
        # TEMP: disable validation to test navigation
        # if not self.page2.name_edit.text().strip():
        #     QMessageBox.warning(self, "Validation", "Please enter the operator name.")
        #     return
        # if not self.page2.id_edit.text().strip():
        #     QMessageBox.warning(self, "Validation", "Please enter the operator ID.")
        #     return
        # if self.page2.brand_combo.currentIndex() == 0:
        #     QMessageBox.warning(self, "Validation", "Please select a brand.")
        #     return
        # if self.page2.cat_combo.currentIndex() == 0:
        #     QMessageBox.warning(self, "Validation", "Please select a category.")
        #     return

        self.stack.setCurrentWidget(self.page2)

    def go_to_page3(self):
        # Now enable validation here when going from Page 2 → 3
        if not self.page2.name_edit.text().strip():
            QMessageBox.warning(self, "Validation", "Please enter the operator name.")
            return
        if not self.page2.id_edit.text().strip():
            QMessageBox.warning(self, "Validation", "Please enter the operator ID.")
            return
        if self.page2.brand_combo.currentIndex() == 0:
            QMessageBox.warning(self, "Validation", "Please select a brand.")
            return
        if self.page2.cat_combo.currentIndex() == 0:
            QMessageBox.warning(self, "Validation", "Please select a category.")
            return

        self.stack.setCurrentWidget(self.page3)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
