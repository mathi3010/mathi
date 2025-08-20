import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QStackedWidget, QMessageBox,
    QListWidget, QListWidgetItem, QLineEdit, QFormLayout, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon


# ==============================================================
# ----------------------- Image Processing ---------------------
# ==============================================================

def step1_extract_thread(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

    white_bg = np.ones_like(image) * 255
    result = np.where(mask[:, :, np.newaxis] == 255, image, white_bg)
    return result

def step2_remove_center_ring(image):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, ring_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_clean = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_clean, [largest], -1, 255, -1)

    img[mask_clean == 255] = [255, 255, 255]
    return img

def step3_remove_gray_circle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            per = cv2.arcLength(cnt, True)
            if per == 0:
                continue
            circ = 4 * np.pi * area / (per * per)
            if 0.7 < circ < 1.2:
                candidate_contours.append(cnt)

    if not candidate_contours:
        return image.copy(), None

    inner = max(candidate_contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [inner], -1, 255, thickness=cv2.FILLED)

    white_bg = np.full_like(image, 255)
    thread_part = cv2.bitwise_and(image, image, mask=mask)
    inv_mask = cv2.bitwise_not(mask)
    white_bg_with_hole = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
    removed = cv2.add(thread_part, white_bg_with_hole)

    x, y, w, h = cv2.boundingRect(inner)
    cropped_ring = removed[y:y+h, x:x+w]

    img_result = image.copy()
    img_result[mask == 255] = [255, 255, 255]

    return img_result, cropped_ring


# ==============================================================
# ----------------------- Analysis Functions -------------------
# ==============================================================

def analyze_blocks(image, block_size=50, region_name="Thread"):
    h, w, _ = image.shape
    rows = []
    block_id = 1
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            if block.size == 0:
                continue
            mask = np.any(block < 250, axis=-1)
            roi = block[mask]
            if roi.size == 0:
                mean=[255,255,255]; std=[0,0,0]; mn=[255,255,255]; mx=[255,255,255]; lap=0; px=0
            else:
                mean = roi.mean(axis=0); std = roi.std(axis=0)
                mn = roi.min(axis=0);   mx  = roi.max(axis=0)
                gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
                lap = cv2.Laplacian(gray_block, cv2.CV_64F).var()
                px = len(roi)
            rows.append({
                "Region": region_name,
                "Block_ID": block_id, "Block_X": x, "Block_Y": y,
                "Pixel_Count": px,
                "Mean_B": mean[0], "Mean_G": mean[1], "Mean_R": mean[2],
                "Std_B": std[0], "Std_G": std[1], "Std_R": std[2],
                "Min_B": mn[0], "Min_G": mn[1], "Min_R": mn[2],
                "Max_B": mx[0], "Max_G": mx[1], "Max_R": mx[2],
                "Laplacian_Var": lap
            })
            block_id += 1
    return pd.DataFrame(rows)

def analyze_whole_image(image, region_name="Ring"):
    if image is None or image.size == 0:
        return pd.DataFrame([{
            "Region": region_name, "Block_ID": 0, "Block_X": 0, "Block_Y": 0,
            "Pixel_Count": 0,
            "Mean_B": 255, "Mean_G": 255, "Mean_R": 255,
            "Std_B": 0, "Std_G": 0, "Std_R": 0,
            "Min_B": 255, "Min_G": 255, "Min_R": 255,
            "Max_B": 255, "Max_G": 255, "Max_R": 255,
            "Laplacian_Var": 0
        }])

    mask = np.any(image < 250, axis=-1)
    roi = image[mask]
    if roi.size == 0:
        mean=[255,255,255]; std=[0,0,0]; mn=[255,255,255]; mx=[255,255,255]; lap=0; px=0
    else:
        mean = roi.mean(axis=0); std = roi.std(axis=0)
        mn = roi.min(axis=0);   mx  = roi.max(axis=0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        px = len(roi)

    return pd.DataFrame([{
        "Region": region_name, "Block_ID": 0, "Block_X": 0, "Block_Y": 0,
        "Pixel_Count": px,
        "Mean_B": mean[0], "Mean_G": mean[1], "Mean_R": mean[2],
        "Std_B": std[0], "Std_G": std[1], "Std_R": std[2],
        "Min_B": mn[0], "Min_G": mn[1], "Min_R": mn[2],
        "Max_B": mx[0], "Max_G": mx[1], "Max_R": mx[2],
        "Laplacian_Var": lap
    }])

def write_analysis_excel(path, df_thread, df_ring):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_thread.to_excel(writer, index=False, sheet_name="Thread")
        df_ring.to_excel(writer, index=False, sheet_name="Ring")


# ==============================================================
# -------------------------- Comparison ------------------------
# ==============================================================

THRESHOLDS = {"mean_diff": 20, "lap_var_diff": 30, "pixel_diff": 50}
FAIL_THRESHOLD = 0.10  # <=10% bad blocks allowed

def compare_thread(df_ref, df_test):
    merged = df_ref.merge(df_test, on="Block_ID", suffixes=("_ref", "_test"))
    def judge(row):
        for ch in ["R","G","B"]:
            if abs(row[f"Mean_{ch}_ref"] - row[f"Mean_{ch}_test"]) > THRESHOLDS["mean_diff"]:
                return "Bad"
        if abs(row["Laplacian_Var_ref"] - row["Laplacian_Var_test"]) > THRESHOLDS["lap_var_diff"]:
            return "Bad"
        if abs(row["Pixel_Count_ref"] - row["Pixel_Count_test"]) > THRESHOLDS["pixel_diff"]:
            return "Bad"
        return "Good"
    merged["Quality"] = merged.apply(judge, axis=1)
    return merged

def compare_ring(df_ref, df_test):
    merged = df_ref.merge(df_test, on="Block_ID", suffixes=("_ref", "_test"))
    def judge(row):
        for ch in ["R","G","B"]:
            if abs(row[f"Mean_{ch}_ref"] - row[f"Mean_{ch}_test"]) > THRESHOLDS["mean_diff"]:
                return "Bad"
        if abs(row["Laplacian_Var_ref"] - row["Laplacian_Var_test"]) > THRESHOLDS["lap_var_diff"]:
            return "Bad"
        if abs(row["Pixel_Count_ref"] - row["Pixel_Count_test"]) > THRESHOLDS["pixel_diff"]:
                return "Bad"
        return "Good"
    merged["Quality"] = merged.apply(judge, axis=1)
    return merged

def overall_quality(df_thread_cmp, df_ring_cmp):
    def ok(df):
        if df is None or len(df) == 0:
            return True
        bad = (df["Quality"] == "Bad").sum()
        return (bad / len(df)) <= FAIL_THRESHOLD
    t_ok = ok(df_thread_cmp); r_ok = ok(df_ring_cmp)
    if t_ok and r_ok: return "Good"
    if not t_ok and r_ok: return "Bad (Thread mismatched)"
    if t_ok and not r_ok: return "Bad (Ring mismatched)"
    return "Bad (Thread + Ring mismatched)"


# ==============================================================
# ---------------------- Processing Wrapper --------------------
# ==============================================================

def process_image_to_excels(img_bgr, out_dir, base_name):
    """
    Returns:
      paths: (thread_path, ring_path_or_None, analysis_xlsx_path)
      dataframes: (df_thread, df_ring)
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    img1 = step1_extract_thread(img_bgr)
    img2 = step2_remove_center_ring(img1)
    thread_img, ring_img = step3_remove_gray_circle(img2)

    thread_path = out_dir / f"{base_name}_thread.bmp"
    cv2.imwrite(str(thread_path), thread_img)
    ring_path = None
    if ring_img is not None:
        ring_path = out_dir / f"{base_name}_ring.bmp"
        cv2.imwrite(str(ring_path), ring_img)

    df_thread = analyze_blocks(thread_img, block_size=50, region_name="Thread")
    df_ring   = analyze_whole_image(ring_img, region_name="Ring")
    xlsx_path = out_dir / f"{base_name}_analysis.xlsx"
    write_analysis_excel(str(xlsx_path), df_thread, df_ring)

    return (str(thread_path), (str(ring_path) if ring_path else None), str(xlsx_path)), (df_thread, df_ring)


# ==============================================================
# ----------------------------- UI -----------------------------
# ==============================================================

class TrainPage(QWidget):
    def __init__(self, add_reference_callback, go_back):
        super().__init__()
        self.add_reference_callback = add_reference_callback
        self.go_back = go_back

        self.reports_dir = Path.cwd() / "reports"
        self.refs_dir = self.reports_dir / "references"
        (self.reports_dir / "train").mkdir(parents=True, exist_ok=True)
        self.refs_dir.mkdir(parents=True, exist_ok=True)

        layout = QVBoxLayout()
        self.setStyleSheet("background-color: #f2f2f2;")

        title = QLabel("Train Mode - Capture GOOD and Name it (1, 2, 3, ...)")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #003366; margin: 10px;")
        layout.addWidget(title)

        # name input row
        form = QFormLayout()
        self.name_edit = QLineEdit(self.next_suggested_name())
        self.name_edit.setPlaceholderText("Enter GOOD name (e.g., 1, 2, 3)")
        form.addRow("GOOD Name:", self.name_edit)
        layout.addLayout(form)

        self.cam_label = QLabel()
        self.cam_label.setFixedSize(640, 480)
        self.cam_label.setStyleSheet("border: 3px solid #003366; border-radius: 10px;")
        layout.addWidget(self.cam_label, alignment=Qt.AlignCenter)

        self.status = QLabel("Status: —")
        self.status.setFont(QFont("Arial", 12, QFont.Bold))
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("color: #003366; margin: 6px;")
        layout.addWidget(self.status)

        row = QHBoxLayout()
        self.btn_capture = QPushButton("Capture & Save as GOOD")
        self.btn_capture.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_capture.setStyleSheet("background-color: #339966; color: white; padding: 8px; border-radius: 8px;")
        self.btn_capture.clicked.connect(self.capture_and_save_good)
        row.addWidget(self.btn_capture)

        btn_back = QPushButton("Back")
        btn_back.setFont(QFont("Arial", 12, QFont.Bold))
        btn_back.setStyleSheet("background-color: #cc3333; color: white; padding: 8px; border-radius: 8px;")
        btn_back.clicked.connect(self.close_and_back)
        row.addWidget(btn_back)

        layout.addLayout(row)
        self.setLayout(layout)

        # camera
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def next_suggested_name(self):
        # suggest next integer name based on existing xlsx files
        nums = []
        for x in (Path.cwd() / "reports" / "references").glob("*.xlsx"):
            try:
                nums.append(int(x.stem))
            except:
                pass
        return str(max(nums)+1) if nums else "1"

    def update_frame(self):
        ok, frame = self.cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w,ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
            self.cam_label.setPixmap(QPixmap.fromImage(qimg))

    def capture_and_save_good(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Name required", "Please enter a GOOD name (e.g., 1, 2, 3).")
            return
        ok, frame = self.cap.read()
        if not ok:
            QMessageBox.critical(self, "Camera Error", "Unable to capture frame.")
            return

        # Process and save as named GOOD reference
        (thread_path, ring_path, xlsx_path), (df_thread, df_ring) = process_image_to_excels(
            frame, self.reports_dir / "train", f"good_{name}"
        )
        # Save canonical reference bundle in references/
        ref_xlsx = self.refs_dir / f"{name}.xlsx"
        write_analysis_excel(str(ref_xlsx), df_thread, df_ring)

        # Save a small thumbnail to show in the list
        thumb_path = self.refs_dir / f"{name}.bmp"
        # Use the processed thread image as thumbnail if exists
        thread_img = cv2.imread(thread_path)
        if thread_img is None:
            thread_img = frame
        thumb = cv2.resize(thread_img, (160, 120), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(thumb_path), thumb)

        # notify app/state + UI
        self.add_reference_callback(str(ref_xlsx))
        self.status.setText(f"Status: SAVED GOOD ✅  name={name}\nExcel: {ref_xlsx}")
        # suggest next name
        self.name_edit.setText(self.next_suggested_name())

    def close_and_back(self):
        if self.cap.isOpened():
            self.cap.release()
        self.timer.stop()
        self.go_back()

    def closeEvent(self, e):
        if self.cap.isOpened():
            self.cap.release()
        e.accept()


class PredictionPage(QWidget):
    def __init__(self, list_references_callback, go_back):
        super().__init__()
        self.list_references_callback = list_references_callback
        self.go_back = go_back

        self.reports_dir = Path.cwd() / "reports"
        (self.reports_dir / "prediction").mkdir(parents=True, exist_ok=True)
        self.refs_dir = self.reports_dir / "references"

        # ---------- main layout: left list + right camera & result ----------
        main = QHBoxLayout()
        self.setStyleSheet("background-color: #f2f2f2;")

        # LEFT: references list
        left = QVBoxLayout()
        title_left = QLabel("GOOD References")
        title_left.setFont(QFont("Arial", 16, QFont.Bold))
        title_left.setAlignment(Qt.AlignCenter)
        title_left.setStyleSheet("color: #003366; margin: 6px;")
        left.addWidget(title_left)

        self.list_refs = QListWidget()
        self.list_refs.setIconSize(QSize(96, 72))
        self.list_refs.setMinimumWidth(260)
        self.list_refs.setStyleSheet("QListWidget { background: #ffffff; border: 2px solid #003366; }")
        left.addWidget(self.list_refs, stretch=1)

        self.refresh_btn = QPushButton("Refresh List")
        self.refresh_btn.setStyleSheet("background-color: #888888; color: white; padding: 6px; border-radius: 6px;")
        self.refresh_btn.clicked.connect(self.populate_refs)
        left.addWidget(self.refresh_btn)

        main.addLayout(left, 0)

        # RIGHT: camera + buttons + result
        right = QVBoxLayout()
        title = QLabel("Prediction - Capture & Compare with Selected GOOD")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #003366; margin: 10px;")
        right.addWidget(title)

        self.cam_label = QLabel()
        self.cam_label.setFixedSize(640, 480)
        self.cam_label.setStyleSheet("border: 3px solid #003366; border-radius: 10px;")
        right.addWidget(self.cam_label, alignment=Qt.AlignCenter)

        self.result = QLabel("Result: —")
        self.result.setFont(QFont("Arial", 13, QFont.Bold))
        self.result.setAlignment(Qt.AlignCenter)
        self.result.setStyleSheet("color: #003366; margin: 6px;")
        right.addWidget(self.result)

        row = QHBoxLayout()
        self.btn_capture = QPushButton("Capture & Predict")
        self.btn_capture.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_capture.setStyleSheet("background-color: #0066cc; color: white; padding: 8px; border-radius: 8px;")
        self.btn_capture.clicked.connect(self.capture_and_predict)
        row.addWidget(self.btn_capture)

        btn_back = QPushButton("Back")
        btn_back.setFont(QFont("Arial", 12, QFont.Bold))
        btn_back.setStyleSheet("background-color: #cc3333; color: white; padding: 8px; border-radius: 8px;")
        btn_back.clicked.connect(self.close_and_back)
        row.addWidget(btn_back)

        right.addLayout(row)
        main.addLayout(right, 1)

        self.setLayout(main)

        # camera
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.populate_refs()

    def populate_refs(self):
        """Load all references (1.xlsx, 2.xlsx, ...) with thumbnails into the left list."""
        self.list_refs.clear()
        for x in sorted(self.refs_dir.glob("*.xlsx"), key=lambda p: p.stem):
            name = x.stem
            thumb = self.refs_dir / f"{name}.bmp"
            icon = QIcon()
            if thumb.exists():
                pix = QPixmap(str(thumb))
                if not pix.isNull():
                    icon = QIcon(pix)
            item = QListWidgetItem(icon, f"{name}")
            item.setData(Qt.UserRole, str(x))  # store full path
            self.list_refs.addItem(item)

    def selected_reference_path(self):
        it = self.list_refs.currentItem()
        if not it:
            return None
        return it.data(Qt.UserRole)

    def update_frame(self):
        ok, frame = self.cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w,ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
            self.cam_label.setPixmap(QPixmap.fromImage(qimg))

    def capture_and_predict(self):
        ref_path = self.selected_reference_path()
        if not ref_path or not Path(ref_path).exists():
            QMessageBox.warning(self, "Select GOOD", "Please select a GOOD reference on the left.")
            return
        ok, frame = self.cap.read()
        if not ok:
            QMessageBox.critical(self, "Camera Error", "Unable to capture frame.")
            return

        # process current capture
        (thread_path, ring_path, test_xlsx), (df_tst_thread, df_tst_ring) = process_image_to_excels(
            frame, self.reports_dir / "prediction", "test"
        )

        # load reference
        df_ref_thread = pd.read_excel(ref_path, sheet_name="Thread")
        df_ref_ring   = pd.read_excel(ref_path, sheet_name="Ring")

        # compare
        df_cmp_thread = compare_thread(df_ref_thread, df_tst_thread)
        df_cmp_ring   = compare_ring(df_ref_ring, df_tst_ring)
        verdict = overall_quality(df_cmp_thread, df_cmp_ring)

        # save comparison
        cmp_xlsx = self.reports_dir / "prediction" / f"comparison_with_{Path(ref_path).stem}.xlsx"
        with pd.ExcelWriter(cmp_xlsx, engine="openpyxl") as writer:
            df_tst_thread.to_excel(writer, index=False, sheet_name="Thread_Test")
            df_tst_ring.to_excel(writer, index=False, sheet_name="Ring_Test")
            df_cmp_thread.to_excel(writer, index=False, sheet_name="Thread_Compare")
            df_cmp_ring.to_excel(writer, index=False, sheet_name="Ring_Compare")

        self.result.setText(f"Result: {verdict}\nSaved: {test_xlsx} | {cmp_xlsx}")

    def close_and_back(self):
        if self.cap.isOpened():
            self.cap.release()
        self.timer.stop()
        self.go_back()

    def closeEvent(self, e):
        if self.cap.isOpened():
            self.cap.release()
        e.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Texa Innovates - Cone Inspection System")
        self.setGeometry(200, 100, 1000, 720)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Home page
        home = QWidget()
        v = QVBoxLayout()
        home.setStyleSheet("background-color: #e6f2ff;")

        title = QLabel("Texa Innovates\nCone Inspection System")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #003366; margin: 40px;")
        v.addWidget(title)

        btn_train = QPushButton("Train Mode")
        btn_train.setFont(QFont("Arial", 14, QFont.Bold))
        btn_train.setStyleSheet("background-color: #0066cc; color: white; padding: 12px; border-radius: 10px;")
        btn_train.clicked.connect(self.open_train)
        v.addWidget(btn_train, alignment=Qt.AlignCenter)

        btn_pred = QPushButton("Prediction Mode")
        btn_pred.setFont(QFont("Arial", 14, QFont.Bold))
        btn_pred.setStyleSheet("background-color: #009933; color: white; padding: 12px; border-radius: 10px;")
        btn_pred.clicked.connect(self.open_pred)
        v.addWidget(btn_pred, alignment=Qt.AlignCenter)

        home.setLayout(v)
        self.stack.addWidget(home)

    # no global reference needed; we list them dynamically
    def open_train(self):
        self.train_page = TrainPage(self.on_reference_added, self.go_home)
        self.stack.addWidget(self.train_page)
        self.stack.setCurrentWidget(self.train_page)

    def open_pred(self):
        self.pred_page = PredictionPage(self.list_references, self.go_home)
        self.stack.addWidget(self.pred_page)
        self.stack.setCurrentWidget(self.pred_page)

    def on_reference_added(self, ref_xlsx_path: str):
        # If prediction page exists, refresh its list
        if hasattr(self, "pred_page"):
            self.pred_page.populate_refs()

    def list_references(self):
        return list((Path.cwd() / "reports" / "references").glob("*.xlsx"))

    def go_home(self):
        self.stack.setCurrentIndex(0)


# ==============================================================
# ------------------------------ MAIN --------------------------
# ==============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
