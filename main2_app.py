# texa_inspector.py
# One-file PyQt5 app: 4-camera live view, polygon training, threshold-based inspection,
# PLC Modbus/TCP trigger + response (1=GOOD, 2=BAD). Aesthetic dark UI with orange TEXA label.

import os, sys, time, json
from datetime import datetime
import numpy as np
import cv2

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLineEdit, QMessageBox, QFrame
)

from pymodbus.client.sync import ModbusTcpClient

# -------------------- CONFIG --------------------
CAMERA_INDEXES = [4, 1, 2, 3]   # Change to your system (use only existing cameras)
PLC_IP_DEFAULT = "192.168.3.1"
PLC_PORT_DEFAULT = 507
READ_REGISTER  = 1   # 40001: trigger from PLC (1 = inspect)
WRITE_REGISTER = 2   # 40002: reply to PLC (1 = good, 2 = bad; auto-reset to 0)

# Processing/grid
BLOCK_SIZE = 20
GRID_X = 50
GRID_Y = 50

# Timing
PLC_POLL_INTERVAL_MS = 60
TOTAL_PROCESS_TIMEOUT = 3.0  # seconds to complete cycle
PLC_RESET_DELAY = 0.3        # seconds before writing back 0

# Files/paths
POLYGON_TEMPLATE   = "polygon_camera{}.json"
THRESHOLD_TEMPLATE = "threshold_camera{}.json"
CAPTURE_DIR = "captures"
BGREM_DIR   = "bg_removed"
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(BGREM_DIR, exist_ok=True)

# -------------------- UTILS --------------------
def cv_to_qpixmap_bgr(frame):
    """Convert BGR frame (OpenCV) to QPixmap."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def load_polygon(cam_idx):
    fn = POLYGON_TEMPLATE.format(cam_idx)
    if not os.path.exists(fn):
        return None
    with open(fn, "r") as f:
        data = json.load(f)
    pts = data.get("polygon_points")
    if not isinstance(pts, list) or len(pts) < 3:
        return None
    return np.array(pts, dtype=np.int32)

def save_polygon(cam_idx, points):
    with open(POLYGON_TEMPLATE.format(cam_idx), "w") as f:
        json.dump({"polygon_points": points}, f)

def load_thresholds(cam_idx):
    fn = THRESHOLD_TEMPLATE.format(cam_idx)
    if not os.path.exists(fn):
        return None
    with open(fn, "r") as f:
        return json.load(f)

def save_thresholds(cam_idx, block_info):
    with open(THRESHOLD_TEMPLATE.format(cam_idx), "w") as f:
        json.dump(block_info, f, indent=2)

def apply_clahe_bgr(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def remove_background_with_polygon(img_bgr, polygon_pts):
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_pts], 255)
    white_bg = np.full_like(img_bgr, 255)
    fg = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    bg = cv2.bitwise_and(white_bg, white_bg, mask=cv2.bitwise_not(mask))
    return cv2.add(fg, bg)

def compute_block_means(img_bgr, block=BLOCK_SIZE, gx=GRID_X, gy=GRID_Y):
    target_w = gx * block
    target_h = gy * block
    resized = cv2.resize(img_bgr, (target_w, target_h)).astype(np.float32)
    # vectorized block mean
    try:
        reshaped = resized.reshape(gy, block, gx, block, 3)
        means = reshaped.mean(axis=(1, 3))  # (gy, gx, 3) in BGR
    except Exception:
        means = np.zeros((gy, gx, 3), dtype=np.float32)
        for y in range(gy):
            for x in range(gx):
                ys, xs = y * block, x * block
                means[y, x] = resized[ys:ys+block, xs:xs+block].mean(axis=(0, 1))
    return means

def check_against_thresholds(block_means, thresholds):
    """Return True if ALL blocks are within thresholds; else False."""
    if thresholds is None:
        return False, 0
    gy, gx, _ = block_means.shape
    bad_count = 0
    for y in range(gy):
        for x in range(gx):
            bid = f"block_{y*gx + x + 1}"
            th = thresholds.get(bid)
            if not th:
                continue
            b, g, r = block_means[y, x]
            Br = th["B"]; Gr = th["G"]; Rr = th["R"]
            if not (Br[0] <= b <= Br[1] and Gr[0] <= g <= Gr[1] and Rr[0] <= r <= Rr[1]):
                bad_count += 1
    return (bad_count == 0), bad_count

# -------------------- WORKER --------------------
class InspectWorker(QThread):
    cam_result = pyqtSignal(int, bool, int)  # cam (1..4), is_good, bad_count
    finished   = pyqtSignal(bool, float)     # overall_good, elapsed

    def __init__(self, frames, polygons, thresholds, timeout=TOTAL_PROCESS_TIMEOUT, parent=None):
        super().__init__(parent)
        self.frames = frames
        self.polygons = polygons
        self.thresholds = thresholds
        self.timeout = timeout

    def run(self):
        start = time.time()
        per_good = []
        for i, frame in enumerate(self.frames, start=1):
            if time.time() - start > self.timeout:
                self.finished.emit(False, time.time()-start)
                return
            if frame is None:
                per_good.append(False)
                self.cam_result.emit(i, False, -1)
                continue

            # 1) CLAHE
            proc = apply_clahe_bgr(frame)
            # 2) Background removal
            poly = self.polygons[i-1]
            if poly is not None:
                proc = remove_background_with_polygon(proc, poly)
            # 3) Block means & threshold compare
            block_means = compute_block_means(proc)
            is_good, bad_cnt = check_against_thresholds(block_means, self.thresholds[i-1])
            per_good.append(is_good)
            self.cam_result.emit(i, is_good, bad_cnt)

        overall = all(per_good) if per_good else False
        self.finished.emit(overall, time.time() - start)

# -------------------- MAIN UI --------------------
class TexaApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TEXA – 4-Camera Inspector (Threshold)")
        self.resize(1500, 950)
        self._apply_dark_theme()

        self.caps = []
        self.polygons = [None]*4
        self.thresholds = [None]*4
        self.plc = None
        self.last_plc_val = 0
        self.total = 0
        self.good_total = 0
        self.bad_total = 0
        self.processing = False

        self._build_ui()
        self._open_cameras()
        self._load_all_configs()

        # Timers
        self.live_timer = QTimer(self)
        self.live_timer.timeout.connect(self._update_live_view)
        self.live_timer.start(33)

        self.plc_timer = QTimer(self)
        self.plc_timer.timeout.connect(self._poll_plc)

    # ---------- UI ----------
    def _apply_dark_theme(self):
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(24, 26, 27))
        pal.setColor(QPalette.WindowText, Qt.white)
        pal.setColor(QPalette.Base, QColor(30, 32, 33))
        pal.setColor(QPalette.AlternateBase, QColor(24, 26, 27))
        pal.setColor(QPalette.ToolTipBase, Qt.white)
        pal.setColor(QPalette.ToolTipText, Qt.white)
        pal.setColor(QPalette.Text, Qt.white)
        pal.setColor(QPalette.Button, QColor(30, 32, 33))
        pal.setColor(QPalette.ButtonText, Qt.white)
        pal.setColor(QPalette.Highlight, QColor(255, 140, 0))
        pal.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(pal)
        self.setStyleSheet("""
            QWidget { font-family: Segoe UI, Arial; font-size: 14px; }
            QPushButton {
                padding: 8px 14px;
                border: 1px solid #444;
                border-radius: 8px;
                background: #3a3d3f;
            }
            QPushButton:hover { background: #4a4f53; }
            QPushButton:checked { background: #ff8c00; color:#000; }
            QLineEdit {
                background: #2f3235; border: 1px solid #555; border-radius:6px; padding:6px;
            }
            QLabel#TEXA {
                color: #ff8c00; font-size: 28px; font-weight: 800; letter-spacing: 2px;
            }
            QLabel.bigStatus {
                font-size: 20px; font-weight: 700;
            }
            QFrame.cam {
                background: #1c1e20; border: 4px solid #333; border-radius: 12px;
            }
        """)

    def _camera_card(self, title):
        wrapper = QVBoxLayout()
        title_lbl = QLabel(title)
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setStyleSheet("color:#bbb; padding:4px;")
        frame = QFrame()
        frame.setObjectName("cam")
        frame.setFrameStyle(QFrame.NoFrame)
        frame.setFixedSize(640, 480)
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedSize(632, 472)
        lbl.setStyleSheet("background:#111; border: none;")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(4,4,4,4)
        frame_layout.addWidget(lbl)
        wrapper.addWidget(title_lbl)
        wrapper.addWidget(frame, alignment=Qt.AlignCenter)
        return wrapper, frame, lbl

    def _status_chip(self, label, value_text):
        box = QVBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet("color:#ccc;")
        val = QLabel(value_text)
        val.setObjectName("chip")
        val.setStyleSheet("background:#26292b; border:1px solid #444; border-radius:8px; padding:6px 10px; font-weight:700;")
        box.addWidget(lbl)
        box.addWidget(val)
        return box, val

    def _build_ui(self):
        # Header
        header = QHBoxLayout()
        texa = QLabel("TEXA"); texa.setObjectName("TEXA")
        header.addWidget(texa, alignment=Qt.AlignLeft)

        header.addStretch(1)

        plc_lbl = QLabel("PLC:")
        self.plc_dot = QLabel("●")
        self._set_plc_dot(False)
        plc_lbl.setStyleSheet("color:#bbb; margin-right:4px;")
        self.plc_dot.setStyleSheet("font-size:22px;")
        ip_edit = QLabel("IP")
        self.ip_inp = QLineEdit(PLC_IP_DEFAULT); self.ip_inp.setFixedWidth(140)
        port_edit = QLabel("Port")
        self.port_inp = QLineEdit(str(PLC_PORT_DEFAULT)); self.port_inp.setFixedWidth(80)
        self.btn_plc = QPushButton("Reconnect PLC")
        self.btn_plc.clicked.connect(self._reconnect_plc)

        header.addWidget(plc_lbl)
        header.addWidget(self.plc_dot)
        header.addSpacing(12)
        header.addWidget(ip_edit)
        header.addWidget(self.ip_inp)
        header.addWidget(port_edit)
        header.addWidget(self.port_inp)
        header.addWidget(self.btn_plc)

        # Controls
        controls = QHBoxLayout()
        self.btn_auto = QPushButton("Start Auto")
        self.btn_auto.setCheckable(True)
        self.btn_auto.clicked.connect(self._toggle_auto)

        self.btn_train = QPushButton("Redraw Polygons & Train")
        self.btn_train.clicked.connect(self._train_all)

        self.btn_cam = QPushButton("Disconnect Cameras")
        self.btn_cam.clicked.connect(self._disconnect_cameras)

        controls.addWidget(self.btn_auto)
        controls.addWidget(self.btn_train)
        controls.addWidget(self.btn_cam)
        controls.addStretch(1)

        # Camera grid 2x2
        grid = QGridLayout()
        self.cam_frames = []
        self.cam_labels = []
        for i in range(4):
            card, frame, lbl = self._camera_card(f"Camera {i+1}")
            self.cam_frames.append(frame)
            self.cam_labels.append(lbl)
            cont = QWidget()
            cont.setLayout(card)
            grid.addWidget(cont, i//2, i%2)

        # Status strip
        status = QHBoxLayout()
        box_total, self.lbl_total = self._status_chip("TOTAL", "0")
        box_good,  self.lbl_good  = self._status_chip("GOOD", "0")
        box_bad,   self.lbl_bad   = self._status_chip("BAD",  "0")
        status.addLayout(box_total); status.addSpacing(20)
        status.addLayout(box_good);  status.addSpacing(20)
        status.addLayout(box_bad)
        status.addStretch(1)
        self.big_status = QLabel("STATUS: —")
        self.big_status.setObjectName("bigStatus")
        status.addWidget(self.big_status)

        # Main layout
        root = QVBoxLayout()
        root.addLayout(header)
        root.addSpacing(6)
        root.addLayout(controls)
        root.addSpacing(8)
        root.addLayout(grid)
        root.addSpacing(8)
        root.addLayout(status)
        self.setLayout(root)

    def _set_plc_dot(self, connected: bool):
        color = "#00d26a" if connected else "#ff3b30"
        self.plc_dot.setStyleSheet(f"color:{color}; font-size:22px;")

    # ---------- Cameras ----------
    def _open_cameras(self):
        self._log("Opening cameras…")
        self.caps = []
        for idx in CAMERA_INDEXES:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                self._log(f"⚠ Camera index {idx} not available.")
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.caps.append(cap)
        self._log("Camera init done.")

    def _disconnect_cameras(self):
        for cap in self.caps:
            try:
                if cap and cap.isOpened():
                    cap.release()
            except: pass
        for fr in self.cam_frames:
            fr.setStyleSheet("background:#1c1e20; border: 4px solid #333; border-radius: 12px;")
        for lbl in self.cam_labels:
            lbl.clear()
        self._log("Cameras disconnected.")

    def _update_live_view(self):
        for i, (cap, lbl) in enumerate(zip(self.caps, self.cam_labels)):
            frame = None
            if cap and cap.isOpened():
                ok, frm = cap.read()
                if ok and frm is not None:
                    # preview polygon overlay if available
                    poly = self.polygons[i]
                    display = frm.copy()
                    if poly is not None:
                        cv2.polylines(display, [poly], True, (0, 200, 0), 2)
                        overlay = display.copy()
                        cv2.fillPoly(overlay, [poly], (0, 200, 0))
                        cv2.addWeighted(overlay, 0.12, display, 0.88, 0, display)
                    frame = display
            if frame is None:
                frame = np.full((480, 640, 3), 40, dtype=np.uint8)
            lbl.setPixmap(cv_to_qpixmap_bgr(frame))

    # ---------- PLC ----------
    def _reconnect_plc(self):
        ip = self.ip_inp.text().strip()
        try:
            port = int(self.port_inp.text().strip())
        except:
            QMessageBox.warning(self, "PLC", "Invalid port.")
            return
        try:
            if self.plc:
                try: self.plc.close()
                except: pass
            self.plc = ModbusTcpClient(ip, port=port)
            if self.plc.connect():
                self._log(f"PLC connected {ip}:{port}")
                self._set_plc_dot(True)
            else:
                self._log("PLC connect failed.")
                self.plc = None
                self._set_plc_dot(False)
        except Exception as e:
            self._log(f"PLC error: {e}")
            self.plc = None
            self._set_plc_dot(False)

    def _toggle_auto(self, checked):
        if checked:
            if not self.plc:
                self._reconnect_plc()
                if not self.plc:
                    QMessageBox.warning(self, "PLC", "PLC not connected.")
                    self.btn_auto.setChecked(False)
                    return
            self.btn_auto.setText("Stop Auto")
            self.plc_timer.start(PLC_POLL_INTERVAL_MS)
            self._log("Auto mode ON.")
        else:
            self.btn_auto.setText("Start Auto")
            self.plc_timer.stop()
            self._log("Auto mode OFF.")

    def _poll_plc(self):
        if not self.plc or self.processing:
            return
        try:
            rr = self.plc.read_holding_registers(READ_REGISTER, 1, unit=1)
            if rr is None or rr.isError():
                return
            val = rr.registers[0]
            # rising edge & not already processing
            if val == 1 and self.last_plc_val != 1:
                self._log("PLC trigger=1 → start inspection.")
                self._start_inspection_cycle()
            self.last_plc_val = val
        except Exception as e:
            self._log(f"PLC poll error: {e}")

    # ---------- Training ----------
    def _train_all(self):
        if not self.caps:
            QMessageBox.warning(self, "Train", "No cameras.")
            return
        QMessageBox.information(
            self, "Training",
            "For each camera: click polygon points on the popped-up image, press Enter when done."
        )
        for i, cap in enumerate(self.caps, start=1):
            if not (cap and cap.isOpened()):
                self._log(f"Skip Cam{i}: not opened.")
                continue
            ok, frm = cap.read()
            if not ok or frm is None:
                self._log(f"Skip Cam{i}: no frame.")
                continue

            img = apply_clahe_bgr(frm.copy())
            draw = img.copy()
            points = []

            def on_mouse(ev, x, y, flags, param):
                if ev == cv2.EVENT_LBUTTONDOWN:
                    points.append((x, y))
                    cv2.circle(draw, (x, y), 3, (0, 0, 255), -1)

            win = f"Draw polygon - Camera {i}"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 960, 540)
            cv2.setMouseCallback(win, on_mouse)
            while True:
                cv2.imshow(win, draw)
                k = cv2.waitKey(1) & 0xFF
                if k in (13, 10):  # Enter
                    break
                if k == 27:       # Esc -> cancel
                    points = []
                    break
            cv2.destroyWindow(win)

            if len(points) < 3:
                self._log(f"Cam{i}: skipped (need >= 3 points).")
                continue

            # Save polygon
            save_polygon(i, points)
            self.polygons[i-1] = np.array(points, dtype=np.int32)
            self._log(f"Saved polygon_camera{i}.json")

            # Build thresholds
            masked = remove_background_with_polygon(img, self.polygons[i-1])
            grid_img = cv2.resize(masked, (GRID_X*BLOCK_SIZE, GRID_Y*BLOCK_SIZE))
            info = {}
            for y in range(GRID_Y):
                for x in range(GRID_X):
                    xs, ys = x*BLOCK_SIZE, y*BLOCK_SIZE
                    block = grid_img[ys:ys+BLOCK_SIZE, xs:xs+BLOCK_SIZE]
                    b, g, r = block.mean(axis=(0,1))
                    bid = f"block_{y*GRID_X + x + 1}"
                    info[bid] = {
                        "R": [float(r*0.40), float(r*1.60)],
                        "G": [float(g*0.40), float(g*1.60)],
                        "B": [float(b*0.40), float(b*1.60)],
                    }
            save_thresholds(i, info)
            self.thresholds[i-1] = info
            self._log(f"Saved threshold_camera{i}.json")
        QMessageBox.information(self, "Training", "Done. Polygons & thresholds updated.")

    def _load_all_configs(self):
        for i in range(1, 5):
            self.polygons[i-1] = load_polygon(i)
            self.thresholds[i-1] = load_thresholds(i)
            if self.polygons[i-1] is not None:
                self._log(f"Loaded polygon_camera{i}.json")
            else:
                self._log(f"polygon_camera{i}.json not found.")
            if self.thresholds[i-1] is not None:
                self._log(f"Loaded threshold_camera{i}.json")
            else:
                self._log(f"threshold_camera{i}.json not found.")

    # ---------- Inspection ----------
    def _start_inspection_cycle(self):
        self.processing = True
        QCoreApplication.processEvents()

        # capture frames (and store originals)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frames = []
        for i, cap in enumerate(self.caps, start=1):
            f = None
            if cap and cap.isOpened():
                ok, frm = cap.read()
                if ok and frm is not None:
                    f = frm.copy()
                    cam_dir = os.path.join(CAPTURE_DIR, f"camera{i}")
                    os.makedirs(cam_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(cam_dir, f"{timestamp}.bmp"), f)
            frames.append(f)

        self.worker = InspectWorker(frames, self.polygons, self.thresholds, timeout=TOTAL_PROCESS_TIMEOUT)
        self.worker.cam_result.connect(self._on_cam_result)
        self.worker.finished.connect(lambda good, el: self._on_finished(good, el, timestamp))
        self.worker.start()

    def _on_cam_result(self, cam_idx, is_good, bad_cnt):
        # Set per-camera border (GOOD green / BAD red)
        frame = self.cam_frames[cam_idx-1]
        color = "#00d26a" if is_good else "#ff3b30"
        frame.setStyleSheet(f"background:#1c1e20; border: 4px solid {color}; border-radius: 12px;")

    def _on_finished(self, overall_good, elapsed, ts):
        self.total += 1
        if overall_good:
            self.good_total += 1
        else:
            self.bad_total += 1

        # Update UI counters
        self.lbl_total.setText(str(self.total))
        self.lbl_good.setText(str(self.good_total))
        self.lbl_bad.setText(str(self.bad_total))
        self.big_status.setText(f"STATUS: {'GOOD' if overall_good else 'BAD'}  ({elapsed:.2f}s)")
        self.big_status.setStyleSheet(
            "font-size: 20px; font-weight:700; color:#00d26a;" if overall_good
            else "font-size: 20px; font-weight:700; color:#ff3b30;"
        )

        # Write result to PLC (1 good / 2 bad), then reset to 0
        try:
            if self.plc:
                val = 1 if overall_good else 2
                self.plc.write_register(WRITE_REGISTER, val, unit=1)
                self._log(f"PLC write: {val} -> 40002")
                QTimer.singleShot(int(PLC_RESET_DELAY*1000),
                                  lambda: self.plc.write_register(WRITE_REGISTER, 0, unit=1))
            else:
                self._log("PLC not connected: skipping write.")
        except Exception as e:
            self._log(f"PLC write error: {e}")

        # Save bg-removed snapshots (optional / quick)
        for i, cap in enumerate(self.caps, start=1):
            try:
                if cap and cap.isOpened():
                    ok, frm = cap.read()
                    if ok and frm is not None:
                        poly = self.polygons[i-1]
                        img = apply_clahe_bgr(frm)
                        if poly is not None:
                            img = remove_background_with_polygon(img, poly)
                        out_dir = os.path.join(BGREM_DIR, f"camera{i}")
                        os.makedirs(out_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(out_dir, f"{ts}_bg.bmp"), img)
            except: pass

        self.processing = False

    # ---------- Misc ----------
    def _log(self, msg):
        # print only; UI kept clean on purpose
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def closeEvent(self, e):
        try:
            self.live_timer.stop()
            self.plc_timer.stop()
        except: pass
        try:
            if self.plc: self.plc.close()
        except: pass
        try:
            self._disconnect_cameras()
        except: pass
        super().closeEvent(e)

# -------------------- ENTRY --------------------
def main():
    app = QApplication(sys.argv)
    win = TexaApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
