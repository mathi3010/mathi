# texa_auto_inspection.py
# One-file PyQt5 app
# Pages: Welcome → Operator → Inspection (4 cams)
# Uses pre-saved polygon_cameraX.json + threshold_cameraX.json
# PLC flow: read 40001==1 -> capture 4 cams -> pixel-threshold decision -> write 1(good)/2(bad) to 40002 -> reset 0

import os, sys, time, json
from datetime import datetime

import numpy as np
import cv2

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QFormLayout, QLineEdit, QFrame, QMessageBox, QGridLayout
)

# ===== Modbus (pymodbus) =====
try:
    from pymodbus.client.sync import ModbusTcpClient
except Exception:
    ModbusTcpClient = None  # App still runs (no PLC)

# ---------------- CONFIG ----------------
CAMERA_INDEXES = [4, 1, 2, 3]  # change to your device indices
PLC_IP_DEFAULT = "192.168.3.1"
PLC_PORT_DEFAULT = 507

READ_REGISTER  = 1   # 40001 (request from PLC; 1 = inspect)
WRITE_REGISTER = 2   # 40002 (our result; 1=good, 2=bad; auto-reset to 0)

TOTAL_PROCESS_TIMEOUT = 3.0   # must complete within 3 seconds
PLC_POLL_INTERVAL_MS = 60     # poll PLC ~16 times/sec
PLC_RESET_DELAY = 0.3         # reset result to 0 after 300 ms

BLOCK_SIZE = 20
GRID_X = 50
GRID_Y = 50

POLYGON_TEMPLATE   = "polygon_camera{}.json"
THRESHOLD_TEMPLATE = "threshold_camera{}.json"

CAPTURE_DIR = "captures"
BGREM_DIR   = "bg_removed"
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(BGREM_DIR, exist_ok=True)

# -------------- Helpers ---------------
def cv_to_qpixmap_bgr(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

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

def load_polygon(cam_idx):
    fn = POLYGON_TEMPLATE.format(cam_idx)
    if not os.path.exists(fn): return None
    with open(fn, "r") as f:
        data = json.load(f)
    pts = data.get("polygon_points")
    if not isinstance(pts, list) or len(pts) < 3: return None
    return np.array(pts, dtype=np.int32)

def load_thresholds(cam_idx):
    fn = THRESHOLD_TEMPLATE.format(cam_idx)
    if not os.path.exists(fn): return None
    with open(fn, "r") as f:
        return json.load(f)

def compute_block_means(img_bgr, block=BLOCK_SIZE, gx=GRID_X, gy=GRID_Y):
    target_w = gx * block
    target_h = gy * block
    resized = cv2.resize(img_bgr, (target_w, target_h)).astype(np.float32)
    # Fast block mean (vectorized)
    try:
        resh = resized.reshape(gy, block, gx, block, 3)
        means = resh.mean(axis=(1, 3))
    except Exception:
        means = np.zeros((gy, gx, 3), dtype=np.float32)
        for y in range(gy):
            for x in range(gx):
                ys, xs = y*block, x*block
                means[y, x] = resized[ys:ys+block, xs:xs+block].mean(axis=(0,1))
    return means  # (gy, gx, 3) BGR

def check_against_thresholds(block_means, thresholds):
    """Return (is_all_good, bad_blocks_count)."""
    if thresholds is None:
        return False, 0
    gy, gx, _ = block_means.shape
    bad = 0
    for y in range(gy):
        for x in range(gx):
            bid = f"block_{y*gx + x + 1}"
            th = thresholds.get(bid)
            if not th:
                continue
            b, g, r = block_means[y, x]
            Br, Gr, Rr = th["B"], th["G"], th["R"]
            if not (Br[0] <= b <= Br[1] and Gr[0] <= g <= Gr[1] and Rr[0] <= r <= Rr[1]):
                bad += 1
    return (bad == 0), bad

# --------- Worker Thread for one cycle ---------
class InspectWorker(QThread):
    cam_done = pyqtSignal(int, bool, int)   # cam_idx, is_good, bad_count
    finished = pyqtSignal(bool, float)      # overall_good, elapsed

    def __init__(self, frames, polygons, thresholds, timeout=TOTAL_PROCESS_TIMEOUT):
        super().__init__()
        self.frames = frames
        self.polygons = polygons
        self.thresholds = thresholds
        self.timeout = timeout

    def run(self):
        t0 = time.time()
        results = []
        for i, f in enumerate(self.frames, start=1):
            if time.time() - t0 > self.timeout:
                self.finished.emit(False, time.time()-t0)
                return
            if f is None:
                results.append(False)
                self.cam_done.emit(i, False, -1)
                continue

            img = apply_clahe_bgr(f)
            poly = self.polygons[i-1]
            if poly is not None:
                img = remove_background_with_polygon(img, poly)

            th = self.thresholds[i-1]
            block_means = compute_block_means(img)
            good, badcnt = check_against_thresholds(block_means, th)

            results.append(good)
            self.cam_done.emit(i, good, badcnt)

        overall = all(results) if results else False
        self.finished.emit(overall, time.time()-t0)

# ---------------- UI Pages ----------------
class WelcomePage(QWidget):
    def __init__(self, go_next):
        super().__init__()
        self.setAutoFillBackground(True)
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(24, 26, 27))
        pal.setColor(QPalette.WindowText, Qt.white)
        self.setPalette(pal)

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(16)

        title = QLabel("TEXA INNOVATES")
        title.setStyleSheet("color:#ff8c00; font-size:42px; font-weight:900; letter-spacing:2px;")
        title.setAlignment(Qt.AlignCenter)
        sub1 = QLabel("CHROMA PLAST")
        sub1.setStyleSheet("color:#e6e6e6; font-size:28px; font-weight:700;")
        sub1.setAlignment(Qt.AlignCenter)
        sub2 = QLabel("Cup Label Inspection System")
        sub2.setStyleSheet("color:#c5c5c5; font-size:22px;")
        sub2.setAlignment(Qt.AlignCenter)

        root.addStretch(1)
        root.addWidget(title)
        root.addWidget(sub1)
        root.addWidget(sub2)
        root.addStretch(2)

        btn_next = QPushButton("Next  ➜")
        btn_next.setCursor(Qt.PointingHandCursor)
        btn_next.setFixedWidth(160)
        btn_next.setStyleSheet("""
            QPushButton {
                background:#ff8c00; color:#111; font-weight:800; padding:10px 16px;
                border:none; border-radius:10px;
            }
            QPushButton:hover { background:#ffa940; }
        """)
        btn_bar = QHBoxLayout()
        btn_bar.addStretch(1)
        btn_bar.addWidget(btn_next)
        root.addLayout(btn_bar)

        btn_next.clicked.connect(go_next)

class OperatorPage(QWidget):
    def __init__(self, go_prev, go_next):
        super().__init__()
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(24, 26, 27))
        pal.setColor(QPalette.WindowText, Qt.white)
        self.setPalette(pal)

        self.incharge_name = QLineEdit(); self.incharge_name.setPlaceholderText("Incharge Name")
        self.incharge_id   = QLineEdit(); self.incharge_id.setPlaceholderText("Incharge ID")
        self.employee_id   = QLineEdit(); self.employee_id.setPlaceholderText("Employee ID")
        self.brand         = QLineEdit(); self.brand.setPlaceholderText("Brand")
        self.category      = QLineEdit(); self.category.setPlaceholderText("Category")

        for w in (self.incharge_name, self.incharge_id, self.employee_id, self.brand, self.category):
            w.setStyleSheet("""
                QLineEdit {
                    background:#2f3235; color:#fff; border:1px solid #555;
                    border-radius:8px; padding:8px; min-width:340px;
                }
                QLineEdit:focus { border:1px solid #ff8c00; }
            """)

        form = QFormLayout()
        form.setSpacing(12)
        form.addRow("Incharge Name:", self.incharge_name)
        form.addRow("Incharge ID:",   self.incharge_id)
        form.addRow("Employee ID:",   self.employee_id)
        form.addRow("Brand:",         self.brand)
        form.addRow("Category:",      self.category)

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(16)

        title = QLabel("Operator Details")
        title.setStyleSheet("color:#ff8c00; font-size:24px; font-weight:800;")
        root.addWidget(title)

        form_line = QHBoxLayout()
        form_line.addStretch(1); form_line.addLayout(form); form_line.addStretch(1)
        root.addLayout(form_line)
        root.addStretch(1)

        btn_prev = QPushButton("⟵ Previous")
        btn_prev.setCursor(Qt.PointingHandCursor)
        btn_prev.setStyleSheet("background:#3a3d3f; color:#fff; padding:8px 14px; border-radius:8px;")
        btn_next = QPushButton("Next  ➜")
        btn_next.setCursor(Qt.PointingHandCursor)
        btn_next.setStyleSheet("background:#ff8c00; color:#111; padding:8px 14px; border-radius:8px; font-weight:800;")

        nav = QHBoxLayout()
        nav.addWidget(btn_prev); nav.addStretch(1); nav.addWidget(btn_next)
        root.addLayout(nav)

        btn_prev.clicked.connect(go_prev)
        btn_next.clicked.connect(go_next)

class InspectionPage(QWidget):
    def __init__(self, go_prev):
        super().__init__()
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(24, 26, 27))
        pal.setColor(QPalette.WindowText, Qt.white)
        pal.setColor(QPalette.Base, QColor(30, 32, 33))
        self.setPalette(pal)

        # State
        self.caps = []
        self.polygons = [None]*4
        self.thresholds = [None]*4
        self.processing = False
        self.worker = None
        self.plc = None
        self.last_plc_val = 0
        self.total = 0
        self.good_total = 0
        self.bad_total = 0

        # ---- Header: TEXA + PLC panel ----
        top = QHBoxLayout()
        brand = QLabel("TEXA")
        brand.setStyleSheet("color:#ff8c00; font-size:28px; font-weight:900; letter-spacing:2px;")
        top.addWidget(brand, alignment=Qt.AlignLeft)
        top.addStretch(1)

        self.plc_dot = QLabel("●")
        self._set_plc_dot(False)
        self.ip_inp = QLineEdit(PLC_IP_DEFAULT); self.ip_inp.setFixedWidth(160)
        self.port_inp = QLineEdit(str(PLC_PORT_DEFAULT)); self.port_inp.setFixedWidth(80)
        for w in (self.ip_inp, self.port_inp):
            w.setStyleSheet("background:#2f3235; color:#fff; border:1px solid #555; border-radius:6px; padding:6px;")

        self.btn_plc = QPushButton("Connect PLC")
        self.btn_plc.setCursor(Qt.PointingHandCursor)
        self.btn_plc.setStyleSheet("background:#3a3d3f; color:#fff; padding:8px 10px; border-radius:8px;")
        self.btn_plc.clicked.connect(self._reconnect_plc)

        self.btn_auto = QPushButton("Auto Running")
        self.btn_auto.setCheckable(True)
        self.btn_auto.setChecked(True)  # start in auto
        self.btn_auto.setCursor(Qt.PointingHandCursor)
        self.btn_auto.setStyleSheet("""
            QPushButton { background:#3a3d3f; color:#fff; padding:8px 10px; border-radius:8px; }
            QPushButton:checked { background:#ff8c00; color:#111; font-weight:800; }
        """)
        self.btn_auto.clicked.connect(self._toggle_auto)

        top.addWidget(QLabel("PLC:"))
        top.addWidget(self.plc_dot)
        top.addSpacing(8)
        top.addWidget(QLabel("IP:"));   top.addWidget(self.ip_inp)
        top.addWidget(QLabel("Port:")); top.addWidget(self.port_inp)
        top.addWidget(self.btn_plc)
        top.addSpacing(14)
        top.addWidget(self.btn_auto)

        # ---- Cameras grid ----
        grid = QGridLayout()
        grid.setSpacing(12)
        self.cam_frames, self.cam_labels = [], []
        for i in range(4):
            frame = QFrame()
            frame.setStyleSheet("background:#111; border:4px solid #333; border-radius:12px;")
            frame.setFixedSize(640, 480)
            lbl = QLabel(alignment=Qt.AlignCenter)
            lbl.setStyleSheet("background:#000; border:none;")
            lbl.setFixedSize(632, 472)
            lay = QVBoxLayout(frame); lay.setContentsMargins(4,4,4,4); lay.addWidget(lbl)
            grid.addWidget(frame, i//2, i%2)
            self.cam_frames.append(frame); self.cam_labels.append(lbl)

        # ---- Status bar ----
        status = QHBoxLayout()
        self.lbl_total = QLabel("TOTAL: 0")
        self.lbl_good  = QLabel("GOOD:  0")
        self.lbl_bad   = QLabel("BAD:   0")
        for s in (self.lbl_total, self.lbl_good, self.lbl_bad):
            s.setStyleSheet("background:#26292b; border:1px solid #444; border-radius:8px; padding:6px 10px; font-weight:700;")
        self.big_status = QLabel("STATUS: Waiting for PLC…")
        self.big_status.setStyleSheet("font-size:22px; font-weight:900; color:#e6e6e6;")
        status.addWidget(self.lbl_total); status.addSpacing(8)
        status.addWidget(self.lbl_good);  status.addSpacing(8)
        status.addWidget(self.lbl_bad)
        status.addStretch(1)
        status.addWidget(self.big_status)

        # ---- Nav ----
        nav = QHBoxLayout()
        btn_prev = QPushButton("⟵ Previous")
        btn_prev.setCursor(Qt.PointingHandCursor)
        btn_prev.setStyleSheet("background:#3a3d3f; color:#fff; padding:8px 14px; border-radius:8px;")
        btn_prev.clicked.connect(go_prev)
        btn_exit = QPushButton("Exit")
        btn_exit.setCursor(Qt.PointingHandCursor)
        btn_exit.setStyleSheet("background:#ff3b30; color:#fff; padding:8px 14px; border-radius:8px;")
        btn_exit.clicked.connect(QApplication.quit)
        nav.addWidget(btn_prev); nav.addStretch(1); nav.addWidget(btn_exit)

        # ---- Root layout ----
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.addLayout(top)
        root.addLayout(grid)
        root.addLayout(status)
        root.addLayout(nav)

        # ---- Timers ----
        self.live_timer = QTimer(self); self.live_timer.timeout.connect(self._update_live); self.live_timer.start(33)
        self.plc_timer = QTimer(self);  self.plc_timer.timeout.connect(self._poll_plc)

        # ---- Init ----
        self._open_cameras()
        self._load_configs()

        # Try PLC connect immediately (optional)
        if ModbusTcpClient is not None:
            self._reconnect_plc()
            self._toggle_auto(True)  # start auto polling

    # theming/util
    def _set_plc_dot(self, ok: bool):
        self.plc_dot.setStyleSheet(f"color: {'#00d26a' if ok else '#ff3b30'}; font-size:22px;")

    # cameras
    def _open_cameras(self):
        self.caps = []
        for idx in CAMERA_INDEXES:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.caps.append(cap)

    def _disconnect_cameras(self):
        for c in self.caps:
            try:
                if c and c.isOpened(): c.release()
            except: pass
        for f in self.cam_frames:
            f.setStyleSheet("background:#111; border:4px solid #333; border-radius:12px;")
        for l in self.cam_labels: l.clear()

    def _update_live(self):
        for i, (cap, lbl) in enumerate(zip(self.caps, self.cam_labels)):
            frame = None
            if cap and cap.isOpened():
                ok, frm = cap.read()
                if ok and frm is not None:
                    poly = self.polygons[i]
                    disp = frm.copy()
                    if poly is not None:
                        cv2.polylines(disp, [poly], True, (0, 200, 0), 2)
                        overlay = disp.copy()
                        cv2.fillPoly(overlay, [poly], (0, 200, 0))
                        cv2.addWeighted(overlay, 0.10, disp, 0.90, 0, disp)
                    frame = disp
            if frame is None:
                frame = np.full((480, 640, 3), 30, dtype=np.uint8)
            lbl.setPixmap(cv_to_qpixmap_bgr(frame))

    # PLC
    def _reconnect_plc(self):
        if ModbusTcpClient is None:
            QMessageBox.warning(self, "PLC", "pymodbus not installed. Install: pip install pymodbus==2.5.3")
            return
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
            ok = self.plc.connect()
            self._set_plc_dot(bool(ok))
            if not ok:
                QMessageBox.warning(self, "PLC", "Failed to connect to PLC.")
            else:
                self.big_status.setText("STATUS: Connected. Waiting for PLC…")
        except Exception as e:
            self._set_plc_dot(False)
            QMessageBox.warning(self, "PLC", f"PLC connect error: {e}")

    def _toggle_auto(self, checked):
        if checked:
            if not self.plc:
                self._reconnect_plc()
                if not self.plc:
                    self.btn_auto.setChecked(False)
                    return
            self.btn_auto.setText("Auto Running")
            self.plc_timer.start(PLC_POLL_INTERVAL_MS)
        else:
            self.btn_auto.setText("Start Auto")
            self.plc_timer.stop()

    def _poll_plc(self):
        if not self.plc or self.processing: return
        try:
            rr = self.plc.read_holding_registers(READ_REGISTER, 1, unit=1)
            if rr is None or rr.isError(): return
            val = rr.registers[0]
            if val == 1:
                self._start_cycle()
        except Exception:
            pass

    # load configs
    def _load_configs(self):
        for i in range(1, 5):
            self.polygons[i-1]   = load_polygon(i)
            self.thresholds[i-1] = load_thresholds(i)

        missing = [str(i) for i in range(1,5) if self.polygons[i-1] is None or self.thresholds[i-1] is None]
        if missing:
            QMessageBox.warning(self, "Config",
                                "Missing polygon/threshold JSON for camera(s): " + ", ".join(missing) +
                                "\nPlease ensure files like polygon_camera1.json & threshold_camera1.json exist.")

    # cycle
    def _start_cycle(self):
        self.processing = True
        self.big_status.setText("STATUS: Inspecting…")
        QCoreApplication.processEvents()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        frames = []
        for i, cap in enumerate(self.caps, start=1):
            f = None
            if cap and cap.isOpened():
                ok, frm = cap.read()
                if ok and frm is not None:
                    f = frm.copy()
                    cdir = os.path.join(CAPTURE_DIR, f"camera{i}")
                    os.makedirs(cdir, exist_ok=True)
                    cv2.imwrite(os.path.join(cdir, f"{ts}.bmp"), f)
            frames.append(f)

        self.worker = InspectWorker(frames, self.polygons, self.thresholds, timeout=TOTAL_PROCESS_TIMEOUT)
        self.worker.cam_done.connect(self._on_cam_done)
        self.worker.finished.connect(lambda good, el: self._on_cycle_done(good, el, ts))
        self.worker.start()

    def _on_cam_done(self, cam_idx, is_good, bad_cnt):
        color = "#00d26a" if is_good else "#ff3b30"
        self.cam_frames[cam_idx-1].setStyleSheet(f"background:#111; border:4px solid {color}; border-radius:12px;")

    def _on_cycle_done(self, overall_good, elapsed, ts):
        self.total += 1
        if overall_good: self.good_total += 1
        else: self.bad_total += 1

        self.lbl_total.setText(f"TOTAL: {self.total}")
        self.lbl_good.setText(f"GOOD:  {self.good_total}")
        self.lbl_bad.setText(f"BAD:   {self.bad_total}")
        self.big_status.setText(f"STATUS: {'GOOD' if overall_good else 'BAD'}  ({elapsed:.2f}s)")
        self.big_status.setStyleSheet(
            "font-size:22px; font-weight:900; color:#00d26a;" if overall_good
            else "font-size:22px; font-weight:900; color:#ff3b30;"
        )

        # Send to PLC: 1 good / 2 bad, then reset to 0
        try:
            if self.plc:
                val = 1 if overall_good else 2
                self.plc.write_register(WRITE_REGISTER, val, unit=1)
                QTimer.singleShot(int(PLC_RESET_DELAY*1000),
                    lambda: self.plc and self.plc.write_register(WRITE_REGISTER, 0, unit=1))
        except Exception:
            pass

        # Optional: save bg-removed preview
        for i, f in enumerate(self.worker.frames, start=1):
            if f is None: continue
            img = apply_clahe_bgr(f)
            poly = self.polygons[i-1]
            if poly is not None:
                img = remove_background_with_polygon(img, poly)
            outdir = os.path.join(BGREM_DIR, f"camera{i}")
            os.makedirs(outdir, exist_ok=True)
            cv2.imwrite(os.path.join(outdir, f"{ts}_bg.bmp"), img)

        self.processing = False
        self.big_status.setText("STATUS: Waiting for PLC…")

    # cleanup
    def closeEvent(self, e):
        try: self.live_timer.stop()
        except: pass
        try: self.plc_timer.stop()
        except: pass
        try:
            if self.plc: self.plc.close()
        except: pass
        self._disconnect_cameras()
        super().closeEvent(e)

# --------- Main Window with 3 pages ---------
class MainWindow(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TEXA • Cup Label Inspection System")
        self.resize(1520, 980)

        self.welcome  = WelcomePage(lambda: self.setCurrentIndex(1))
        self.operator = OperatorPage(lambda: self.setCurrentIndex(0), lambda: self.setCurrentIndex(2))
        self.inspect  = InspectionPage(lambda: self.setCurrentIndex(1))

        self.addWidget(self.welcome)   # 0
        self.addWidget(self.operator)  # 1
        self.addWidget(self.inspect)   # 2

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
