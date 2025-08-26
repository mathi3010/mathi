# app_page_headers_texa.py
# PyQt5 shell: page-level headers (centered "Intelli Cone" + TEXA logo on right)
# Sidebar styled to match your reference image.

# ===== imports =====
import sys, os, time, json
from pathlib import Path
import cv2
import pandas as pd
from pathlib import Path

from PyQt5.QtWidgets import QDialog, QMessageBox, QFormLayout, QLineEdit, QPushButton, QHBoxLayout
from datetime import datetime
import os, csv

from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QPixmap, QFont, QIcon, QImage, QPainter, QColor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QStackedWidget,
    QFrame, QDialog, QListWidget, QListWidgetItem,
    QDialogButtonBox, QFormLayout, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QSizePolicy
)
#=====================================================================================================
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(r"D:/cone_design/reports/data/app.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def db_connect():
    con = sqlite3.connect(str(DB_PATH))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con

def db_init():
    con = db_connect()
    cur = con.cursor()

    # Raw log of each prediction event
    cur.execute("""
    CREATE TABLE IF NOT EXISTS prediction_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,          -- ISO timestamp
        date TEXT NOT NULL,        -- YYYY-MM-DD
        time TEXT NOT NULL,        -- HH:MM:SS
        jobtype TEXT NOT NULL,     -- your 'Job' from JobDialog
        job_count INTEGER NOT NULL,-- 1,2,3... per (date, jobtype)
        good_or_bad TEXT NOT NULL, -- 'Good' or 'Bad'
        ref_name TEXT,             -- selected reference name
        operator_id TEXT,
        notes TEXT,
        thread_bad_pct REAL,
        ring_bad_pct REAL
    );
    """)

    # Daily rollup (kept in sync by upsert in code)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dashboard_daily (
        date TEXT NOT NULL,
        jobtype TEXT NOT NULL,
        total INTEGER NOT NULL DEFAULT 0,
        good INTEGER NOT NULL DEFAULT 0,
        bad INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (date, jobtype)
    );
    """)

    con.commit()
    con.close()

def _next_job_count(con, date_str: str, jobtype: str) -> int:
    cur = con.cursor()
    cur.execute("""
        SELECT COALESCE(MAX(job_count), 0) FROM prediction_log
        WHERE date = ? AND jobtype = ?;
    """, (date_str, jobtype))
    last = cur.fetchone()[0] or 0
    return last + 1

def db_insert_prediction(jobtype: str, is_good: bool, ref_name: str,
                         operator_id: str, notes: str,
                         thread_bad_pct: float, ring_bad_pct: float):
    """Insert one prediction row + update dashboard_daily."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    good_or_bad = "Good" if is_good else "Bad"

    con = db_connect()
    try:
        jc = _next_job_count(con, date_str, jobtype)

        con.execute("""
            INSERT INTO prediction_log
            (ts, date, time, jobtype, job_count, good_or_bad,
             ref_name, operator_id, notes, thread_bad_pct, ring_bad_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            now.isoformat(timespec="seconds"), date_str, time_str,
            jobtype, jc, good_or_bad,
            ref_name or "", operator_id or "", notes or "",
            float(thread_bad_pct or 0.0), float(ring_bad_pct or 0.0)
        ))

        # Upsert into dashboard_daily
        # Try update; if no row, insert new
        cur = con.cursor()
        cur.execute("""
            UPDATE dashboard_daily
               SET total = total + 1,
                   good  = good  + CASE WHEN ?='Good' THEN 1 ELSE 0 END,
                   bad   = bad   + CASE WHEN ?='Bad'  THEN 1 ELSE 0 END
             WHERE date = ? AND jobtype = ?;
        """, (good_or_bad, good_or_bad, date_str, jobtype))
        if cur.rowcount == 0:
            cur.execute("""
                INSERT INTO dashboard_daily (date, jobtype, total, good, bad)
                VALUES (?, ?, 1, CASE WHEN ?='Good' THEN 1 ELSE 0 END,
                                CASE WHEN ?='Bad'  THEN 1 ELSE 0 END);
            """, (date_str, jobtype, good_or_bad, good_or_bad))

        con.commit()
    finally:
        con.close()

from datetime import datetime
import sqlite3
from pathlib import Path

# Use your existing DB_PATH and db_connect()
# prediction_log schema used here:
# (ts, date, time, jobtype, good_or_bad, ref_name, thread_bad_pct, ring_bad_pct)

def db_fetch_recent(limit=200):
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        SELECT date,time,jobtype,good_or_bad,ref_name,thread_bad_pct,ring_bad_pct
          FROM prediction_log
         ORDER BY id DESC
         LIMIT ?;""", (limit,))
    rows = cur.fetchall()
    con.close()
    return rows

def db_fetch_summary(date_from=None, date_to=None, job_like=None):
    """
    Returns grouped rows: Date, Job, Total, Good, Bad
    """
    con = db_connect()
    cur = con.cursor()
    q = """
        SELECT date,
               jobtype,
               COUNT(*) AS total,
               SUM(CASE WHEN good_or_bad='Good' THEN 1 ELSE 0 END) AS good_cnt,
               SUM(CASE WHEN good_or_bad='Bad'  THEN 1 ELSE 0 END) AS bad_cnt
          FROM prediction_log
         WHERE 1=1
    """
    args = []
    if date_from:
        q += " AND date >= ?"
        args.append(date_from)
    if date_to:
        q += " AND date <= ?"
        args.append(date_to)
    if job_like:
        q += " AND jobtype LIKE ?"
        args.append(f"%{job_like}%")
    q += " GROUP BY date, jobtype ORDER BY date DESC, jobtype ASC;"
    cur.execute(q, args)
    rows = cur.fetchall()
    con.close()
    return rows

def db_fetch_kpis(date_from=None, date_to=None, job_like=None):
    """
    Returns dict: total, good, bad, good_pct for filters.
    """
    con = db_connect()
    cur = con.cursor()
    q = "SELECT COUNT(*), SUM(CASE WHEN good_or_bad='Good' THEN 1 ELSE 0 END) FROM prediction_log WHERE 1=1"
    args = []
    if date_from:
        q += " AND date >= ?"; args.append(date_from)
    if date_to:
        q += " AND date <= ?"; args.append(date_to)
    if job_like:
        q += " AND jobtype LIKE ?"; args.append(f"%{job_like}%")
    cur.execute(q, args)
    total, good = cur.fetchone()
    con.close()
    total = total or 0
    good = good or 0
    bad = max(total - good, 0)
    good_pct = (good * 100.0 / total) if total else 0.0
    return {"total": total, "good": good, "bad": bad, "good_pct": good_pct}


# ===== style / paths =====
REDWINE   = "#7B1E3A"
BORDER    = "#94a3b8"
LOGO_PATH = "D:/cone_design/assets/texa.png"   # adjust if needed

def ref_dirs():
    """Return (root_dir, refs_dir)."""
    root = Path("D:/cone_design/reports")
    refs = root / "references"
    refs.mkdir(parents=True, exist_ok=True)
    return root, refs
def app_root() -> Path:
    # change this if you want a different base
    return Path("D:/cone_design/reports")

DIRS = {
    "root":        lambda: app_root(),
    "refs":        lambda: app_root() / "references",          # GOOD reference .xlsx with Thread/Ring sheets
    "pred":        lambda: app_root() / "prediction",          # results_log.jsonl
    "captures":    lambda: app_root() / "training" / "captures",   # saved camera frames
    "train_excel": lambda: app_root() / "training" / "excel",      # index/listing of captures, etc.
    "compare":     lambda: app_root() / "comparison",          # exported comparison xlsx
    "tmp":         lambda: app_root() / "tmp",
}

def ensure_dirs():
    for k, getp in DIRS.items():
        p = getp()
        p.mkdir(parents=True, exist_ok=True)
ensure_dirs()

def apply_shadow(widget, blur=20, dx=0, dy=2, rgba=(0,0,0,80)):
    """Optional: visual shadow effect (Qt alternative to CSS box-shadow)."""
    eff = QGraphicsDropShadowEffect()
    eff.setBlurRadius(blur)
    eff.setOffset(dx, dy)
    eff.setColor(QColor(*rgba))
    widget.setGraphicsEffect(eff)

# Safe camera wrapper
class CameraGuard:
    def __init__(self, preferred_index=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(preferred_index, cv2.CAP_DSHOW)
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            self.cap = None

    def read(self):
        if not self.cap:
            return False, None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def release(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
# ---------------- Reusable Page Header ----------------
class PageHeader(QWidget):
    """
    Slim header used at the top of each page:
    - Centered "Intelli Cone" title
    - TEXA logo on the right (sized to match your reference)
    """
    LOGO_PATH = "assets/texa_logo.png"
    LOGO_WIDTH = 140  # ~ matches screenshot size; change if needed
    HEIGHT = 72

    def __init__(self, title_text="Intelli Cone", parent=None):
        super().__init__(parent)
        self.setObjectName("PageHeader")
        self.setFixedHeight(self.HEIGHT)

        self.title = QLabel(title_text)
        self.title.setObjectName("HeaderTitle")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.logo = QLabel()
        self.logo.setObjectName("HeaderLogo")
        self.logo.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._load_logo()

        # 3-column grid ensures the title is mathematically centered
        grid = QGridLayout(self)
        grid.setContentsMargins(16, 12, 16, 12)
        grid.setHorizontalSpacing(8)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 0)
        grid.setColumnStretch(2, 1)

        # left dummy expands so title stays centered
        left_dummy = QWidget()
        left_dummy.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        grid.addWidget(left_dummy, 0, 0)
        grid.addWidget(self.title, 0, 1, alignment=Qt.AlignCenter)
        grid.addWidget(self.logo, 0, 2, alignment=Qt.AlignRight)

    def _load_logo(self):
        pix = QPixmap(self.LOGO_PATH)
        if pix.isNull():
            # Fallback text if logo not found
            fallback = QLabel("TEXA")
            fallback.setFont(QFont("Segoe UI", 18, QFont.Black))
            fallback.setStyleSheet("color:#4b0f12;")
            self.logo.setText("TEXA")
            self.logo.setFont(QFont("Segoe UI", 18, QFont.Black))
            self.logo.setStyleSheet("color:#4b0f12;")
        else:
            scaled = pix.scaledToWidth(self.LOGO_WIDTH, Qt.SmoothTransformation)
            self.logo.setPixmap(scaled)

# ---------------- Sidebar button ----------------
class NavButton(QPushButton):
    """Rounded silver button to mimic the screenshot."""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setObjectName("NavButton")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(50)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCheckable(True)


# ---------------- Example Pages (replace internals with your real widgets) ----------------
class HomePage(QWidget):
    def __init__(self,
                 parent=None,
                 bg_path=r"D:\cone_design\Screenshot 2025-08-22 120829.png",   # background image
                 logo_path=r"D:\cone_design\assets\texa_logo.png",                  # TEXA logo
                 title_text="Intelli Cone",
                 on_go=None):
        super().__init__(parent)

        # --- state ---
        self._bg_pix = QPixmap(bg_path) if bg_path else QPixmap()
        self._bg_scaled = QPixmap()
        self._bg_lighten = 0.8   # 0 = dark, 1 = fully white
        self._on_go = on_go

        # --- title ---
        self.title = QLabel(title_text, self)
        f = QFont("Times New Roman", 36, QFont.Bold)
        self.title.setFont(f)
        self.title.setStyleSheet(
            "color:#111; background:rgba(255,255,255,0.8); "
            "padding:6px 14px; border-radius:8px;"
        )
        self.title.setAlignment(Qt.AlignCenter)

        # --- GO button ---
        self.go_btn = QPushButton("GO", self)
        self.go_btn.setCursor(Qt.PointingHandCursor)
        self.go_btn.setFixedSize(QSize(150, 40))
        self.go_btn.setStyleSheet("""
            QPushButton {
                background-color:#7b001b; color:#fff;
                border:none; border-radius:8px;
                font-family:"Times New Roman"; font-size:16px; font-weight:bold;
            }
            QPushButton:hover { opacity:0.9; }
        """)
        if callable(self._on_go):
            self.go_btn.clicked.connect(self._on_go)

        # --- TEXA logo ---
        self._logo = QLabel(self)
        if logo_path:
            lpix = QPixmap(logo_path)
            if not lpix.isNull():
                self._logo.setPixmap(lpix.scaledToWidth(140, Qt.SmoothTransformation))
        self._logo.setAlignment(Qt.AlignRight | Qt.AlignTop)

        # --- layout ---
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(20)
        lay.addStretch(1)
        lay.addWidget(self.title, 0, Qt.AlignHCenter)
        lay.addWidget(self.go_btn, 0, Qt.AlignHCenter)
        lay.addStretch(2)

        self.setMinimumSize(900, 600)

    # --- handle resizing ---
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self._bg_pix.isNull():
            self._bg_scaled = self._bg_pix.scaled(
                self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
        # reposition logo at top-right
        if self._logo.pixmap():
            lw = self._logo.pixmap().width()
            lh = self._logo.pixmap().height()
            self._logo.setGeometry(self.width()-lw-30, 20, lw, lh)

    # --- paint background ---
    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        if not self._bg_scaled.isNull():
            x = (self.width()-self._bg_scaled.width())//2
            y = (self.height()-self._bg_scaled.height())//2
            p.drawPixmap(x, y, self._bg_scaled)

        if self._bg_lighten > 0:
            alpha = int(self._bg_lighten*255)
            p.fillRect(self.rect(), QColor(255,255,255,alpha))

#------------------------------------------------------------------------------------------
class ChooseGoodDialog(QDialog):
    """Pick a GOOD reference .xlsx from reports/references."""
    def __init__(self, refs_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose a GOOD Reference"); self.setModal(True)
        lay = QVBoxLayout(self)
        self.list = QListWidget(); self.list.setIconSize(QSize(80, 60)); lay.addWidget(self.list)

        files = sorted(refs_dir.glob("*.xlsx"), key=lambda p: (len(p.stem), p.stem))
        if not files:
            it = QListWidgetItem("No GOOD references found. Please Train first.")
            it.setFlags(it.flags() & ~Qt.ItemIsSelectable); self.list.addItem(it)
        else:
            for x in files:
                name = x.stem
                icon = QIcon()
                thumb = refs_dir / f"{name}.bmp"
                if thumb.exists():
                    pix = QPixmap(str(thumb))
                    if not pix.isNull(): icon = QIcon(pix)
                it = QListWidgetItem(icon, name); it.setData(Qt.UserRole, str(x))
                self.list.addItem(it)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def selected_path(self):
        it = self.list.currentItem()
        return Path(it.data(Qt.UserRole)) if it and it.data(Qt.UserRole) else None

# ---------- Job details ----------
class PredictSetupDialog(QDialog):
    """
    One popup:
      - pick a GOOD name (from Train index)
      - enter Job
    OK is enabled only when a name is selected AND Job is non-empty.
    """
    def __init__(self, names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prediction Setup")
        self._selected_name = None

        lay = QVBoxLayout(self)

        lay.addWidget(QLabel("Choose GOOD Name:"))
        self.listw = QListWidget(self); lay.addWidget(self.listw)

        # form for Job only
        form = QFormLayout()
        self.edt_job = QLineEdit()
        form.addRow("Job:", self.edt_job)
        lay.addLayout(form)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)

        # Populate names (show only the name strings)
        names = sorted({str(n).strip() for n in names if n and str(n).strip()})
        if names:
            for n in names:
                it = QListWidgetItem(n)
                it.setData(Qt.UserRole, n)
                self.listw.addItem(it)
            self.listw.setCurrentRow(0)
        else:
            it = QListWidgetItem("No GOOD names found. Please Train first.")
            it.setFlags(Qt.NoItemFlags)
            self.listw.addItem(it)

        # validate to enable/disable OK
        self.listw.currentRowChanged.connect(self._update_ok_enabled)
        self.edt_job.textChanged.connect(self._update_ok_enabled)
        self._update_ok_enabled()

    def _update_ok_enabled(self):
        has_name = bool(self.listw.currentItem() and self.listw.currentItem().flags() & Qt.ItemIsSelectable)
        has_job  = bool(self.edt_job.text().strip())
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(has_name and has_job)

    def _accept(self):
        it = self.listw.currentItem()
        if not it:
            return
        self._selected_name = it.data(Qt.UserRole)
        self.accept()

    def values(self):
        return {
            "name": self._selected_name,
            "job": self.edt_job.text().strip(),
        }

#---------------------------------------------------------------------------------------------------
# They let the UI run end-to-end without your real ML pipeline.

def process_image_to_dfs_only(bgr_frame):
    """
    Convert a frame to two small DataFrames shaped like your pipeline expects.
    Each has a 'Quality' column. Here we mark everything 'Good'.
    Replace with your real processing later.
    """
    return (
        pd.DataFrame({"Quality": ["Good"]}),  # thread
        pd.DataFrame({"Quality": ["Good"]}),  # ring
    )

def compare_df(df_ref, df_test):
    """
    Compare reference vs test and return a DataFrame with at least 'Quality'.
    In the stub, just return df_test (assume it's already evaluated).
    """
    if "Quality" not in df_test.columns:
        return pd.DataFrame({"Quality": ["Good"]})
    return df_test.copy()

def verdict(cmp_thread, cmp_ring):
    """
    Decide Good/Bad based on presence of any 'Bad' rows.
    """
    def has_bad(df):
        return (not df.empty) and ("Quality" in df.columns) and (df["Quality"].eq("Bad").any())
    return "Bad" if has_bad(cmp_thread) or has_bad(cmp_ring) else "Good"

def log_prediction_result(root_dir: Path, data: dict):
    """
    Append a JSONL line so Dashboard can read it later.
    File: <root_dir>/prediction/results_log.jsonl
    """
    pred_dir = Path(root_dir) / "prediction"
    pred_dir.mkdir(parents=True, exist_ok=True)
    log_path = pred_dir / "results_log.jsonl"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception:
        pass
#----------------------------------------------------------------------------------
class GoodReferenceDialog(QDialog):
    """
    Choose a GOOD reference file. If none available, show a message
    and disable OK (matches your screenshot behavior).
    """
    def __init__(self, ref_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose a GOOD reference")
        self._selected = None

        lay = QVBoxLayout(self)

        self.listw = QListWidget(self)
        lay.addWidget(self.listw)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)

        # Normalize inputs to Paths
        paths = sorted([Path(p) for p in ref_paths], key=lambda x: x.stem.lower())

        if paths:
            for p in paths:
                it = QListWidgetItem(p.stem)          # <-- show only name
                it.setData(Qt.UserRole, str(p))       # <-- keep full path in data
                self.listw.addItem(it)
            self.listw.setCurrentRow(0)
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(True)
        else:
            msg = "No GOOD references found. Please add at least one in Train page."
            it = QListWidgetItem(msg)
            it.setFlags(Qt.NoItemFlags)               # non-selectable placeholder
            self.listw.addItem(it)
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)

    def _accept(self):
        it = self.listw.currentItem()
        if not it:
            return
        self._selected = Path(it.data(Qt.UserRole))
        self.accept()

    def selected_path(self) -> Path:
        return self._selected
#--------------------------------------------------------

class NameListDialog(QDialog):
    """Pick a saved TRAIN 'Good Name'."""
    def __init__(self, names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose GOOD Name")
        self._selected = None

        lay = QVBoxLayout(self)
        self.listw = QListWidget(self); lay.addWidget(self.listw)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self._accept); btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        names = sorted({n.strip() for n in names if n and str(n).strip()})
        if names:
            for n in names:
                it = QListWidgetItem(str(n))      # show only the name
                it.setData(Qt.UserRole, str(n))   # keep value
                self.listw.addItem(it)
            self.listw.setCurrentRow(0)
            btns.button(QDialogButtonBox.Ok).setEnabled(True)
        else:
            it = QListWidgetItem("No GOOD names found. Please Train first.")
            it.setFlags(Qt.NoItemFlags)
            self.listw.addItem(it)
            btns.button(QDialogButtonBox.Ok).setEnabled(False)

    def _accept(self):
        it = self.listw.currentItem()
        if not it: return
        self._selected = it.data(Qt.UserRole)
        self.accept()

    def selected_name(self):
        return self._selected


#========================================================================================
class PredictionPage(QWidget):
    """
    Intelli Cone – Prediction
    - Page header (center title + TEXA logo via PageHeader)
    - Camera preview with moving scan line
    - AUTO (continuous) / MANUAL + TRIGGER
    - Saves results ONLY to Excel (no JSON):
        D:/cone_design/reports/prediction/prediction_log.xlsx  (sheet: Results)
      And per-run comparison export:
        D:/cone_design/reports/comparison/compare_<YYYYmmdd_HHMMSS>_<ref>.xlsx
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PredictionPage")

        # ---- colors / paths (safe fallbacks if constants missing) ----
        self.BORDER  = globals().get("BORDER", "#94a3b8")
        self.REDWINE = globals().get("REDWINE", "#5a191f")
        self.base_dir = Path("D:/cone_design/reports")
        self.pred_dir = self.base_dir / "prediction"
        self.comp_dir = self.base_dir / "comparison"
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.comp_dir.mkdir(parents=True, exist_ok=True)
        self.results_xlsx = self.pred_dir / "prediction_log.xlsx"  # append log here

        # ---- state ----
        self.mode_auto = False
        self.prediction_enabled = False
        self.df_ref_thread = pd.DataFrame()
        self.df_ref_ring   = pd.DataFrame()
        self._job_info = None
        self._ref_name = None
        self._scan_x = 0
        self._scan_dir = 4
        self._scanning = True
        self._last_color = "#9CA3AF"
        self._frame_counter = 0

        # ===== Main vertical layout (no sidebar here) =====
        main = QVBoxLayout(self)
        main.setContentsMargins(16, 16, 16, 16)
        main.setSpacing(12)

        # ---- Page header (center title + logo on right) ----
        main.addWidget(PageHeader("Intelli Cone"))

        # ---- Middle row: camera on left, round controls on right ----
        mid = QHBoxLayout()
        mid.setSpacing(12)

        cam_wrap = QFrame()
        cam_wrap.setStyleSheet(f"QFrame {{ border:1px solid {self.BORDER}; border-radius:12px; background:#ffffff; }}")
        cam_lay = QVBoxLayout(cam_wrap)
        cam_lay.setContentsMargins(12, 12, 12, 12)
        cam_lay.setSpacing(8)

        self.cam = QLabel()
        self.cam.setFixedSize(880, 540)  # tweak if needed
        self.cam.setStyleSheet(f"border:3px solid {self._last_color}; border-radius:12px; background:#000;")
        self.cam.setAlignment(Qt.AlignCenter)
        cam_lay.addWidget(self.cam, alignment=Qt.AlignCenter)

        mid.addWidget(cam_wrap, 1)

        # Right control column
        btn_col = QVBoxLayout()
        btn_col.setSpacing(12)

        self.btn_auto = QPushButton("AUTO")
        self._style_circle(self.btn_auto, self.REDWINE)
        self.btn_auto.clicked.connect(self._on_auto_pressed)
        btn_col.addWidget(self.btn_auto, 0, Qt.AlignTop)

        self.btn_manual = QPushButton("MANUAL")
        self._style_circle(self.btn_manual, self.REDWINE)
        self.btn_manual.clicked.connect(self._on_manual_pressed)
        btn_col.addWidget(self.btn_manual)

        self.btn_trigger = QPushButton("TRIGGER")
        self._style_circle(self.btn_trigger, self.REDWINE)
        self.btn_trigger.clicked.connect(self._on_trigger_clicked)
        self.btn_trigger.setVisible(False)
        btn_col.addWidget(self.btn_trigger)

        btn_col.addStretch(1)
        mid.addLayout(btn_col)

        main.addLayout(mid)

        # ---- Result card ----
        self.card = QLabel("Tap AUTO to start, or choose MANUAL → TRIGGER for one-shot.")
        self.card.setAlignment(Qt.AlignCenter)
        self.card.setStyleSheet(
            f"QLabel {{ border:1px solid {self.BORDER}; border-radius:10px; padding:12px; "
            "background:#FFFFFF; font-weight:700; color:#0f172a; }}"
        )
        main.addWidget(self.card)

        # ---- Camera guard + timer loop ----
        # If you still use ref_dirs() elsewhere, keep it; otherwise comment out the next line.
        self.root, self.refs = ref_dirs()
        self.camguard = CameraGuard(preferred_index=0, width=1280, height=720)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._upd)
        self.timer.start(30)

    # ---------- styles ----------
    def _style_circle(self, btn, redwine="#5a191f"):
        d = 84
        btn.setFixedSize(d, d)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                border-radius: {d//2}px;
                border: 2px solid #94a3b8;
                background-color: #ffffff;
                font-weight: 900; color: #0f172a;
            }}
            QPushButton:hover {{
                border-color: {redwine};
                background-color: rgba(123,30,58,20);
                color: {redwine};
            }}
            QPushButton:pressed {{
                border-color: {redwine};
                background-color: rgba(123,30,58,41);
                color: {redwine};
            }}
        """)
    def _load_training_index(self):
        """
        Load Train index from Excel (preferred) or CSV fallback.
        Returns a DataFrame with columns: name, job, timestamp, image_path (if available).
        """
        idx_xlsx = Path(r"D:/cone_design/reports/training/excel/training_index.xlsx")
        csv1     = Path(r"D:/cone_design/reports/training/excel/training_index.csv")
        csv2     = Path(r"D:/cone_design/reports/training/excel") / "training_index.csv"

        # Excel
        try:
            if idx_xlsx.exists():
                return pd.read_excel(idx_xlsx, sheet_name="Captures")
        except Exception:
            pass

        # CSV fallbacks
        for p in (csv1, csv2):
            try:
                if p.exists():
                    return pd.read_csv(p)
            except Exception:
                pass

        return pd.DataFrame()

    def _build_reference_from_latest(self, df_all, chosen_name: str) -> bool:
        """
        From the training index df, pick the latest row for chosen_name,
        read its image, and compute reference DataFrames.
        """
        if df_all.empty or "name" not in df_all.columns or "image_path" not in df_all.columns:
            return False

        dfn = df_all[df_all["name"].astype(str).str.strip() == str(chosen_name).strip()].copy()
        if dfn.empty:
            return False

        # choose the latest (by timestamp if present)
        if "timestamp" in dfn.columns:
            try:
                dfn["_ts"] = pd.to_datetime(dfn["timestamp"], errors="coerce")
                dfn = dfn.sort_values("_ts")
            except Exception:
                pass

        row = dfn.iloc[-1]
        img_path = str(row.get("image_path", "")).strip()
        if not img_path or not Path(img_path).exists():
            return False

        img = cv2.imread(img_path)  # BGR
        if img is None:
            return False

        ref_thread, ref_ring = process_image_to_dfs_only(img)
        self.df_ref_thread = ref_thread if ref_thread is not None else pd.DataFrame()
        self.df_ref_ring   = ref_ring   if ref_ring   is not None else pd.DataFrame()
        self._ref_name = chosen_name
        return True
    # ---------- prerequisite pickers ----------
    def _ensure_ref_and_job(self) -> bool:
    # Show ONE setup dialog if ref/job not set
      if (self.df_ref_thread.empty and self.df_ref_ring.empty) or not self._job_info:
        df_idx = self._load_training_index()
        names = df_idx["name"].tolist() if "name" in df_idx.columns else []

        dlg = PredictSetupDialog(names, self)
        if dlg.exec_() != QDialog.Accepted:
            return False
        vals = dlg.values()
        chosen_name = vals.get("name")
        if not chosen_name:
            return False

        # build reference from latest capture for that name
        if not self._build_reference_from_latest(df_idx, chosen_name):
            self.card.setText("Could not build reference from Train data for the selected name.")
            return False

        # store job info (only job now)
        if not vals.get("job"):
            return False
        self._job_info = {"job": vals.get("job", "")}

      return True

 # ---------- AUTO / MANUAL / TRIGGER ----------
    def _on_auto_pressed(self):
        if not self._ensure_ref_and_job():
            self.card.setText("AUTO canceled.")
            return
        self.btn_trigger.setVisible(False)
        self.mode_auto = True
        self.prediction_enabled = True
        self.card.setText(f"AUTO mode: job '{self._job_info.get('job','')}'. Predicting…")

    def _on_manual_pressed(self):
        self.mode_auto = False
        self.prediction_enabled = False
        self.btn_trigger.setVisible(True)
        self.card.setText("MANUAL mode: press TRIGGER to capture & predict.")
        if not self._ensure_ref_and_job():
            self.card.setText("MANUAL canceled.")
            self.btn_trigger.setVisible(False)
  
    def _on_trigger_clicked(self):
        if not self._ensure_ref_and_job():
            self.card.setText("TRIGGER canceled.")
            return
        ok, frame = self.camguard.read()
        if ok:
            self._predict_on(frame)
    


    # ---------- frame loop ----------
    def _upd(self):
        ok, frame = self.camguard.read()
        if not ok:
            return

        # vertical scan overlay
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self._scan_x += self._scan_dir
        if self._scan_x >= w or self._scan_x <= 0:
            self._scan_dir *= -1
        if self._scanning:
            cv2.line(rgb, (self._scan_x, 0), (self._scan_x, h), (0, 200, 0), 2)

        self.cam.setPixmap(QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)))

        if self.mode_auto and self.prediction_enabled:
            self._frame_counter = (self._frame_counter + 1) % 10
            if self._frame_counter == 0:
                self._predict_on(frame)

    # ---------- prediction ----------
    def _predict_on(self, bgr_frame):
        df_tst_thread, df_tst_ring = process_image_to_dfs_only(bgr_frame)

        cmp_thread = compare_df(self.df_ref_thread, df_tst_thread) if not self.df_ref_thread.empty and not df_tst_thread.empty else pd.DataFrame()
        cmp_ring   = compare_df(self.df_ref_ring,   df_tst_ring)   if not self.df_ref_ring.empty   and not df_tst_ring.empty   else pd.DataFrame()

        def fail_pct(df):
            if df is None or len(df) == 0:
                return 0.0
            bad = (df["Quality"] == "Bad").sum() if "Quality" in df.columns else 0
            return (bad * 100.0 / len(df)) if len(df) else 0.0

        f_thread = fail_pct(cmp_thread)
        f_ring   = fail_pct(cmp_ring)
        v = verdict(cmp_thread, cmp_ring)
        is_good = (v == "Good")

        color = "#059669" if is_good else "#DC2626"
        if color != self._last_color:
            self._last_color = color
            self.cam.setStyleSheet(f"border: 3px solid {color}; border-radius:12px; background:#000;")

        self.card.setText(
            f"{'GOOD ✅' if is_good else 'BAD ❌'}\n"
            f"Thread bad: {f_thread:.1f}% | Ring bad: {f_ring:.1f}%\n{v}"
        )
        # --- after self.card.setText(...) and before exporting comparison is fine ---

        try:
            jobtype     = (self._job_info or {}).get("job", "")
            operator_id = (self._job_info or {}).get("operator_id", "")
            notes       = (self._job_info or {}).get("notes", "")
            db_insert_prediction(
                jobtype=jobtype,
                is_good=bool(is_good),
                ref_name=self._ref_name or "",
                operator_id=operator_id,
                notes=notes,
                thread_bad_pct=float(f_thread),
                ring_bad_pct=float(f_ring),
            )
        except Exception as e:
            # non-fatal: show on card but don't crash
            self.card.setText(self.card.text() + f"\n(SQL save failed: {e})")


        # -------------- SAVE TO EXCEL (no JSON) --------------
        self._append_result_to_excel({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "job": (self._job_info or {}).get("job", ""),
            "operator_id": (self._job_info or {}).get("operator_id", ""),
            "notes": (self._job_info or {}).get("notes", ""),
            "ref_name": (self._ref_name or "unknown"),
            "is_good": bool(is_good),
            "thread_bad_pct": float(f_thread),
            "ring_bad_pct": float(f_ring),
            "verdict": str(v),
        })

        # Per-run comparison export (Thread/Ring compare sheets)
        self._export_comparison_excel(cmp_thread, cmp_ring, self._ref_name)

    # ---------- Excel I/O ----------
    def _append_result_to_excel(self, row: dict):
        try:
            if self.results_xlsx.exists():
                old = pd.read_excel(self.results_xlsx, sheet_name="Results")
                df = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
            else:
                df = pd.DataFrame([row])
            with pd.ExcelWriter(self.results_xlsx, engine="openpyxl", mode="w") as w:
                df.to_excel(w, index=False, sheet_name="Results")
        except Exception as e:
            self.card.setText(f"Result save failed: {e}")

    def _export_comparison_excel(self, cmp_thread: pd.DataFrame, cmp_ring: pd.DataFrame, ref_name: str):
        try:
            ts_tag = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            out = self.comp_dir / f"compare_{ts_tag}_{ref_name or 'unknown'}.xlsx"
            with pd.ExcelWriter(out, engine="openpyxl", mode="w") as w:
                (cmp_thread if cmp_thread is not None else pd.DataFrame()).to_excel(
                    w, index=False, sheet_name="Thread_Compare"
                )
                (cmp_ring if cmp_ring is not None else pd.DataFrame()).to_excel(
                    w, index=False, sheet_name="Ring_Compare"
                )
        except Exception:
            pass  # non-fatal

    def closeEvent(self, e):
        if hasattr(self, "camguard") and self.camguard:
            self.camguard.release()
        e.accept()

#================================================================================
# ===== Trigger dialog (Name + Job only) =====
from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QPushButton, QHBoxLayout
)

class TriggerInfoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trigger Info")

        form = QFormLayout(self)
        self.name_edit = QLineEdit(self)
        self.job_edit  = QLineEdit(self)

        form.addRow("Name:", self.name_edit)
        form.addRow("Job:",  self.job_edit)

        btn_save   = QPushButton("Save", self)
        btn_cancel = QPushButton("Cancel", self)
        btn_save.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        row = QHBoxLayout()
        row.addWidget(btn_save)
        row.addWidget(btn_cancel)
        form.addRow(row)

    def values(self):
        return {
            "name": self.name_edit.text().strip(),
            "job":  self.job_edit.text().strip(),
        }

# ----------------- imports needed -----------------

import cv2
try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False


# ----------------- TrainPage -----------------
class TrainPage(QWidget):
    """
    Train page with camera preview and TRIGGER button only.
    On TRIGGER: opens dialog (Name, Job); after Save,
    - saves PNG capture to training/captures
    - appends Excel row to training/excel/training_index.xlsx
    Optionally calls on_trigger(frame, info) after saving.
    """
    def __init__(self, on_trigger=None, parent=None):
        super().__init__(parent)
        self.setObjectName("TrainPage")
        self._on_trigger_cb = on_trigger
        self._last_border = "#9CA3AF"
        self.border_col = globals().get("BORDER", self._last_border)
        self.redwine    = globals().get("REDWINE", "#5a191f")

        # ---- storage paths ----
        self.base_dir     = Path("D:/cone_design/reports")
        self.captures_dir = self.base_dir / "training" / "captures"
        self.excel_dir    = self.base_dir / "training" / "excel"
        self.index_xlsx   = self.excel_dir / "training_index.xlsx"
        self._ensure_dirs()

        # ===== UI =====
        main = QVBoxLayout(self)
        main.setContentsMargins(16,16,16,16)
        main.setSpacing(12)

        # Header (use PageHeader if defined)
        try:
            main.addWidget(PageHeader("Intelli Cone"))
        except NameError:
            hdr = QFrame(self); hdr.setObjectName("PageHeader")
            hl = QHBoxLayout(hdr); hl.setContentsMargins(14,12,14,12)
            t = QLabel("Train", hdr); t.setObjectName("HeaderTitle"); hl.addWidget(t); hl.addStretch(1)
            main.addWidget(hdr)

        mid = QHBoxLayout(); mid.setSpacing(12)

        # ---- Camera preview ----
        cam_wrap = QFrame()
        cam_wrap.setStyleSheet(
            f"QFrame {{ border:1px solid {self.border_col}; border-radius:12px; background:#ffffff; }}"
        )
        cam_lay = QVBoxLayout(cam_wrap)
        cam_lay.setContentsMargins(12,12,12,12)
        cam_lay.setSpacing(8)

        self.cam = QLabel()
        self.cam.setFixedSize(880, 540)
        self.cam.setAlignment(Qt.AlignCenter)
        self.cam.setStyleSheet(
            f"border:3px solid {self._last_border}; border-radius:12px; background:#000;"
        )
        cam_lay.addWidget(self.cam, alignment=Qt.AlignCenter)
        mid.addWidget(cam_wrap, 1)

        # ---- Right column: Trigger only ----
        right = QVBoxLayout(); right.setSpacing(12)

        self.btn_trigger = QPushButton("TRIGGER")
        self._style_circle(self.btn_trigger, self.redwine)
        self.btn_trigger.clicked.connect(self._on_trigger_clicked)
        right.addWidget(self.btn_trigger, 0, Qt.AlignTop)

        right.addStretch(1)
        mid.addLayout(right)
        main.addLayout(mid)

        # ---- Status card ----
        self.card = QLabel("Ready. Press TRIGGER to capture a training image.")
        self.card.setAlignment(Qt.AlignCenter)
        self.card.setStyleSheet(
            f"QLabel {{ border:1px solid {self.border_col}; border-radius:10px; padding:12px; "
            "background:#FFFFFF; font-weight:700; color:#0f172a; }}"
        )
        main.addWidget(self.card)

        # ===== Camera =====
        try:
            self.camguard = CameraGuard(preferred_index=0, width=1280, height=720)
        except NameError:
            self.camguard = cv2.VideoCapture(0)
            self.camguard.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camguard.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self._last_frame = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(30)

    # ---------- helpers ----------
    def _ensure_dirs(self):
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        self.excel_dir.mkdir(parents=True, exist_ok=True)

    def _style_circle(self, btn, accent="#A6244B"):
        d = 84
        btn.setFixedSize(d, d)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                border-radius: {d//2}px;
                border: 2px solid #94a3b8;
                background-color: #ffffff;
                font-weight: 900; color: #0f172a;
            }}
            QPushButton:hover {{
                border-color: {accent};
                background-color: rgba(123,30,58,20);
                color: {accent};
            }}
            QPushButton:pressed {{
                border-color: {accent};
                background-color: rgba(123,30,58,41);
                color: {accent};
            }}
        """)

    def _update_frame(self):
        ok, frame = self._read_frame()
        if not ok:
            return
        self._last_frame = frame  # BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.cam.setPixmap(QPixmap.fromImage(qimg))

    def _read_frame(self):
        if hasattr(self.camguard, "read"):
            try:
                return self.camguard.read()
            except TypeError:
                ok, frame = self.camguard.read()
                return ok, frame
        return False, None

    def _on_trigger_clicked(self):
        # --- dialog for name + job ---
        dlg = TriggerInfoDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        info = dlg.values()
        name, job = info.get("name",""), info.get("job","")
        if not name or not job:
            QMessageBox.warning(self, "Missing data", "Please fill both Name and Job.")
            return

        frame = self._last_frame
        if frame is None:
            QMessageBox.warning(self, "No camera", "No camera frame available yet.")
            return

        try:
            img_path = self._save_capture(frame, info)
            self.card.setText(f"Saved: {img_path.name}  |  Name: {name}  Job: {job}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            self.card.setText(f"Save failed: {e}")
            return

        # flash border
        self.cam.setStyleSheet("border:3px solid #2563EB; border-radius:12px; background:#000;")
        QTimer.singleShot(140, lambda: self.cam.setStyleSheet(
            f"border:3px solid {self._last_border}; border-radius:12px; background:#000;"
        ))

        if callable(self._on_trigger_cb):
            try:
                self._on_trigger_cb(frame, info)
            except Exception:
                pass

    def _save_capture(self, bgr_frame, info: dict) -> Path:
        from datetime import datetime
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img_path = self.captures_dir / f"cap_{ts_tag}.bmp"

        ok = cv2.imwrite(str(img_path), bgr_frame)
        if not ok or not img_path.exists():
            raise RuntimeError(f"cv2.imwrite failed for: {img_path}")

        # Excel index row
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": info.get("name", ""),
            "job":  info.get("job", ""),
            "image_path": str(img_path),
        }

        if self.index_xlsx.exists():
            df = pd.read_excel(self.index_xlsx, sheet_name="Captures")
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        with pd.ExcelWriter(self.index_xlsx, engine="openpyxl", mode="w") as w:
            df.to_excel(w, index=False, sheet_name="Captures")

        return img_path

    def closeEvent(self, e):
        if hasattr(self, "camguard") and self.camguard is not None:
            try:
                self.camguard.release()
            except Exception:
                pass
        e.accept()


#---------------------------------------------------------------------------------------------------------
# ======= Dashboard (neat & unique) =======
from pathlib import Path
from datetime import datetime
import sqlite3

from PyQt5.QtCore import Qt, QDate
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,
    QDateEdit, QLineEdit, QTableWidget, QTableWidgetItem
)

# Fallbacks (use your globals if already set)
try:
    LOGO_PATH
except NameError:
    LOGO_PATH = Path(r"D:\\cone_design\\assets\\texa_logo.png")

try:
    DB_PATH
except NameError:
    DB_PATH = Path(r"D:/cone_design/reports/data/app.db")


class KPIBadge(QFrame):
    """Small KPI tile used at the top of the dashboard."""
    def __init__(self, title, value="0"):
        super().__init__()
        self.setObjectName("BodyCard")
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 12, 16, 12)
        self.lab_title = QLabel(title)
        self.lab_title.setStyleSheet("font-weight:700; color:#374151;")
        self.lab_val = QLabel(str(value))
        self.lab_val.setStyleSheet("font-size:28px; font-weight:800;")
        v.addWidget(self.lab_title)
        v.addWidget(self.lab_val)

    def setValue(self, value):
        self.lab_val.setText(str(value))


class DashboardPage(QWidget):
    """
    Clean dashboard:
      - Header with centered title + TEXA logo (right)
      - KPI row (Total / Good / Bad / Good %)
      - Filters (From, To, Job)
      - Recent predictions table
      - Daily summary grouped by Date & Job
    Reads from SQLite DB at DB_PATH, table: prediction_log
      columns: (id, ts, date, time, jobtype, good_or_bad, ref_name, thread_bad_pct, ring_bad_pct)
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ---------- Header ----------
        header = QFrame(self)
        header.setObjectName("PageHeader")
        hh = QHBoxLayout(header)
        hh.setContentsMargins(14, 12, 14, 12)
        hh.setSpacing(8)

        # Right logo
        logo = QLabel(header)
        logo.setObjectName("HeaderLogo")
        p = Path(r"D:\cone_design\texa_logo.png")  # make sure LOGO_PATH points to your Texa logo file
        px = QPixmap(str(p)) if p.exists() else QPixmap()
        if not px.isNull():
            px = px.scaledToHeight(42, Qt.SmoothTransformation)
            logo.setPixmap(px)
        else:
            # graceful fallback text if image missing
            logo.setText("TEXA")
            logo.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            #logo.setFont(QFont("Segoe UI", 18, QFont.Bold))
            font = QFont("Times New Roman", 18, QFont.Bold)
            logo.setFont(font)

        # Left ghost spacer (same width as logo) so title stays visually centered
        ghost = QLabel(header)
        ghost.setFixedWidth(logo.sizeHint().width() or 100)

        title = QLabel("Dashboard", header)
        title.setObjectName("HeaderTitle")
        title.setAlignment(Qt.AlignCenter)

        # layout order: [ghost] [title expands center] [logo right]
        hh.addWidget(ghost, 0, Qt.AlignLeft | Qt.AlignVCenter)
        hh.addWidget(title, 1)
        hh.addWidget(logo, 0, Qt.AlignRight | Qt.AlignVCenter)

        root.addWidget(header)

        # ---------- KPI row ----------
        krow = QHBoxLayout()
        krow.setSpacing(12)
        self.kpi_total = KPIBadge("Total")
        self.kpi_good = KPIBadge("Good")
        self.kpi_bad  = KPIBadge("Bad")
        self.kpi_rate = KPIBadge("Good %")
        krow.addWidget(self.kpi_total)
        krow.addWidget(self.kpi_good)
        krow.addWidget(self.kpi_bad)
        krow.addWidget(self.kpi_rate)
        root.addLayout(krow)

        # ---------- Filters ----------
        filt = QFrame(self)
        filt.setObjectName("BodyCard")
        fl = QHBoxLayout(filt)
        fl.setContentsMargins(12, 10, 12, 10)
        fl.setSpacing(8)

        self.ed_from = QDateEdit(calendarPopup=True)
        self.ed_from.setDisplayFormat("yyyy-MM-dd")
        self.ed_to = QDateEdit(calendarPopup=True)
        self.ed_to.setDisplayFormat("yyyy-MM-dd")
        today = QDate.currentDate()
        self.ed_from.setDate(today)
        self.ed_to.setDate(today)

        self.ed_job = QLineEdit()
        self.ed_job.setPlaceholderText("Job (optional)")

        b_apply = QPushButton("Refresh")
        b_clear = QPushButton("Clear")
        b_apply.clicked.connect(self._refresh_all)
        b_clear.clicked.connect(self._clear_filters)

        fl.addWidget(QLabel("From:"))
        fl.addWidget(self.ed_from)
        fl.addWidget(QLabel("To:"))
        fl.addWidget(self.ed_to)
        fl.addWidget(QLabel("Job:"))
        fl.addWidget(self.ed_job, 1)
        fl.addWidget(b_apply)
        fl.addWidget(b_clear)

        root.addWidget(filt)

        # ---------- Recent predictions ----------
        card1 = QFrame(self)
        card1.setObjectName("BodyCard")
        v1 = QVBoxLayout(card1)
        v1.setContentsMargins(12, 12, 12, 12)
        v1.addWidget(QLabel("Recent predictions", card1))

        self.tbl_recent = QTableWidget(0, 7, card1)
        self.tbl_recent.setHorizontalHeaderLabels(
            ["Date", "Time", "Job", "Good/Bad", "Ref", "ThreadBad%", "RingBad%"]
        )
        self.tbl_recent.horizontalHeader().setStretchLastSection(True)
        self.tbl_recent.setAlternatingRowColors(True)
        v1.addWidget(self.tbl_recent)

        root.addWidget(card1)

        # ---------- Daily summary ----------
        card2 = QFrame(self)
        card2.setObjectName("BodyCard")
        v2 = QVBoxLayout(card2)
        v2.setContentsMargins(12, 12, 12, 12)
        v2.addWidget(QLabel("Daily summary", card2))

        self.tbl_summary = QTableWidget(0, 5, card2)
        self.tbl_summary.setHorizontalHeaderLabels(["Date", "Job", "Total", "Good", "Bad"])
        self.tbl_summary.horizontalHeader().setStretchLastSection(True)
        self.tbl_summary.setAlternatingRowColors(True)
        v2.addWidget(self.tbl_summary)

        root.addWidget(card2, 1)

        # Optional style polish
        self.setStyleSheet("""
        #PageHeader { background: #ffffff; border-radius: 12px; }
        #HeaderTitle { font: 700 20px "Times new roman"; }
        #HeaderLogo { margin-left: 8px; }
        #BodyCard { background: #ffffff; border-radius: 12px; }
        """)

        # First fill
        self._refresh_all()

    # ===================== DB helpers (local) =====================
    def _con(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(DB_PATH))
        con.execute("PRAGMA foreign_keys=ON;")
        return con

    def _db_recent(self, limit=200):
        try:
            con = self._con()
            cur = con.cursor()
            cur.execute("""
                SELECT date,time,jobtype,good_or_bad,ref_name,thread_bad_pct,ring_bad_pct
                  FROM prediction_log
                 ORDER BY id DESC
                 LIMIT ?;
            """, (limit,))
            rows = cur.fetchall()
            con.close()
            return rows
        except Exception:
            return []

    def _db_summary(self, date_from=None, date_to=None, job_like=None):
        try:
            con = self._con()
            cur = con.cursor()
            q = """
                SELECT date,
                       jobtype,
                       COUNT(*) AS total,
                       SUM(CASE WHEN good_or_bad='Good' THEN 1 ELSE 0 END) AS good_cnt,
                       SUM(CASE WHEN good_or_bad='Bad'  THEN 1 ELSE 0 END)  AS bad_cnt
                  FROM prediction_log
                 WHERE 1=1
            """
            args = []
            if date_from:
                q += " AND date >= ?"; args.append(date_from)
            if date_to:
                q += " AND date <= ?"; args.append(date_to)
            if job_like:
                q += " AND jobtype LIKE ?"; args.append(f"%{job_like}%")
            q += " GROUP BY date, jobtype ORDER BY date DESC, jobtype ASC;"
            cur.execute(q, args)
            rows = cur.fetchall()
            con.close()
            return rows
        except Exception:
            return []

    def _db_kpis(self, date_from=None, date_to=None, job_like=None):
        try:
            con = self._con()
            cur = con.cursor()
            q = "SELECT COUNT(*), SUM(CASE WHEN good_or_bad='Good' THEN 1 ELSE 0 END) FROM prediction_log WHERE 1=1"
            args = []
            if date_from:
                q += " AND date >= ?"; args.append(date_from)
            if date_to:
                q += " AND date <= ?"; args.append(date_to)
            if job_like:
                q += " AND jobtype LIKE ?"; args.append(f"%{job_like}%")
            cur.execute(q, args)
            total, good = cur.fetchone() or (0, 0)
            con.close()
        except Exception:
            total, good = 0, 0
        total = total or 0
        good = good or 0
        bad = max(total - good, 0)
        good_pct = (good * 100.0 / total) if total else 0.0
        return {"total": total, "good": good, "bad": bad, "good_pct": good_pct}

    # ===================== UI helpers =====================
    def _filters(self):
        dfrom = self.ed_from.date().toString("yyyy-MM-dd") if self.ed_from.date().isValid() else None
        dto   = self.ed_to.date().toString("yyyy-MM-dd") if self.ed_to.date().isValid() else None
        job   = self.ed_job.text().strip() or None
        return dfrom, dto, job

    def _clear_filters(self):
        today = QDate.currentDate()
        self.ed_from.setDate(today)
        self.ed_to.setDate(today)
        self.ed_job.clear()
        self._refresh_all()

    def _refresh_all(self):
        date_from, date_to, job_like = self._filters()

        # KPIs
        k = self._db_kpis(date_from, date_to, job_like)
        self.kpi_total.setValue(k["total"])
        self.kpi_good.setValue(k["good"])
        self.kpi_bad.setValue(k["bad"])
        self.kpi_rate.setValue(f"{k['good_pct']:.1f}%")

        # Recent (unfiltered last N)
        recent = self._db_recent(limit=200)
        self.tbl_recent.setRowCount(0)
        for r in recent:
            ridx = self.tbl_recent.rowCount()
            self.tbl_recent.insertRow(ridx)
            for c, val in enumerate(r):
                it = QTableWidgetItem(str(val))
                if c == 3:  # Good/Bad
                    it.setTextAlignment(Qt.AlignCenter)
                self.tbl_recent.setItem(ridx, c, it)

        # Summary (filtered)
        rows = self._db_summary(date_from, date_to, job_like)
        self.tbl_summary.setRowCount(0)
        for r in rows:
            ridx = self.tbl_summary.rowCount()
            self.tbl_summary.insertRow(ridx)
            for c, val in enumerate(r):
                self.tbl_summary.setItem(ridx, c, QTableWidgetItem(str(val)))

# ---------------- Main Window (no global header) ----------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelli Cone")
        self.resize(1365, 768)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Left sidebar (keep a reference so we can hide on Home)
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(240)
        s = QVBoxLayout(self.sidebar)
        s.setContentsMargins(16, 20, 16, 20)
        s.setSpacing(18)

        # --- Sidebar buttons ---
        self.btn_home = NavButton("  Home")
        self.btn_pred = NavButton("  Prediction")
        self.btn_train = NavButton("  Train")
        self.btn_dash = NavButton("  Dashboard")
        self.btn_exit = NavButton("  Exit")

        # make them toggleable and mutually exclusive
        for b in (self.btn_home, self.btn_pred, self.btn_train, self.btn_dash, self.btn_exit):
            b.setCheckable(True)

        # optional: keep Exit not part of the exclusive group
        from PyQt5.QtWidgets import QButtonGroup
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        for b in (self.btn_home, self.btn_pred, self.btn_train, self.btn_dash):
            self.nav_group.addButton(b)

        for b in (self.btn_home, self.btn_pred, self.btn_train, self.btn_dash):
            s.addWidget(b)
        s.addStretch(1)
        s.addWidget(self.btn_exit)

        # Main stack
        self.stack = QStackedWidget()

        # Home wires GO -> Prediction and uses your background image
        self.pages = {
            "home": HomePage(
                on_go=self._go_to_prediction,
                bg_path=r"D:\cone_design\Screenshot 2025-08-22 120829.png",
                logo_path=r"D:\cone_design\texa_logo.png"
            ),
            "prediction": PredictionPage(),
            "train": TrainPage(),
            "dashboard": DashboardPage(),
        }

        for key in ("home", "prediction", "train", "dashboard"):
            self.stack.addWidget(self.pages[key])

        root.addWidget(self.sidebar)
        root.addWidget(self.stack, 1)

        # Navigation logic
        self.btn_home.clicked.connect(lambda: self._nav("home"))
        self.btn_pred.clicked.connect(lambda: self._nav("prediction"))
        self.btn_train.clicked.connect(lambda: self._nav("train"))
        self.btn_dash.clicked.connect(lambda: self._nav("dashboard"))
        self.btn_exit.clicked.connect(self.close)

        # Start on Home
        self._nav("home")
        self.apply_styles()

    # Called by HomePage's GO button
    def _go_to_prediction(self):
        self._nav("prediction")

    def _nav(self, key: str):
        self.stack.setCurrentWidget(self.pages[key])

        # toggle sidebar visibility (hide on Home)
        self.sidebar.setVisible(key != "home")

        # update checked state for visual selection
        mapping = {
            "home": self.btn_home,
            "prediction": self.btn_pred,
            "train": self.btn_train,
            "dashboard": self.btn_dash,
        }
        # clear all, then set selected
        for b in (self.btn_home, self.btn_pred, self.btn_train, self.btn_dash):
            b.setChecked(False)
        if key in mapping:
            mapping[key].setChecked(True)

    def apply_styles(self):
        self.setStyleSheet("""
/* Global */
QWidget { 
    color: #111827; 
    font-family: "Times New Roman";  
}
HomePage { background: transparent; }   /* let paintEvent draw */

/* ============ SIDEBAR ============ */
#Sidebar {
    background: #7b0f1b;
    border-right: 8px solid #1a1a1a;
}

/* Default (idle) button */
#Sidebar QPushButton#NavButton {
    background: #e9edf2;                 /* lighter default */
    color: #222222;
    border: 1px solid rgba(0,0,0,38);
    border-radius: 24px;
    padding: 10px 14px;
    text-align: left;
    font-size: 14px;
    font-weight: 600;
    font-family: "Times New Roman", serif;
}

/* Hover */
#Sidebar QPushButton#NavButton:hover {
    background: #d9dee5;
}

/* Pressed (click feedback) */
#Sidebar QPushButton#NavButton:pressed {
    background: #cfd5dd;
}

/* Selected / Active */
#Sidebar QPushButton#NavButton:checked {
    background: #4b0f12;                 /* dark maroon active */
    color: #ffffff;
    border: 2px solid #ffffff;
}

/* Keep a slight hover even when selected */
#Sidebar QPushButton#NavButton:checked:hover {
    background: #3f0d10;
}

/* ============ PAGE HEADER ============ */
#PageHeader {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
}
#HeaderTitle {
    font-size: 28px;
    font-weight: 800;
    color: #111827;
    padding: 2px 8px;
    font-family: "Times New Roman", serif;
}
#HeaderLogo { padding-right: 8px; }

/* ============ BODY CARDS ============ */
#BodyCard {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 20px;
    font-family: "Times New Roman", 
}

/* ============ STACK LIGHT / LABELS ============ */
#StackLight {
    border-radius: 6px;
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                stop:0 #ff6161, stop:1 #b30000);
}
#DTLabelBig {
    font-size: 24px;
    font-weight: 800;
    font-family: "Times New Roman", serif;
}

/* ============ TABLE ============ */

QTableWidget {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    gridline-color: #e5e7eb;
    selection-background-color: #e5f3ff;
    selection-color: #111827;
    font-family: "Times New Roman";
}
QHeaderView::section {
    background: #f3f4f6;
    color: #111827;
    padding: 6px 8px;
    border: 1px solid #e5e7eb;
    font-family: "Times New Roman";
}

/* ============ GENERIC BUTTONS ============ */
QPushButton {
    font-family: "Times New Roman", serif;
}
""")
# ---------------- Run ----------------
if __name__ == "__main__":
    db_init()
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
