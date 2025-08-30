# app_page_headers_texa.py
# PyQt5 shell: page-level headers (centered "Intelli Cone" + TEXA logo on right)
# Sidebar styled to match your reference image.

# ===== imports =====
import sys, os, time, json
from pathlib import Path
import cv2
import numpy as np
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
from PyQt5.QtCore import Qt, QDate
from PyQt5.QtGui import QFont, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QDateEdit, QComboBox, QPushButton
)
from PyQt5.QtChart import (
    QChart, QChartView, QBarSet, QBarSeries,
    QBarCategoryAxis, QValueAxis
)
#=====================================================================================================
# ======================= DB helpers (inline) =======================
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

DB_PATH = Path("D:/cone_design/reports/data/app.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def db_connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con

def db_init() -> None:
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS prediction_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        jobtype TEXT NOT NULL,
        job_count INTEGER NOT NULL,
        good_or_bad TEXT NOT NULL CHECK (good_or_bad IN ('Good','Bad')),
        ref_name TEXT,
        operator_id TEXT,
        notes TEXT,
        thread_bad_pct REAL DEFAULT 0.0,
        ring_bad_pct REAL DEFAULT 0.0,
        UNIQUE(date, jobtype, job_count)
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dashboard_daily (
        date    TEXT NOT NULL,
        jobtype TEXT NOT NULL,
        total   INTEGER NOT NULL DEFAULT 0,
        good    INTEGER NOT NULL DEFAULT 0,
        bad     INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (date, jobtype)
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS good_refs (
        name       TEXT NOT NULL,
        job        TEXT NOT NULL,
        created_ts TEXT NOT NULL
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_date ON prediction_log(date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_jobtype ON prediction_log(jobtype);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_quality ON prediction_log(good_or_bad);")
    con.commit()
    con.close()

def _next_job_count(con: sqlite3.Connection, date_str: str, jobtype: str) -> int:
    cur = con.cursor()
    cur.execute("""
        SELECT COALESCE(MAX(job_count), 0)
        FROM prediction_log
        WHERE date = ? AND jobtype = ?;
    """, (date_str, jobtype))
    last = cur.fetchone()[0] or 0
    return last + 1

def db_insert_prediction(jobtype: str,
                         is_good: bool,
                         ref_name: Optional[str] = None,
                         operator_id: Optional[str] = None,
                         notes: Optional[str] = None,
                         thread_bad_pct: float = 0.0,
                         ring_bad_pct: float = 0.0) -> None:
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
            now.isoformat(timespec="seconds"),
            date_str, time_str,
            jobtype, jc, good_or_bad,
            ref_name or "", operator_id or "", notes or "",
            float(thread_bad_pct or 0.0), float(ring_bad_pct or 0.0)
        ))
        con.execute("""
            INSERT INTO dashboard_daily (date, jobtype, total, good, bad)
            VALUES (
                ?, ?, 1,
                CASE WHEN ?='Good' THEN 1 ELSE 0 END,
                CASE WHEN ?='Bad'  THEN 1 ELSE 0 END
            )
            ON CONFLICT(date, jobtype) DO UPDATE SET
                total = dashboard_daily.total + 1,
                good  = dashboard_daily.good  + CASE WHEN excluded.good = 1 THEN 1 ELSE 0 END,
                bad   = dashboard_daily.bad   + CASE WHEN excluded.bad  = 1 THEN 1 ELSE 0 END;
        """, (date_str, jobtype, good_or_bad, good_or_bad))
        con.commit()
    finally:
        con.close()

def db_fetch_recent(limit: int = 200) -> List[Tuple]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        SELECT date, time, jobtype, good_or_bad, ref_name, thread_bad_pct, ring_bad_pct
        FROM prediction_log
        ORDER BY id DESC
        LIMIT ?;
    """, (limit,))
    rows = cur.fetchall()
    con.close()
    return rows

def db_fetch_summary(date_from: Optional[str] = None,
                     date_to: Optional[str] = None,
                     job_like: Optional[str] = None) -> List[Tuple]:
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
    args: List = []
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

def db_fetch_kpis(date_from: Optional[str] = None,
                  date_to: Optional[str] = None,
                  job_like: Optional[str] = None) -> Dict[str, float]:
    con = db_connect()
    cur = con.cursor()
    q = """
        SELECT COUNT(*),
               SUM(CASE WHEN good_or_bad='Good' THEN 1 ELSE 0 END)
        FROM prediction_log
        WHERE 1=1
    """
    args: List = []
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

def db_fetch_jobs() -> List[str]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT DISTINCT jobtype FROM prediction_log WHERE jobtype<>'' ORDER BY jobtype;")
    rows = [r[0] for r in cur.fetchall() if r and r[0]]
    con.close()
    return rows


#================================================================================================================

# ===== style / paths =====
REDWINE   = "#5a191f"
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
# ---------------- PredictionPage.py (vertical line scanning, full) ----------------
import time
from pathlib import Path
import cv2, numpy as np, pandas as pd
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton, QDialog, QSizePolicy

# ---- cv2 BGR -> QPixmap
def _cv_to_qpix(bgr):
    if bgr is None:
        return QPixmap()
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

# ---- small UI helpers
class _Title(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.setStyleSheet("color:#222; padding:4px 6px;")

class _Pane(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setObjectName("pane")
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame#pane{background:#fafafa; border:1px solid #dcdcdc; border-radius:8px;}"
            "QLabel#img{background-color:#000; border:0;}"
        )
        lay = QVBoxLayout(self); lay.setContentsMargins(8,8,8,8); lay.setSpacing(6)
        self.title = _Title(title); lay.addWidget(self.title)
        self.img = QLabel(objectName="img")
        self.img.setAlignment(Qt.AlignCenter)
        self.img.setMinimumSize(QSize(240,160))
        self.img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img.setScaledContents(True)
        lay.addWidget(self.img)

    def set_pix(self, pm):
        if not pm.isNull():
            self.img.setPixmap(pm)

    def set_good(self, is_good: bool):
        color = "#059669" if is_good else "#DC2626"
        self.setStyleSheet(
            f"QFrame#pane{{background:#fafafa; border:2px solid {color}; border-radius:8px;}}"
            "QLabel#img{background-color:#000; border:0;}"
        )
        self.title.setStyleSheet(f"color:{color}; padding:4px 6px; font-weight:bold;")

# ======================== PredictionPage ========================
# ======================== PredictionPage (Vertical scan, 2 panes) ========================
from PyQt5.QtCore import pyqtSignal
class PredictionPage(QWidget):
    """
    Live (left) + Thread/Ring previews (right).
    No circular/polar scanning. A vertical scan line sweeps across the live feed.
    AUTO predicts when the line crosses image center (or every 1.2s fallback).
    MANUAL predicts on TRIGGER.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PredictionPage")

        # colors/paths
        self.BORDER  = globals().get("BORDER", "#94a3b8")
        self.REDWINE = globals().get("REDWINE", "#5a191f")
        base = Path("D:/cone_design/reports")
        self.pred_dir = base / "prediction"; self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.comp_dir = base / "comparison"; self.comp_dir.mkdir(parents=True, exist_ok=True)
        self.results_xlsx = self.pred_dir / "prediction_log.xlsx"

        # state
        self.mode_auto = False
        self.prediction_enabled = False
        self.df_ref_thread = pd.DataFrame()
        self.df_ref_ring   = pd.DataFrame()
        self._ref_image_bgr = None
        self._ref_name = None
        self._job_info = None

        # similarity + scan
        self._sim_thresh_hist = 0.80
        self._sim_thresh_kp   = 12
        self._scan_x = 0
        self._scan_dir = 6
        self._scanning = True
        self._align_px = 20
        self._snap_cooldown_s = 0.9
        self._last_snap_ts = 0.0
        self._prev_dx = None

        # overlay cache
        self._overall_good = None
        self._thread_good = None
        self._ring_good = None
        self._line_x_current = 0

        # ===== UI =====
        root = QVBoxLayout(self); root.setContentsMargins(16,16,16,16); root.setSpacing(12)
        root.addWidget(PageHeader("VisI CoNe"))

        row = QHBoxLayout(); row.setSpacing(12)

        # live
        live_wrap = QFrame()
        live_wrap.setStyleSheet(f"QFrame{{border:1px solid {self.BORDER}; border-radius:12px; background:#fff;}}")
        live_lay = QVBoxLayout(live_wrap); live_lay.setContentsMargins(12,12,12,12); live_lay.setSpacing(8)
        self.live = QLabel()
        self.live.setFixedSize(880, 540)
        self.live.setAlignment(Qt.AlignCenter)
        self.live.setStyleSheet(f"border:3px solid #9CA3AF; border-radius:12px; background-color:#000;")
        live_lay.addWidget(self.live, alignment=Qt.AlignCenter)
        row.addWidget(live_wrap, 3)

        # previews (ONLY TWO: Thread + Ring)
        right_col = QVBoxLayout(); right_col.setSpacing(10)
        self.p_thread = _Pane("Thread")   # will show RAW COLOR strip (your image)
        self.p_ring   = _Pane("Ring")     # will show processed (binary) band
        right_col.addWidget(self.p_thread, 1)
        right_col.addWidget(self.p_ring,   1)
        right_wrap = QFrame(); right_wrap.setLayout(right_col); right_wrap.setStyleSheet("QFrame{border:0;}")
        row.addWidget(right_wrap, 2)

        # controls
        side = QVBoxLayout(); side.setSpacing(12)
        self.btn_auto = QPushButton("AUTO");     self._style_round(self.btn_auto);     self.btn_auto.clicked.connect(self._on_auto)
        self.btn_manual = QPushButton("MANUAL"); self._style_round(self.btn_manual);   self.btn_manual.clicked.connect(self._on_manual)
        self.btn_trigger = QPushButton("TRIGGER"); self._style_round(self.btn_trigger); self.btn_trigger.clicked.connect(self._on_trigger)
        self.btn_trigger.setVisible(False)
        side.addWidget(self.btn_auto,   0, Qt.AlignTop)
        side.addWidget(self.btn_manual, 0, Qt.AlignTop)
        side.addWidget(self.btn_trigger,0, Qt.AlignTop)
        side.addStretch(1)
        row.addLayout(side)
        root.addLayout(row)

        self.card = QLabel("Tap AUTO to start, or MANUAL → TRIGGER.")
        self.card.setAlignment(Qt.AlignCenter)
        self.card.setStyleSheet(f"QLabel{{border:1px solid {self.BORDER}; border-radius:10px; padding:12px; background:#fff; font-weight:bold; color:#0f172a;}}")
        root.addWidget(self.card)

        # camera + timers
        self.root, self.refs = ref_dirs()
        self.camguard = CameraGuard(preferred_index=0, width=1280, height=720)

        # frame loop
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(30)

        # periodic AUTO snap
        self.auto_timer = QTimer(self); self.auto_timer.setInterval(1200)
        self.auto_timer.timeout.connect(self._auto_snap)

    def _style_round(self, btn):
        d=84; btn.setFixedSize(d,d); btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton{{border-radius:{d//2}px; border:2px solid #94a3b8; background:#fff; font-weight:bold; color:#0f172a;}}
            QPushButton:hover{{border-color:{self.REDWINE}; background:rgba(123,30,58,0.2); color:{self.REDWINE};}}
            QPushButton:pressed{{border-color:{self.REDWINE}; background:rgba(123,30,58,0.41); color:{self.REDWINE};}}
        """)

    # ---------- training ----------
    def _load_training_index(self):
        xlsx = Path(r"D:/cone_design/reports/training/excel/training_index.xlsx")
        csv1 = Path(r"D:/cone_design/reports/training/excel/training_index.csv")
        csv2 = Path(r"D:/cone_design/reports/training/excel") / "training_index.csv"
        try:
            if xlsx.exists(): return pd.read_excel(xlsx, sheet_name="Captures")
        except Exception: pass
        for p in (csv1, csv2):
            try:
                if p.exists(): return pd.read_csv(p)
            except Exception: pass
        return pd.DataFrame()

    def _build_reference_from_latest(self, df_all, name) -> bool:
        if df_all.empty or "name" not in df_all.columns or "image_path" not in df_all.columns: return False
        dfn = df_all[df_all["name"].astype(str).str.strip() == str(name).strip()].copy()
        if dfn.empty: return False
        if "timestamp" in dfn.columns:
            dfn["_ts"] = pd.to_datetime(dfn["timestamp"], errors="coerce"); dfn = dfn.sort_values("_ts")
        img_path = str(dfn.iloc[-1].get("image_path","")).strip()
        if not img_path or not Path(img_path).exists(): return False
        ref = cv2.imread(img_path)
        if ref is None: return False
        self._ref_image_bgr = ref.copy()
        th, rg = process_image_to_dfs_only(ref)
        self.df_ref_thread = th if th is not None else pd.DataFrame()
        self.df_ref_ring   = rg if rg is not None else pd.DataFrame()
        self._ref_name = name
        self.card.setText(f"Reference set → {name}. AUTO ready.")
        return True

    def _ensure_setup(self) -> bool:
        if (self.df_ref_thread.empty and self.df_ref_ring.empty) or not self._job_info:
            df_idx = self._load_training_index()
            names = df_idx["name"].tolist() if "name" in df_idx.columns else []
            dlg = PredictSetupDialog(names, self)
            if dlg.exec_() != QDialog.Accepted: return False
            vals = dlg.values(); name = vals.get("name")
            if not name: return False
            self.df_ref_thread = pd.DataFrame(); self.df_ref_ring = pd.DataFrame()
            self._ref_image_bgr = None; self._ref_name = None
            if not self._build_reference_from_latest(df_idx, name): return False
            if not vals.get("job"): return False
            self._job_info = {"job": vals.get("job",""), "operator_id": vals.get("operator_id",""), "notes": vals.get("notes","")}
        return True

    # ---------- controls ----------
    def _on_auto(self):
        if not self._ensure_setup(): self.card.setText("AUTO canceled."); return
        self.mode_auto=True; self.prediction_enabled=True; self._scanning=True
        self.btn_trigger.setVisible(False)
        self.card.setText(f"AUTO mode: job '{self._job_info.get('job','')}'. Predicting…")
        self._last_snap_ts = 0.0; self._prev_dx = None
        self.auto_timer.start()

    def _on_manual(self):
        self.mode_auto=False; self.prediction_enabled=False; self._scanning=True
        self.btn_trigger.setVisible(True)
        self.card.setText("MANUAL mode: press TRIGGER to capture & predict.")
        self.auto_timer.stop(); self._prev_dx = None
        if not self._ensure_setup(): self.card.setText("MANUAL canceled."); self.btn_trigger.setVisible(False)

    def _on_trigger(self):
        if not self._ensure_setup(): self.card.setText("TRIGGER canceled."); return
        ok, frame = self.camguard.read()
        if ok: self._predict_on(frame)

    # ---------- periodic AUTO snap ----------
    def _auto_snap(self):
        if not (self.mode_auto and self.prediction_enabled): return
        now = time.time()
        if (now - self._last_snap_ts) < self._snap_cooldown_s: return
        ok, frame = self.camguard.read()
        if not ok or frame is None: return
        self._last_snap_ts = now
        self.card.setText("AUTO snap…")
        self._predict_on(frame)

    # ---------- frame loop ----------
    def _tick(self):
        ok, frame = self.camguard.read()
        if not ok or frame is None: return
        disp = frame.copy()
        h, w = disp.shape[:2]

        # move scan line
        # --- vertical scan line sweep ---
        if self._scanning:
                if self._scan_x <= 10:      self._scan_dir = abs(self._scan_dir)
                if self._scan_x >= w - 10:  self._scan_dir = -abs(self._scan_dir)
                self._scan_x = int(np.clip(self._scan_x + self._scan_dir, 0, w - 1))

        line_x = int(np.clip(self._scan_x, 0, w - 1))
        # --- draw ONE vertical line only (color & thickness you like) ---
        cv2.line(disp, (line_x, 0), (line_x, h), (0, 255, 180), 4)

        # AUTO trigger near center
        now = time.time()
        if self.mode_auto and self.prediction_enabled:
            ready = (now - self._last_snap_ts) >= self._snap_cooldown_s
            dx = (w//2) - line_x
            crossed = (self._prev_dx is not None) and ((self._prev_dx > 0 and dx <= 0) or (self._prev_dx < 0 and dx >= 0))
            close   = abs(dx) <= self._align_px
            if ready and (close or crossed):
                self._last_snap_ts = now
                self._predict_on(frame)
            self._prev_dx = dx

        # ---- build two previews from the vertical strip ----
        # Thread pane = RAW COLOR strip (upper-mid band)
        # Ring pane   = BINARY processed (near center-bottom band)
        strip_w = 18
        base_img, thread_img, ring_img = self._bands_from_line(frame, line_x, strip_w)

        # THREAD: show RAW color (your image) of the thread band
        thread_raw = thread_img if thread_img.size else np.zeros((80,60,3), np.uint8)

        # RING: show Otsu + median processed band
        if ring_img.size:
            ring_g = cv2.cvtColor(ring_img, cv2.COLOR_BGR2GRAY)
            _, ring_bw = cv2.threshold(ring_g, 0, 255, cv2.THRESH_OTSU)
            ring_bw = cv2.medianBlur(ring_bw, 3)
            ring_show = cv2.cvtColor(ring_bw, cv2.COLOR_GRAY2BGR)
        else:
            ring_show = np.zeros((80,60,3), np.uint8)

        # push to UI
        self.p_thread.set_pix(_cv_to_qpix(cv2.resize(thread_raw,(320,200),interpolation=cv2.INTER_NEAREST)))
        self.p_ring.set_pix(_cv_to_qpix(cv2.resize(ring_show,(320,200),interpolation=cv2.INTER_NEAREST)))

        if self._overall_good is not None:
            self._overlay_status(disp, self._overall_good, self._thread_good, self._ring_good)

        self.live.setPixmap(_cv_to_qpix(disp))

    # ---------- bands from line (no circular/polar math) ----------
    def _bands_from_line(self, bgr, line_x, strip_w=16):
        H,W = bgr.shape[:2]
        x1 = max(0, line_x - strip_w//2); x2 = min(W, line_x + strip_w//2)
        strip = bgr[:,x1:x2]
        if strip is None or strip.size==0:
            z = np.zeros((1, max(1,x2-x1),3), np.uint8); return z,z,z
        # Height fractions — tune to your framing
        by1,by2=int(0.55*H),int(0.95*H)   # Base (still computed but not shown)
        ty1,ty2=int(0.20*H),int(0.45*H)   # Thread (upper-mid)
        ry1,ry2=int(0.45*H),int(0.60*H)   # Ring (near center-bottom)
        base   = strip[by1:by2,:].copy() if by2>by1 else np.zeros((1,strip.shape[1],3),np.uint8)
        thread = strip[ty1:ty2,:].copy() if ty2>ty1 else np.zeros((1,strip.shape[1],3),np.uint8)
        ring   = strip[ry1:ry2,:].copy() if ry2>ry1 else np.zeros((1,strip.shape[1],3),np.uint8)
        return base,thread,ring

    # ---------- similarity (Hue mask + ORB) ----------
    def _similar(self, cur_bgr):
        if self._ref_image_bgr is None:
            return True, 1.0, 999

        def mask_from_center(img_shape):
            h, w = img_shape[:2]
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(m, (w // 2, h // 2), min(h, w) // 3, 255, -1)
            return m

        def hue_corr(a, b, m):
            ha = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)[:, :, 0]
            hb = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[:, :, 0]
            h1 = cv2.calcHist([ha], [0], m, [60], [0, 180])
            h2 = cv2.calcHist([hb], [0], m, [60], [0, 180])
            cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)
            return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))

        mask = mask_from_center(cur_bgr.shape)
        corr = hue_corr(self._ref_image_bgr, cur_bgr, mask)

        orb = cv2.ORB_create(nfeatures=800)
        d1 = orb.detectAndCompute(cv2.cvtColor(self._ref_image_bgr, cv2.COLOR_BGR2GRAY), None)[1]
        d2 = orb.detectAndCompute(cv2.cvtColor(cur_bgr,               cv2.COLOR_BGR2GRAY), None)[1]
        good = 0
        if d1 is not None and d2 is not None and len(d1) >= 2 and len(d2) >= 2:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            for pair in bf.knnMatch(d1, d2, k=2):
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.75 * n.distance: good += 1
                elif len(pair) == 1:
                    (m,) = pair
                    if m.distance < 32: good += 1

        ok = (corr >= self._sim_thresh_hist) and (good >= self._sim_thresh_kp)
        return ok, corr, good

    # ---------- overlays ----------
    def _draw_tag(self, img, text, ok=True):
        color = (5, 150, 105) if ok else (220, 38, 38)
        pad = 10
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x1, y1 = pad, pad
        x2, y2 = pad + tw + 20, pad + th + 20
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.putText(img, text, (x1 + 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    def _overlay_status(self, img, overall_good, thread_good=None, ring_good=None):
        self._draw_tag(img, "GOOD", True) if overall_good else self._draw_tag(img, "BAD", False)
        h, w = img.shape[:2]
        def pill(x, label, ok):
            color = (5, 150, 105) if ok else (220, 38, 38)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            y2 = h - 16; y1 = y2 - th - 16; x2 = x + tw + 24
            cv2.rectangle(img, (x, y1), (x2, y2), color, -1)
            cv2.putText(img, label, (x + 12, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            return x2 + 10
        x = 16
        if thread_good is not None: x = pill(x, f"Thread: {'Good' if thread_good else 'Bad'}", thread_good)
        if ring_good   is not None: pill(x,   f"Ring: {'Good'   if ring_good   else 'Bad'}", ring_good)

    # ---------- prediction ----------
    def _predict_on(self, bgr_frame):
        ok_sim, corr, good_kp = self._similar(bgr_frame)
        if not ok_sim:
            self._set_live_border("#DC2626")
            self.card.setText(f"BAD ❌ (Similarity gate)  hist={corr:.2f}, kp={good_kp}")
            self._save_xlsx(dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                                  job=(self._job_info or {}).get("job",""),
                                  operator_id=(self._job_info or {}).get("operator_id",""),
                                  notes="Similarity gate failed", ref_name=self._ref_name or "unknown",
                                  is_good=False, thread_bad_pct=100.0, ring_bad_pct=100.0, verdict="Bad (similarity)"))
            self.p_thread.set_good(False); self.p_ring.set_good(False)
            self._overall_good = False; self._thread_good = False; self._ring_good = False
            return

        df_t_thread, df_t_ring = process_image_to_dfs_only(bgr_frame)
        cmp_thread = compare_df(self.df_ref_thread, df_t_thread) if (self.df_ref_thread is not None and not self.df_ref_thread.empty and df_t_thread is not None and not df_t_thread.empty) else pd.DataFrame()
        cmp_ring   = compare_df(self.df_ref_ring,   df_t_ring)   if (self.df_ref_ring   is not None and not self.df_ref_ring.empty   and df_t_ring   is not None and not df_t_ring.empty)   else pd.DataFrame()

        def fail_pct(df):
            if df is None or len(df)==0: return 100.0
            if "Quality" not in df.columns: return 100.0
            bad=(df["Quality"]=="Bad").sum()
            return (bad*100.0/len(df)) if len(df) else 100.0

        def all_good(df):
            return (df is not None) and (len(df)>0) and ("Quality" in df.columns) and bool((df["Quality"]=="Good").all())

        f_t, f_r = fail_pct(cmp_thread), fail_pct(cmp_ring)
        thread_good = all_good(cmp_thread)
        ring_good   = all_good(cmp_ring)

        self.p_thread.set_good(thread_good)
        self.p_ring.set_good(ring_good)

        try:
            v = verdict(cmp_thread, cmp_ring)
        except Exception:
            v = "Bad"
        both_present = (len(cmp_thread)>0) and (len(cmp_ring)>0)
        overall_good = (v=="Good") and both_present

        self._set_live_border("#059669" if overall_good else "#DC2626")
        self.card.setText(
            f"{'GOOD ✅' if overall_good else 'BAD ❌'}  | "
            f"Thread: {'Good' if thread_good else 'Bad'} ({f_t:.1f}% bad)  | "
            f"Ring: {'Good' if ring_good else 'Bad'} ({f_r:.1f}% bad)  | {v}  "
            f"(hist={corr:.2f}, kp={good_kp})"
        )

        try:
            db_insert_prediction(jobtype=(self._job_info or {}).get("job",""),
                                 is_good=bool(overall_good),
                                 ref_name=self._ref_name or "",
                                 operator_id=(self._job_info or {}).get("operator_id",""),
                                 notes=(self._job_info or {}).get("notes",""),
                                 thread_bad_pct=float(f_t), ring_bad_pct=float(f_r))
        except Exception as e:
            self.card.setText(self.card.text()+f"\n(SQL save failed: {e})")

        self._save_xlsx(dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                             job=(self._job_info or {}).get("job",""),
                             operator_id=(self._job_info or {}).get("operator_id",""),
                             notes=(self._job_info or {}).get("notes",""),
                             ref_name=self._ref_name or "unknown",
                             is_good=bool(overall_good), thread_bad_pct=float(f_t), ring_bad_pct=float(f_r),
                             verdict=str(v)))
        self._export_compare(cmp_thread, cmp_ring, self._ref_name)

        self._overall_good = bool(overall_good)
        self._thread_good  = bool(thread_good)
        self._ring_good    = bool(ring_good)

    # ---------- IO helpers ----------
    def _set_live_border(self, color):
        self.live.setStyleSheet(f"border:3px solid {color}; border-radius:12px; background-color:#000;")

    def _save_xlsx(self, row):
        try:
            if self.results_xlsx.exists():
                old=pd.read_excel(self.results_xlsx, sheet_name="Results"); df=pd.concat([old,pd.DataFrame([row])], ignore_index=True)
            else:
                df=pd.DataFrame([row])
            with pd.ExcelWriter(self.results_xlsx, engine="openpyxl", mode="w") as w:
                df.to_excel(w, index=False, sheet_name="Results")
        except Exception as e:
            self.card.setText(f"Result save failed: {e}")

    def _export_compare(self, t, r, ref_name):
        try:
            ts=time.strftime("%Y%m%d_%H%M%S"); out=self.comp_dir / f"compare_{ts}_{ref_name or 'unknown'}.xlsx"
            with pd.ExcelWriter(out, engine="openpyxl", mode="w") as w:
                (t if t is not None else pd.DataFrame()).to_excel(w, index=False, sheet_name="Thread_Compare")
                (r if r is not None else pd.DataFrame()).to_excel(w, index=False, sheet_name="Ring_Compare")
        except Exception:
            pass


#================================================================================
# ===== Trigger dialog (Name + Job only) =====

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
# ---------------- DashboardPage.py (DB wired) ----------------
from PyQt5.QtCore import Qt, QDate, QTimer
from PyQt5.QtGui import QFont, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QDateEdit, QComboBox, QPushButton
)
from PyQt5.QtChart import (
    QChart, QChartView, QBarSet, QBarSeries,
    QBarCategoryAxis, QValueAxis
)

# Import DB helpers
# from your_db_helpers import db_fetch_jobs, db_fetch_kpis, db_fetch_summary


class DashboardPage(QWidget):
    def __init__(self, parent=None, logo_path: str = None):
        super().__init__(parent)
        self.logo_path = logo_path
        self.setObjectName("DashboardPage")

        # ---------- Global font (Times New Roman) ----------
        self._base_font_family = "Times New Roman"
        self.setFont(QFont(self._base_font_family, 10))

        root = QVBoxLayout(self); root.setContentsMargins(14, 14, 14, 14); root.setSpacing(12)

        # Header
        header = QHBoxLayout()
        self.big_title = QLabel("Dashboard")
        self.big_title.setFont(QFont(self._base_font_family, 22, QFont.Black))
        self.big_title.setStyleSheet("color: rgba(0,0,0,0.28); letter-spacing: 0.5px;")
        header.addWidget(self.big_title)
        header.addStretch()

        self.logo = QLabel(); self.logo.setFixedHeight(28)
        if not self._load_logo():
            self.logo.setText("TEXA")
            self.logo.setFont(QFont(self._base_font_family, 16, QFont.Black))
            self.logo.setStyleSheet("color:#ffffff;")
        logo_wrap = QFrame(); logo_wrap.setObjectName("LogoWrap")
        lw = QHBoxLayout(logo_wrap); lw.setContentsMargins(16, 8, 16, 8); lw.addWidget(self.logo)
        header.addWidget(logo_wrap, 0, Qt.AlignRight | Qt.AlignTop)
        root.addLayout(header)

        # KPI chips
        chips_row = QHBoxLayout(); chips_row.setSpacing(12)
        self.k_total = self._make_kpi("Total", "#fde68a")
        self.k_good  = self._make_kpi("Good",  "#bbf7d0")
        self.k_bad   = self._make_kpi("Bad",   "#c7d2fe")
        self.k_yield = self._make_kpi("Good %","#e9d5ff")
        for w in (self.k_total, self.k_good, self.k_bad, self.k_yield):
            chips_row.addWidget(w)
        kpi_card = self._card(); kpi_card.layout().addLayout(chips_row); root.addWidget(kpi_card)

        # Filters
        filt = QHBoxLayout(); filt.setSpacing(8)
        def lab(t):
            x = QLabel(t)
            x.setFont(QFont(self._base_font_family, 10, QFont.DemiBold))
            x.setStyleSheet("color:#333;")
            return x

        filt.addWidget(lab("From:")); self.dt_from = QDateEdit()
        self.dt_from.setCalendarPopup(True); self.dt_from.setDisplayFormat("dd/MM/yyyy")
        self.dt_from.setDate(QDate.currentDate().addDays(-7)); self.dt_from.setMinimumWidth(120); filt.addWidget(self.dt_from)

        filt.addWidget(lab("To:"));   self.dt_to = QDateEdit()
        self.dt_to.setCalendarPopup(True); self.dt_to.setDisplayFormat("dd/MM/yyyy")
        self.dt_to.setDate(QDate.currentDate()); self.dt_to.setMinimumWidth(120); filt.addWidget(self.dt_to)

        filt.addWidget(lab("Job:"));  self.cmb_job = QComboBox()
        self.cmb_job.setMinimumWidth(220); filt.addWidget(self.cmb_job, 1)

        self.btn_refresh = QPushButton("Refresh"); self.btn_clear = QPushButton("Clear")
        for b in (self.btn_refresh, self.btn_clear):
            b.setObjectName("DeepBtn"); b.setCursor(Qt.PointingHandCursor); b.setMinimumWidth(90)
            b.setFont(QFont(self._base_font_family, 10, QFont.DemiBold))
        filt_card = self._card(); filt_card.layout().addLayout(filt); root.addWidget(filt_card)

        # Auto-refresh (optional)
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(1500)  # 1.5s
        self._auto_timer.timeout.connect(self._recompute_and_update)
        self._auto_timer.start()

        # Charts
        charts = QHBoxLayout(); charts.setSpacing(12)
        self.cv_gb  = self._bar_chart("Good vs Bad")
        self.cv_job = self._bar_chart("Totals by Job")
        charts.addWidget(self._in_card(self.cv_gb))
        charts.addWidget(self._in_card(self.cv_job))
        wrap = QFrame(); wl = QHBoxLayout(wrap); wl.setContentsMargins(0,0,0,0); wl.addLayout(charts); root.addWidget(wrap)

        # Initial data load
        self._reload_jobs_from_db()
        self._recompute_and_update()

        # Styles + wiring
        self._apply_styles()
        self.dt_from.dateChanged.connect(self._recompute_and_update)
        self.dt_to.dateChanged.connect(self._recompute_and_update)
        self.cmb_job.currentIndexChanged.connect(self._recompute_and_update)
        self.btn_refresh.clicked.connect(self._recompute_and_update)
        self.btn_clear.clicked.connect(self._clear_filters)

    # ---------- Public API ----------
    def update_kpis(self, total:int, good:int, bad:int):
        self._set_kpi(self.k_total, total)
        self._set_kpi(self.k_good,  good)
        self._set_kpi(self.k_bad,   bad)
        pct = (good/total*100) if total else 0
        self._set_kpi(self.k_yield, f"{pct:.0f}%")

    def set_good_bad_chart(self, good:int, bad:int):
        chart = self.cv_gb.chart()
        chart.removeAllSeries()
        self._remove_axes(chart)

        s1, s2 = QBarSet("Good"), QBarSet("Bad")
        s1.append([good]); s2.append([bad])
        ser = QBarSeries(); ser.append(s1); ser.append(s2)
        chart.addSeries(ser)

        ax = QBarCategoryAxis(); ax.append(["Counts"])
        ay = QValueAxis()
        # --- integer-only ticks ---
        maxv = max(0, good, bad)
        self._configure_integer_axis(ay, maxv)

        chart.addAxis(ax, Qt.AlignBottom); chart.addAxis(ay, Qt.AlignLeft)
        ser.attachAxis(ax); ser.attachAxis(ay)

    def set_totals_by_job(self, jobs:dict):
        chart = self.cv_job.chart()
        chart.removeAllSeries()
        self._remove_axes(chart)

        bars = QBarSet("Totals"); cats = []
        for k, v in jobs.items():
            cats.append(k); bars.append(int(v))
        ser = QBarSeries(); ser.append(bars); chart.addSeries(ser)

        ax = QBarCategoryAxis(); ax.append(cats or ["Job"])
        ay = QValueAxis()
        # --- integer-only ticks ---
        maxv = max([0] + [int(v) for v in jobs.values()] or [0])
        self._configure_integer_axis(ay, maxv)

        chart.addAxis(ax, Qt.AlignBottom); chart.addAxis(ay, Qt.AlignLeft)
        ser.attachAxis(ax); ser.attachAxis(ay)

    # ---------- Internals ----------
    def _reload_jobs_from_db(self):
        """Populate Job combobox from DB (prediction_log.jobtype)."""
        try:
            jobs = db_fetch_jobs()
        except Exception:
            jobs = []

        self.cmb_job.blockSignals(True)
        self.cmb_job.clear()
        self.cmb_job.addItem("All Jobs")
        for j in jobs:
            self.cmb_job.addItem(str(j))
        self.cmb_job.blockSignals(False)

    def _recompute_and_update(self):
        """Pull KPIs & summaries from SQLite and update UI."""
        # Read filters
        d1 = self.dt_from.date()
        d2 = self.dt_to.date()
        if d2 < d1:
            d1, d2 = d2, d1
            self.dt_from.blockSignals(True); self.dt_to.blockSignals(True)
            self.dt_from.setDate(d1); self.dt_to.setDate(d2)
            self.dt_from.blockSignals(False); self.dt_to.blockSignals(False)

        date_from = f"{d1.year():04d}-{d1.month():02d}-{d1.day():02d}"
        date_to   = f"{d2.year():04d}-{d2.month():02d}-{d2.day():02d}"

        sel_job = self.cmb_job.currentText()
        job_like = None if (sel_job == "All Jobs") else sel_job

        # KPIs
        try:
            kpis = db_fetch_kpis(date_from=date_from, date_to=date_to, job_like=job_like)
        except Exception:
            kpis = {"total": 0, "good": 0, "bad": 0, "good_pct": 0.0}

        total = int(kpis.get("total", 0))
        good  = int(kpis.get("good", 0))
        bad   = int(kpis.get("bad", 0))
        self.update_kpis(total, good, bad)

        # Summary rows (date x job)
        try:
            rows = db_fetch_summary(date_from=date_from, date_to=date_to, job_like=job_like)
        except Exception:
            rows = []

        # Good vs Bad (overall in range)
        sum_good = sum(int(r[3] or 0) for r in rows)
        sum_bad  = sum(int(r[4] or 0) for r in rows)
        self.set_good_bad_chart(sum_good, sum_bad)

        # Totals by Job (aggregate across days)
        job_totals = {}
        for (_date, jobtype, total_ct, _g, _b) in rows:
            job_totals[jobtype] = job_totals.get(jobtype, 0) + int(total_ct or 0)

        ordered = {k: job_totals[k] for k in sorted(job_totals.keys())}
        self.set_totals_by_job(ordered)

    # ---------- Axis helpers (key change) ----------
    def _configure_integer_axis(self, axis: QValueAxis, max_value: int, preferred_step: int = None):
        """
        Force integer-looking Y-axis:
        - labels are integers
        - 'nice' rounded max
        - fixed tick count so ticks land on whole steps
        """
        axis.setLabelFormat("%d")  # show integers (no decimals)

        # Choose a 'nice' step if not provided
        step = preferred_step or self._auto_step(max_value)

        # Round up max to nearest multiple of step
        if step <= 0:
            step = 1
        rounded_max = ((max_value + step - 1) // step) * step
        if rounded_max == 0:
            rounded_max = step  # ensure we have at least one gap

        axis.setRange(0, rounded_max)

        # tickCount = number of ticks = (range/step) + 1
        ticks = int(rounded_max // step) + 1
        # Qt requires tickCount >= 2
        axis.setTickCount(max(2, ticks))

        # Optional: minor ticks (purely cosmetic)
        axis.setMinorTickCount(0)

    def _auto_step(self, max_value: int) -> int:
        """Pick a step so we get ~5–8 ticks, always integer steps."""
        if max_value <= 10:   return 1
        if max_value <= 25:   return 5
        if max_value <= 80:   return 10
        if max_value <= 160:  return 20
        if max_value <= 400:  return 50
        if max_value <= 800:  return 100
        if max_value <= 1600: return 200
        if max_value <= 4000: return 500
        if max_value <= 8000: return 1000
        # scale up in thousands
        # find highest power-of-10-ish step giving ~5–8 ticks
        # e.g., for 23,500 -> step 5000 (≈ 5 ticks)
        for s in (2000, 5000, 10000, 20000, 50000, 100000):
            if max_value <= s * 8:
                return s
        return 200000  # fallback

    # ---------- UI helpers ----------
    def _make_kpi(self, title:str, circle_bg:str) -> QFrame:
        chip = QFrame(); chip.setObjectName("KPIChip")
        lay = QVBoxLayout(chip); lay.setContentsMargins(14,14,14,14); lay.setSpacing(4)
        circle = QFrame(); circle.setObjectName("KPICircle"); circle.setMinimumSize(64,64); circle.setMaximumSize(72,72)
        circle.setStyleSheet(f"background:{circle_bg}; border-radius:36px; border:1px solid rgba(0,0,0,0.05);")
        v = QLabel("0", circle); v.setAlignment(Qt.AlignCenter); v.setFont(QFont(self._base_font_family,16,QFont.Bold)); v.setStyleSheet("color:#222;")
        t = QLabel(title); t.setFont(QFont(self._base_font_family,10,QFont.DemiBold)); t.setStyleSheet("color:#333; margin-top:6px;")
        top = QHBoxLayout(); top.addWidget(circle); top.addStretch()
        lay.addLayout(top); lay.addWidget(t)
        chip._value = v
        return chip

    def _set_kpi(self, chip:QFrame, value):
        chip._value.setText(str(value))

    def _bar_chart(self, title) -> QChartView:
        c = QChart()
        c.setTitle(title)
        c.setTitleFont(QFont(self._base_font_family,11,QFont.DemiBold))
        c.legend().setVisible(False)
        c.setAnimationOptions(QChart.SeriesAnimations)
        v = QChartView(c); v.setMinimumHeight(320); v.setRenderHint(QPainter.Antialiasing, True)
        return v

    def _in_card(self, w:QWidget) -> QFrame:
        f = self._card(); f.layout().addWidget(w); return f

    def _card(self) -> QFrame:
        f = QFrame(); f.setObjectName("Card")
        l = QVBoxLayout(f); l.setContentsMargins(12,12,12,12); l.setSpacing(8)
        return f

    def _load_logo(self) -> bool:
        if not self.logo_path: return False
        pix = QPixmap(self.logo_path)
        if pix.isNull(): return False
        self.logo.setPixmap(pix.scaledToHeight(self.logo.height(), Qt.SmoothTransformation))
        return True

    def resizeEvent(self, e):
        # keep logo crisp when window resizes
        if self.logo.pixmap():
            self.logo.setPixmap(self.logo.pixmap().scaledToHeight(self.logo.height(), Qt.SmoothTransformation))
        super().resizeEvent(e)

    def _clear_filters(self):
        self.dt_from.blockSignals(True); self.dt_to.blockSignals(True); self.cmb_job.blockSignals(True)
        self.dt_from.setDate(QDate.currentDate().addDays(-7))
        self.dt_to.setDate(QDate.currentDate())
        self.cmb_job.setCurrentIndex(0)
        self.dt_from.blockSignals(False); self.dt_to.blockSignals(False); self.cmb_job.blockSignals(False)
        self._recompute_and_update()

    def _remove_axes(self, chart:QChart):
        for ax in list(chart.axes()):
            chart.removeAxis(ax)

    def _apply_styles(self):
        self.setStyleSheet(f"""
            QWidget#DashboardPage {{ background:#f7f7f8; }}
            QFrame#Card {{ background:#ffffff; border:1px solid #e8e8e8; border-radius:12px; }}
            QFrame#LogoWrap {{ background: transparent; border: none; }}
            QFrame#KPIChip {{ background:#ffffff; border:1px solid #eeeeee; border-radius:12px; }}
            QPushButton#DeepBtn {{ background:#4b0f12; color:#fff; border:none; padding:8px 14px; border-radius:8px; }}
            QPushButton#DeepBtn:hover {{ opacity:0.9; }}
        """)

# ---------------- Main Window (no global header) -------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelli Cone")
        self.resize(1365, 768)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Left sidebar
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

        # -------- Main stack --------
        self.stack = QStackedWidget()

        # Create pages (use your existing classes)
        # NOTE: paths use raw strings to avoid backslash escapes on Windows
        home = HomePage(
            on_go=self._go_to_prediction,
            bg_path=r"D:\cone_design\Screenshot 2025-08-22 120829.png",
            logo_path=r"D:\cone_design\texa_logo.png"
        )
        prediction = PredictionPage()
        train = TrainPage()
        dashboard = DashboardPage(logo_path=r"D:\cone_design\texa_logo.png")

        # keep them in a dict for your _nav method
        self.pages = {
            "home": home,
            "prediction": prediction,
            "train": train,
            "dashboard": dashboard,
        }

        # add to stack exactly once
        for key in ("home", "prediction", "train", "dashboard"):
            self.stack.addWidget(self.pages[key])

        # add sidebar + stack to root
        root.addWidget(self.sidebar)
        root.addWidget(self.stack, 1)

        # -------- Navigation wiring --------
        self.btn_home.clicked.connect(lambda: self._nav("home"))
        self.btn_pred.clicked.connect(lambda: self._nav("prediction"))
        self.btn_train.clicked.connect(lambda: self._nav("train"))
        self.btn_dash.clicked.connect(lambda: self._nav("dashboard"))
        self.btn_exit.clicked.connect(self.close)

        # Open directly on Dashboard (as you wanted)
        self._nav("prediction")

        # Styles
        self.apply_styles()

    # Called by HomePage's GO button
    def _go_to_prediction(self):
        self._nav("prediction")

    def _nav(self, key: str):
        # set the page
        self.stack.setCurrentWidget(self.pages[key])

        # toggle sidebar visibility (hide on Home)
        self.sidebar.setVisible(key != "home")

        # update checked state for visual selection
        mapping = {
            "home": self.btn_home,
            "prediction": self.btn_pred,
            "train": self.btn_train,
            "dashboard": self.btn_dash,   # <-- correct button name
        }
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
    font-family: "Times New Roman";
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

# ---------------- Run ----------------# ===== Run demo =====
if __name__ == "__main__":
    db_init()  # <- make sure tables exist
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 720)
    win.show()
    sys.exit(app.exec_())