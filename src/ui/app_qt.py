from __future__ import annotations

import os, sys
import argparse
import importlib
import pkgutil
import sys
from pathlib import Path

import concurrent.futures as cf
import pandas as pd
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt, QSettings, QSize, QObject, QThread, Signal
from PySide6.QtWidgets import (
    QFrame,
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)

from huggingface_hub import snapshot_download

PATH_TO_SRC_FOLDER = str(Path(__file__).parent.parent.parent.resolve())
if PATH_TO_SRC_FOLDER not in sys.path:
    sys.path.append(PATH_TO_SRC_FOLDER)

from src.database.ops import init_db, append_rows_locked, load_db, DEFAULT_COLUMNS
from src.database.pseudonymizer import TextPseudonymizer
from src.database.security import get_or_create_salt_file
from src.database.text_extraction import TextExtractor

LIGHT_QSS = """
QMainWindow, QWidget {
    background: #ffffff;
    color: #111111;
    font-size: 12px;
}
QLabel {
    color: #111111;
}
QLineEdit {
    background: #ffffff;
    color: #111111;
    border: 1px solid #cfcfcf;
    padding: 6px;
    border-radius: 6px;
}
QTableWidget {
    background: #ffffff;
    color: #111111;
    gridline-color: #e6e6e6;
    border: 1px solid #cfcfcf;
}
QHeaderView::section {
    background: #f2f2f2;
    color: #111111;
    padding: 6px;
    border: 0px;
    border-bottom: 1px solid #d9d9d9;
}
QPushButton {
    background: #f2f2f2;
    color: #111111;
    border: 1px solid #cfcfcf;
    padding: 7px 10px;
    border-radius: 8px;
}
QPushButton:hover {
    background: #eaeaea;
}
QPushButton:disabled {
    color: #777777;
    background: #f6f6f6;
    border-color: #e0e0e0;
}
QCheckBox {
    color: #111111;
}
"""

DARK_QSS = """
QMainWindow, QWidget {
    background: #121212;
    color: #f1f1f1;
    font-size: 12px;
}
QLabel {
    color: #f1f1f1;
}
QLineEdit {
    background: #1e1e1e;
    color: #f1f1f1;
    border: 1px solid #3a3a3a;
    padding: 6px;
    border-radius: 6px;
}
QTableWidget {
    background: #1a1a1a;
    color: #f1f1f1;
    gridline-color: #2a2a2a;
    border: 1px solid #2f2f2f;
}
QHeaderView::section {
    background: #202020;
    color: #f1f1f1;
    padding: 6px;
    border: 0px;
    border-bottom: 1px solid #2f2f2f;
}
QPushButton {
    background: #2a2a2a;
    color: #f1f1f1;
    border: 1px solid #3a3a3a;
    padding: 7px 10px;
    border-radius: 8px;
}
QPushButton:hover {
    background: #333333;
}
QPushButton:disabled {
    color: #999999;
    background: #242424;
    border-color: #333333;
}
QCheckBox {
    color: #f1f1f1;
}
"""

class EDSInitWorker(QObject):
    finished = Signal(object)          # emits TextPseudonymizer instance
    failed = Signal(str)               # emits error message

    def __init__(self, eds_path: str | None):
        super().__init__()
        self.eds_path = eds_path

    def run(self) -> None:
        try:
            artifacts_path = resolve_eds_model_path(self.eds_path)
            prepare_eds_registry(artifacts_path.parent)

            pseudo = TextPseudonymizer(
                model_path=str(artifacts_path),
                auto_update=False,
                secret_salt="CHANGE_ME",
            )
            self.finished.emit(pseudo)

        except Exception as e:
            self.failed.emit(str(e))
            
def _import_recursive(package_name: str) -> None:
    """Import a package and all its submodules to trigger any registry decorators."""
    try:
        package = importlib.import_module(package_name)
    except ImportError as e:
        raise RuntimeError(f"Could not import {package_name}: {e}") from e

    if hasattr(package, "__path__"):
        for _, name, _ in pkgutil.walk_packages(package.__path__, package_name + "."):
            try:
                importlib.import_module(name)
            except Exception as e:
                # Non-fatal: some optional submodules may fail, but the core registry should be loaded.
                # Keep going to mirror the behavior in test_eds.py.
                print(f"[eds-pseudo] Warning: failed to import {name}: {e}")


def resolve_eds_model_path(eds_path: str | None) -> Path:
    """
    Resolve an EDS-PSEUDO model 'artifacts' path.

    Resolution order:
    1) CLI --eds_path: expects a folder that contains 'artifacts/'
    2) Relative to this file: <repo>/hf_cache/artifacts
    3) Download from Hugging Face into <repo>/hf_cache, then use artifacts
    """
    # 1) CLI argument provided
    if eds_path:
        cache_dir = Path(eds_path).expanduser().resolve()
        artifacts = cache_dir / "artifacts"
        if artifacts.exists() and artifacts.is_dir():
            return artifacts
        raise FileNotFoundError(
            f"--eds_path was provided but does not contain an 'artifacts' folder: {artifacts}"
        )

    # 2) Assume relative hf_cache next to the app entrypoint
    base_dir = Path(__file__).resolve().parent
    cache_dir = base_dir / "hf_cache"
    artifacts = cache_dir / "artifacts"
    if artifacts.exists() and artifacts.is_dir():
        return artifacts

    # 3) Download model (same approach as test_eds.py)
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="AP-HP/eds-pseudo-public",
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.git*"],
    )
    artifacts = cache_dir / "artifacts"
    if not artifacts.exists():
        raise FileNotFoundError(
            "Downloaded eds-pseudo cache, but could not find 'artifacts' folder at "
            f"{artifacts}. Check the Hugging Face snapshot structure."
        )
    return artifacts


def prepare_eds_registry(cache_dir: Path) -> None:
    """Ensure custom eds_pseudo components are registered (per your test_eds.py)."""
    abs_cache = str(cache_dir.resolve())
    if abs_cache not in sys.path:
        sys.path.insert(0, abs_cache)
    _import_recursive("eds_pseudo")


class MainWindow(QMainWindow):
    def __init__(self, *, eds_path: str | None = None):
        super().__init__()

        self.setWindowTitle("Clinical Database - Document Intake")
        self.resize(980, 560)

        self.db_path_edit = QLineEdit()
        self.db_path_edit.setPlaceholderText("Select a database CSV path...")

        self.settings = QSettings("ClinicalDatabase", "DocumentIntake")
        self.theme = self.settings.value("ui/theme", "light", type=str)

        self.extractor = TextExtractor()

        # Theme toggle button (icon-only, GitHub style)
        self.btn_theme = QPushButton()
        self.btn_theme.setCheckable(True)
        self.btn_theme.setFixedSize(36, 36)
        self.btn_theme.setIconSize(QSize(18, 18))
        self.btn_theme.setToolTip("Toggle light / dark mode")
        self.btn_theme.clicked.connect(self._toggle_theme)

        icon_dir = Path(__file__).parent / "assets" / "icons"
        self.icon_sun = QIcon(str(icon_dir / "sun.svg"))
        self.icon_moon = QIcon(str(icon_dir / "moon.svg"))

        # Prominent in-app message banner (replaces tiny bottom-left status text)
        self.msg_frame = QFrame()
        self.msg_frame.setObjectName("MessageBanner")
        self.msg_frame.setFrameShape(QFrame.StyledPanel)

        self.msg_title = QLabel("Status")
        self.msg_title.setObjectName("MessageTitle")

        msg_layout = QVBoxLayout()
        msg_layout.setContentsMargins(12, 10, 12, 10)
        msg_layout.setSpacing(4)
        msg_layout.addWidget(self.msg_title)

        self.btn_commit = QPushButton("Commit to DB")
        self.btn_commit.clicked.connect(self.commit)

        # EDS-PSEUDO loading
        self._loading_eds(eds_path)

        if self.pseudonymizer is None:
            self.btn_commit.setEnabled(False)

        self.msg_body = QLabel("Ready." if self.pseudonymizer else "Not Ready (pseudonymization model not loaded yet).")
        self.msg_body.setObjectName("MessageBody")
        self.msg_body.setWordWrap(True)

        msg_layout.addWidget(self.msg_body)
        self.msg_frame.setLayout(msg_layout)
        
        # Apply the saved theme now (also sets button label)
        self.set_theme(self.theme)
        self._set_message("info", self.msg_body.text())

        self.btn_choose_db = QPushButton("Select database…")
        self.btn_choose_db.clicked.connect(self.choose_db)

        self.btn_init_db = QPushButton("Create DB…")
        self.btn_init_db.clicked.connect(self.create_db)

        self.btn_add_docs = QPushButton("Documents…")
        self.btn_add_docs.clicked.connect(self.add_documents)

        # pseudo-only export option (checked by default)
        self.chk_pseudo_only = QCheckBox("Make pseudo-only copy")
        self.chk_pseudo_only.setChecked(True)
        self.chk_pseudo_only.setToolTip(
            "After each commit, write a sibling CSV named '<db>_pseudo_only.csv' "
            "containing all columns except DOCUMENT."
        )

        # Table: one row per file, with PID + ORDER entry
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["File path", "PID", "ORDER", "Preview"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 520)
        self.table.setColumnWidth(1, 140)
        self.table.setColumnWidth(2, 80)
        self.table.setColumnWidth(3, 180)

        # Status label
        self.status = QLabel("Ready." if self.pseudonymizer else "Ready (pseudonymization model not loaded).")
        self.status.setStyleSheet("color: #333;")

        top = QGridLayout()
        top.addWidget(self.btn_theme, 0, 6)
        top.addWidget(QLabel("Database:"), 0, 0)
        top.addWidget(self.db_path_edit, 0, 1, 1, 3)
        top.addWidget(self.btn_choose_db, 0, 4)
        top.addWidget(self.btn_init_db, 0, 5)

        actions = QHBoxLayout()
        actions.addWidget(self.btn_add_docs)
        actions.addStretch(1)
        actions.addWidget(self.chk_pseudo_only)
        actions.addWidget(self.btn_commit)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(actions)
        layout.addWidget(self.msg_frame)
        layout.addWidget(self.table)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)

    def _on_eds_ready(self, pseudo: object) -> None:
        self.pseudonymizer = pseudo  # type: ignore[assignment]
        if hasattr(self, "init_dlg") and self.init_dlg:
            self.init_dlg.close()

        self.btn_commit.setEnabled(True)
        self._set_message("info", "EDS-PSEUDO model loaded. Ready.")

    def _on_eds_failed(self, err: str) -> None:
        self.pseudonymizer = None
        if hasattr(self, "init_dlg") and self.init_dlg:
            self.init_dlg.close()

        self.btn_commit.setEnabled(False)
        QMessageBox.critical(
            self,
            "EDS-PSEUDO initialization failed",
            "Could not initialize the eds-pseudo model.\n\n"
            f"Details: {err}\n\n"
            "You can provide --eds_path to point to the hf_cache folder, or ensure hf_cache/artifacts "
            "is available relative to this script.",
        )
        self._set_message("error", "Pseudonymization model not loaded. Commit disabled.")

    def _loading_eds(self, eds_path):
        # EDS-PSEUDO model initialization (async so UI remains responsive)
        self.pseudonymizer: TextPseudonymizer | None = None
        self.btn_commit.setEnabled(False)  # will be enabled when model is ready

        self.init_dlg = self._make_progress("Starting application", "Loading EDS-PSEUDO model…", 0)
        self.init_dlg.setRange(0, 0)  # indeterminate busy animation
        self.init_dlg.setLabelText("Loading EDS-PSEUDO model…")
        self.init_dlg.show()

        self._eds_thread = QThread(self)
        self._eds_worker = EDSInitWorker(eds_path)
        self._eds_worker.moveToThread(self._eds_thread)

        self._eds_thread.started.connect(self._eds_worker.run)
        self._eds_worker.finished.connect(self._on_eds_ready)
        self._eds_worker.failed.connect(self._on_eds_failed)

        # Cleanup
        self._eds_worker.finished.connect(self._eds_thread.quit)
        self._eds_worker.failed.connect(self._eds_thread.quit)
        self._eds_thread.finished.connect(self._eds_worker.deleteLater)
        self._eds_thread.finished.connect(self._eds_thread.deleteLater)

        self._eds_thread.start()

    def _make_progress(self, title: str, label: str, maximum: int) -> QProgressDialog:
        dlg = QProgressDialog(label, None, 0, maximum, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setMinimumDuration(0)  # show immediately
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        dlg.show()
        QApplication.processEvents()
        return dlg
    
    def set_theme(self, theme: str) -> None:
        theme = (theme or "").lower().strip()
        if theme not in {"light", "dark"}:
            theme = "dark"

        self.theme = theme
        self.settings.setValue("ui/theme", self.theme)

        if self.theme == "dark":
            self.btn_theme.setChecked(True)
            self.btn_theme.setIcon(self.icon_moon)
            self.btn_theme.setToolTip("Switch to light mode")
            self.setStyleSheet(DARK_QSS + self._message_banner_qss(dark=True))
        else:
            self.btn_theme.setChecked(False)
            self.btn_theme.setIcon(self.icon_sun)
            self.btn_theme.setToolTip("Switch to dark mode")
            self.setStyleSheet(LIGHT_QSS + self._message_banner_qss(dark=False))

    def _toggle_theme(self) -> None:
        self.set_theme("dark" if self.btn_theme.isChecked() else "light")

    def _message_banner_qss(self, *, dark: bool) -> str:
        # Banner base style + per-severity variants.
        # Use objectName selectors for targeted styling.
        if dark:
            base_bg = "#1b1b1b"
            base_border = "#2f2f2f"
            title_color = "#f1f1f1"
            body_color = "#e6e6e6"
        else:
            base_bg = "#f7f7f7"
            base_border = "#d9d9d9"
            title_color = "#111111"
            body_color = "#222222"

        return f"""
        QFrame#MessageBanner {{
            background: {base_bg};
            border: 1px solid {base_border};
            border-radius: 10px;
        }}
        QLabel#MessageTitle {{
            font-weight: 700;
            color: {title_color};
            font-size: 12px;
        }}
        QLabel#MessageBody {{
            color: {body_color};
            font-size: 12px;
        }}
        """

    def _set_message(self, level: str, message: str) -> None:
        level = (level or "").lower().strip()
        if level not in {"info", "warning", "error"}:
            level = "info"

        if level == "error":
            title = "Error"
        elif level == "warning":
            title = "Warning"
        else:
            title = "Info"

        self.msg_title.setText(title)
        self.msg_body.setText(message)

    def _get_last_dir(self, key: str) -> str:
        val = self.settings.value(key, "", type=str)
        if val and Path(val).exists():
            return val
        return str(Path.home())

    def _set_last_dir(self, key: str, path: str) -> None:
        p = Path(path).expanduser()
        # If a file path is given, store its parent dir; if already a dir, store it.
        folder = p if p.is_dir() else p.parent
        try:
            folder = folder.resolve()
        except Exception:
            pass
        self.settings.setValue(key, str(folder))

    def choose_db(self):
        start_dir = self._get_last_dir("paths/select_db")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select database CSV",
            start_dir,
            "CSV files (*.csv);;All files (*.*)",
        )
        if path:
            self.db_path_edit.setText(path)
            self._set_message("info", f"Selected DB: {path}")
            self._set_last_dir("paths/select_db", path)


    def create_db(self):
        start_dir = self._get_last_dir("paths/create_db")
        default_path = str(Path(start_dir) / "clinical_db.csv")

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Create database CSV",
            default_path,
            "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            init_db(path, columns=DEFAULT_COLUMNS)
            get_or_create_salt_file(path)
            self.db_path_edit.setText(path)
            self._set_message("info", f"Created DB: {path}")
            self._set_last_dir("paths/create_db", path)
        except Exception as e:
            self._error(str(e))


    def add_documents(self):
        start_dir = self._get_last_dir("paths/documents")
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select PDF documents",
            start_dir,
            "PDF files (*.pdf);;All files (*.*)",
        )
        if not files:
            return

        # Persist last directory based on the first selected file
        self._set_last_dir("paths/documents", files[0])

        # Add to table (deduplicate exact same file path already present in the table)
        existing = set()
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it:
                existing.add(it.text().strip())

        added = 0
        for f in files:
            if f in existing:
                continue
            self._add_row(file_path=f, pid="", order="")
            added += 1

        self._set_message("info", f"Loaded {added} file(s). Enter PID for each (ORDER only for known PID).")


    def _pseudo_only_path(self, db_path: str) -> Path:
        p = Path(db_path).expanduser().resolve()
        return p.with_name(f"{p.stem}_pseudo_only{p.suffix}")

    def _write_pseudo_only_copy(self, db_path: str) -> None:
        """
        Create/overwrite '<db_stem>_pseudo_only.csv' with all DB rows but without DOCUMENT.
        """
        src = Path(db_path).expanduser().resolve()
        dst = self._pseudo_only_path(db_path)

        df_full = load_db(str(src))

        # Drop non-pseudonymized text column if present
        if "DOCUMENT" in df_full.columns:
            df_out = df_full.drop(columns=["DOCUMENT"])
        else:
            df_out = df_full

        # Write as a full snapshot (simple + robust)
        df_out.to_csv(dst, index=False)

    def commit(self):
        db_path = self.db_path_edit.text().strip()
        if not db_path:
            self._error("Please select a database file first.")
            return

        if self.pseudonymizer is None:
            self._error(
                "Pseudonymization model is not initialized. "
                "Provide --eds_path pointing to the hf_cache folder (containing artifacts), "
                "or ensure hf_cache/artifacts exists relative to the app script."
            )
            return

        try:
            df = load_db(db_path)
        except FileNotFoundError:
            self._error("Database not found. Create it first or select an existing CSV.")
            return
        except Exception as e:
            self._error(f"Could not read DB: {e}")
            return

        n = self.table.rowCount()
        if n == 0:
            # If DB has at least one non-empty row, do not error; just inform.
            # "Non-empty row" interpreted as: dataframe has at least one row.
            has_any_row = len(df) > 0

            pseudo_msg = ""
            if getattr(self, "chk_pseudo_only", None) is not None and self.chk_pseudo_only.isChecked():
                try:
                    self._write_pseudo_only_copy(db_path)
                    pseudo_msg = " Pseudo-only copy made."
                except Exception as e:
                    # Keep same semantics: DB not updated anyway, but warn that copy failed.
                    QMessageBox.warning(
                        self,
                        "Pseudo-only copy failed",
                        "No documents selected and the database was not updated, but the pseudo-only copy "
                        "could not be updated.\n\n"
                        f"Details: {e}",
                    )
                    self._set_message("info", "No documents selected. The database was not updated.")
                    return

            if has_any_row:
                self._info(f"No documents selected. The database was not updated.{pseudo_msg}")
                return

            # If DB is empty (no rows), keep the previous error-style behavior.
            # (You did not specify a custom message for empty DB; this preserves a useful guard.)
            self._error("No documents selected.")
            return

        required_cols = {"PID", "SOURCE_FILE", "DOCUMENT", "PSEUDO", "ORDER"}
        missing_db = required_cols - set(df.columns)
        if missing_db:
            self._error(
                "Database schema mismatch.\n"
                f"Missing columns in DB: {sorted(missing_db)}\n"
                "Please recreate the DB with the updated DEFAULT_COLUMNS."
            )
            return

        try:
            salt = get_or_create_salt_file(db_path)
            self.pseudonymizer.secret_salt = salt
        except Exception as e:
            self._error(f"Could not initialize pseudonymization salt for this DB:\n{e}")
            return

        # -------------------------
        # Validate rows + collect metadata
        # -------------------------
        rows_meta = []  # keep original order + validated metadata
        for r in range(n):
            fp = (self.table.item(r, 0).text().strip() if self.table.item(r, 0) else "")
            pid = (self.table.item(r, 1).text().strip() if self.table.item(r, 1) else "")
            try:
                f = float(pid)
                if f.is_integer():
                    pid = str(int(f))
            except Exception:
                pass
            order_txt = (self.table.item(r, 2).text().strip() if self.table.item(r, 2) else "")

            if not fp:
                self._error(f"Row {r+1}: missing file path.")
                return

            p = Path(fp)
            if p.suffix.lower() != ".pdf":
                self._error(f"Row {r+1}: only PDF files are allowed:\n{fp}")
                return
            if not p.exists():
                self._error(f"Row {r+1}: file does not exist:\n{fp}")
                return
            if not pid:
                self._error(f"Row {r+1}: PID is required.")
                return

            pid_exists = (df["PID"] == pid).any() if len(df) else False

            if pid_exists:
                if not order_txt:
                    self._error(f"Row {r+1}: ORDER is required because PID={pid} already exists.")
                    return
                try:
                    order_val = int(order_txt)
                except ValueError:
                    self._error(f"Row {r+1}: ORDER must be an integer.")
                    return

                max_order = int(pd.to_numeric(df.loc[df["PID"] == pid, "ORDER"], errors="coerce").max())
                if not (1 <= order_val <= max_order + 1):
                    self._error(
                        f"Row {r+1}: ORDER={order_val} invalid for PID={pid}. "
                        f"Must be between 1 and {max_order + 1}."
                    )
                    return
            else:
                order_val = pd.NA

            rows_meta.append((r, p, pid, order_val))

        # Use half CPU threads (min 1). Threads are safer than processes on Windows.
        max_workers = max(1, (os.cpu_count() or 1) // 2)

        # -------------------------
        # Phase 1: Parallel extraction with visible progress dialog
        # -------------------------
        extract_dlg = self._make_progress("Commit", "Extracting text…", n)
        extract_dlg.setLabelText(f"Extracting text…\nWorkers: {max_workers}")
        QApplication.processEvents()

        extracted_by_row: dict[int, str] = {}
        try:
            with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(self.extractor.pdf_to_text, p): (r, p)
                    for (r, p, _, _) in rows_meta
                }

                completed = 0
                for fut in cf.as_completed(futs):
                    r, p = futs[fut]
                    try:
                        extracted_by_row[r] = fut.result()
                    except Exception as e:
                        extract_dlg.close()
                        self._error(f"Row {r+1}: text extraction failed for {p.name}:\n{e}")
                        return

                    completed += 1
                    extract_dlg.setLabelText(f"Extracting text… ({completed}/{n})\n{p.name}")
                    extract_dlg.setValue(completed)
                    QApplication.processEvents()

        except Exception as e:
            extract_dlg.close()
            self._error(f"Parallel extraction failed:\n{e}")
            return

        extract_dlg.close()

        # Update previews in table order (nice UX; avoids “random” completion order)
        for (r, p, pid, order_val) in rows_meta:
            extracted_text = extracted_by_row.get(r, "")
            preview = extracted_text.strip().replace("\n", " ")
            preview = preview[:160] + ("…" if len(preview) > 160 else "")
            self._set_preview(r, preview)

        # -------------------------
        # Phase 2: Sequential pseudonymization with visible progress dialog
        # -------------------------
        pseudo_dlg = self._make_progress("Commit", "Pseudonymizing extracted texts…", n)

        rows = []
        for idx, (r, p, pid, order_val) in enumerate(rows_meta, start=1):
            extracted_text = extracted_by_row.get(r, "")
            if not extracted_text.strip():
                pseudo_dlg.close()
                self._error(f"Row {r+1}: empty extracted text for {p.name}.")
                return

            pseudo_dlg.setLabelText(f"Pseudonymizing extracted texts… ({idx}/{n})\n{p.name}")
            pseudo_dlg.setValue(idx - 1)
            QApplication.processEvents()

            try:
                pseudo_text = self.pseudonymizer.pseudonymize(extracted_text, pid=pid)
            except Exception as e:
                pseudo_dlg.close()
                self._error(f"Row {r+1}: pseudonymization failed for {p.name}:\n{e}")
                return

            rows.append(
                {
                    "PID": pid,
                    "SOURCE_FILE": str(p.resolve()),
                    "DOCUMENT": extracted_text,
                    "PSEUDO": pseudo_text,
                    "ORDER": order_val,
                }
            )

        pseudo_dlg.setValue(n)
        pseudo_dlg.close()

        new_rows = pd.DataFrame(rows)

        try:
            append_rows_locked(db_path, new_rows)
        except Exception as e:
            self._error(f"Commit failed: {e}")
            return

        if getattr(self, "chk_pseudo_only", None) is not None and self.chk_pseudo_only.isChecked():
            try:
                self._write_pseudo_only_copy(db_path)
            except Exception as e:
                # Do not rollback the main commit; just warn.
                QMessageBox.warning(
                    self,
                    "Pseudo-only copy failed",
                    "Commit succeeded, but the pseudo-only copy could not be updated.\n\n"
                    f"Details: {e}",
                )

        pseudo_note = ""
        if self.chk_pseudo_only.isChecked():
            pseudo_note = f" Pseudo-only copy: {self._pseudo_only_path(db_path)}"
        self.table.setRowCount(0)
        self._set_message("info", f"Committed {n} document(s) to DB.{pseudo_note}")

    def _add_row(self, file_path: str, pid: str, order: str = ""):
        r = self.table.rowCount()
        self.table.insertRow(r)

        fp_item = QTableWidgetItem(file_path)
        fp_item.setFlags(fp_item.flags() ^ Qt.ItemIsEditable)
        self.table.setItem(r, 0, fp_item)

        self.table.setItem(r, 1, QTableWidgetItem(pid))
        self.table.setItem(r, 2, QTableWidgetItem(order))

        preview_item = QTableWidgetItem("")
        preview_item.setFlags(preview_item.flags() ^ Qt.ItemIsEditable)
        self.table.setItem(r, 3, preview_item)

    def _set_preview(self, row: int, preview: str) -> None:
        it = self.table.item(row, 3)
        if it is None:
            it = QTableWidgetItem("")
            it.setFlags(it.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(row, 3, it)
        it.setText(preview)

    def _info(self, message: str):
        QMessageBox.information(self, "Info", message)
        self._set_message("info", message)

    def _error(self, message: str):
        QMessageBox.critical(self, "Error", message)
        self._set_message("error", message)


def main():
    parser = argparse.ArgumentParser(description="Clinical Database - Document Intake")
    parser.add_argument(
        "--eds_path",
        default=None,
        help=(
            "Path to the HuggingFace cache directory containing the eds-pseudo model (expects an 'artifacts' "
            "subdirectory). If omitted, the app tries <script_dir>/hf_cache; if not found, it will download the model."
        ),
    )
    args, qt_args = parser.parse_known_args()
    print("\nStarting application and loading EDS-PSEUDO, this may take a few minutes..")
    app = QApplication([sys.argv[0], *qt_args])
    w = MainWindow(eds_path=args.eds_path)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
