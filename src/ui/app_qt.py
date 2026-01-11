from __future__ import annotations

import os
import argparse
import importlib
import pkgutil
import sys
from pathlib import Path

import concurrent.futures as cf
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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
)

from huggingface_hub import snapshot_download

from src.database.ops import init_db, append_rows_locked, load_db, DEFAULT_COLUMNS
from src.database.pseudonymizer import TextPseudonymizer
from src.database.security import get_or_create_salt_file
from src.database.text_extraction import TextExtractor


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

        self.extractor = TextExtractor()

        init_dlg = self._make_progress("Starting application", "Initializing window…", 0)
        init_dlg.setRange(0, 0)  # indeterminate/busy
        init_dlg.setLabelText("Loading EDS-PSEUDO model…")
        QApplication.processEvents()

        # EDS-PSEUDO model initialization (used later during commit).
        self.pseudonymizer: TextPseudonymizer | None = None
        try:
            artifacts_path = resolve_eds_model_path(eds_path)
            prepare_eds_registry(artifacts_path.parent)
            self.pseudonymizer = TextPseudonymizer(
                model_path=str(artifacts_path),
                auto_update=False,
                secret_salt="CHANGE_ME",
            )
            init_dlg.close()
        except Exception as e:
            # Keep the UI usable (e.g., for extraction), but block commits until the model is available.
            init_dlg.close()
            self.pseudonymizer = None
            QMessageBox.critical(
                self,
                "EDS-PSEUDO initialization failed",
                "Could not initialize the eds-pseudo model.\n\n"
                f"Details: {e}\n\n"
                "You can provide --eds_path to point to the hf_cache folder, or ensure hf_cache/artifacts "
                "is available relative to this script.",
            )

        self.btn_choose_db = QPushButton("Select database…")
        self.btn_choose_db.clicked.connect(self.choose_db)

        self.btn_init_db = QPushButton("Create DB…")
        self.btn_init_db.clicked.connect(self.create_db)

        self.btn_add_docs = QPushButton("Documents…")
        self.btn_add_docs.clicked.connect(self.add_documents)

        self.btn_commit = QPushButton("Commit to DB")
        self.btn_commit.clicked.connect(self.commit)
        if self.pseudonymizer is None:
            self.btn_commit.setEnabled(False)

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
        top.addWidget(QLabel("Database:"), 0, 0)
        top.addWidget(self.db_path_edit, 0, 1, 1, 3)
        top.addWidget(self.btn_choose_db, 0, 4)
        top.addWidget(self.btn_init_db, 0, 5)

        actions = QHBoxLayout()
        actions.addWidget(self.btn_add_docs)
        actions.addStretch(1)
        actions.addWidget(self.btn_commit)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(actions)
        layout.addWidget(self.table)
        layout.addWidget(self.status)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)

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
    
    def choose_db(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select database CSV",
            str(Path.home()),
            "CSV files (*.csv);;All files (*.*)",
        )
        if path:
            self.db_path_edit.setText(path)
            self.status.setText(f"Selected DB: {path}")

    def create_db(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Create database CSV",
            str(Path.home() / "clinical_db.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            init_db(path, columns=DEFAULT_COLUMNS)
            # Create/store a persistent pseudonymization salt for this DB
            get_or_create_salt_file(path)
            self.db_path_edit.setText(path)
            self.status.setText(f"Created DB: {path}")
        except Exception as e:
            self._error(str(e))

    def add_documents(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select PDF documents",
            str(Path.home()),
            "PDF files (*.pdf);;All files (*.*)",
        )
        if not files:
            return

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

        self.status.setText(f"Loaded {added} file(s). Enter PID for each (ORDER only for known PID).")

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
            self._error("No documents loaded. Click 'Documents…' first.")
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

        self.table.setRowCount(0)
        self.status.setText(f"Committed {n} document(s) to DB.")


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

    def _error(self, message: str):
        QMessageBox.critical(self, "Error", message)
        self.status.setText("Error.")


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
