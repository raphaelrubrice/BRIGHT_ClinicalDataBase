from __future__ import annotations

import os
import sys
import argparse
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
    QMenu,
    QCheckBox,
)

PATH_TO_SRC_FOLDER = str(Path(__file__).parent.parent.parent.resolve())
if PATH_TO_SRC_FOLDER not in sys.path:
    sys.path.append(PATH_TO_SRC_FOLDER)

from src.database.ops import (
    init_db,
    append_rows_locked,
    load_db,
    DEFAULT_COLUMNS,
    extract_IPP_from_document,
    extract_IPP_from_path,
)
from src.database.pseudonymizer import TextPseudonymizer
from src.database.security import get_or_create_salt_file
from src.database.text_extraction import TextExtractor
from src.database.utils import resolve_eds_model_path, prepare_eds_registry
from src.ui.utils import SleepInhibitor

CHUNK_SIZE = 1

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
    background: #ffffff;
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
    color: #e6e6ff;
}
"""

# -------------------------
# Simple i18n (EN default)
# -------------------------
I18N = {
    "en": {
        # window / labels
        "window_title": "Clinical Database - Document Intake",
        "database_label": "Database:",
        "placeholder_db": "Select a database CSV path...",

        # buttons / controls
        "btn_select_db": "Select database…",
        "btn_create_db": "Create DB…",
        "btn_add_documents": "Add documents…",
        "btn_add_from_files": "From Files…",
        "btn_add_from_folder": "From Folder…",
        "btn_commit": "Commit to DB",
        "chk_pseudo_only": "Make pseudo-only copy",

        # tooltips
        "tip_theme_dark": "Switch to dark mode",
        "tip_theme_light": "Switch to light mode",
        "tip_lang_to_fr": "Passer en français",
        "tip_lang_to_en": "Switch to English",
        "tip_pseudo_only": (
            "After each commit, write a sibling CSV named '<db>_pseudo_only.csv' "
            "containing all columns except DOCUMENT."
        ),

        # table
        "table_headers": ["File path", "IPP (auto)", "ORDER (auto)", "Preview"],    

        # message banner titles
        "msg_title_info": "Info",
        "msg_title_warning": "Warning",
        "msg_title_error": "Error",

        # general statuses
        "status_ready": "Ready.",
        "status_not_ready": "Not Ready (pseudonymization model not loaded yet).",

        # dialogs: choose/create/select docs
        "dlg_select_db_title": "Select database CSV",
        "dlg_create_db_title": "Create database CSV",
        "dlg_select_pdf_title": "Select PDF documents",
        "dlg_select_folder_title": "Select folder containing PDFs",

        # progress dialogs
        "progress_start_title": "Starting application",
        "progress_loading_eds": "Loading EDS-PSEUDO model…",
        "progress_commit_title": "Commit",
        "progress_extracting": "Extracting text…",
        "progress_pseudonymizing": "Pseudonymizing extracted texts…",
        "progress_extracting_with_workers": "Extracting text…\nWorkers: {workers}",
        "progress_extracting_step": "Extracting text… ({done}/{total})\n{name}",
        "progress_pseudonymizing_step": "Pseudonymizing extracted texts… ({done}/{total})\n{name}",

        # runtime messages (banner + boxes)
        "eds_loaded_ready": "EDS-PSEUDO model loaded. Ready.",
        "model_not_initialized": (
            "Pseudonymization model is not initialized. "
            "Provide --eds_path pointing to the hf_cache folder (containing artifacts), "
            "or ensure hf_cache/artifacts exists relative to the app script."
        ),
        "please_select_db_first": "Please select a database file first.",
        "db_not_found": "Database not found. Create it first or select an existing CSV.",
        "could_not_read_db": "Could not read DB: {err}",
        "no_docs_loaded_click_docs": "No documents loaded. Click 'Documents…' first.",
        "loaded_files_enter_ipp": "Loaded {n} file(s). IPP/ORDER will be detected automatically at commit.",
        "selected_db": "Selected DB: {path}",
        "created_db": "Created DB: {path}",
        "no_docs_selected_db_not_updated": "No documents selected. The database was not updated.{pseudo_msg}",
        "no_docs_selected": "No documents selected.",
        "schema_mismatch": (
            "Database schema mismatch.\n"
            "Missing columns in DB: {missing}\n"
            "Please recreate the DB with the updated DEFAULT_COLUMNS."
        ),
        "salt_init_failed": "Could not initialize pseudonymization salt for this DB:\n{err}",
        "row_missing_file": "Row {row}: missing file path.",
        "row_only_pdf": "Row {row}: only PDF files are allowed:\n{fp}",
        "row_file_missing": "Row {row}: file does not exist:\n{fp}",
        "row_ipp_required": "Row {row}: IPP is required.",
        "row_order_required_existing_ipp": "Row {row}: ORDER is required because IPP={ipp} already exists.",
        "row_order_int": "Row {row}: ORDER must be an integer.",
        "row_order_invalid": (
            "Row {row}: ORDER={order} invalid for IPP={ipp}. Must be between 1 and {max_order}."
        ),
        "row_extract_failed": "Row {row}: text extraction failed for {name}:\n{err}",
        "parallel_extract_failed": "Parallel extraction failed:\n{err}",
        "row_empty_extracted": "Row {row}: empty extracted text for {name}.",
        "row_pseudo_failed": "Row {row}: pseudonymization failed for {name}:\n{err}",
        "commit_failed": "Commit failed: {err}",
        "commit_ok": "Committed {n} document(s) to DB.{pseudo_note}",

        # pseudo-only copy warnings
        "pseudo_only_copy_failed_title": "Pseudo-only copy failed",
        "pseudo_only_copy_failed_no_docs": (
            "No documents selected and the database was not updated, but the pseudo-only copy "
            "could not be updated.\n\nDetails: {err}"
        ),
        "pseudo_only_copy_failed_after_commit": (
            "Commit succeeded, but the pseudo-only copy could not be updated.\n\nDetails: {err}"
        ),
        "pseudo_only_copy_made": " Pseudo-only copy made.",
        "pseudo_only_copy_note": " Pseudo-only copy: {path}",

        # eds init failure dialog
        "eds_init_failed_title": "EDS-PSEUDO initialization failed",
        "eds_init_failed_body": (
            "Could not initialize the eds-pseudo model.\n\n"
            "Details: {err}\n\n"
            "You can provide --eds_path to point to the hf_cache folder, or ensure hf_cache/artifacts "
            "is available relative to this script."
        ),

        # QMessageBox titles
        "box_info": "Info",
        "box_error": "Error",

        # selection / removal
        "btn_select_mode": "Select file(s)",
        "btn_cancel_select": "Cancel",
        "btn_remove_selected": "Remove selected",
        "chk_select_all": "Select all",
        "dlg_confirm_remove_title": "Confirm removal",
        "dlg_confirm_remove_body": "Remove {n} selected document(s) from the list?",
        "no_selection": "No documents are selected.",
    },
    "fr": {
        "window_title": "Base Clinique - Import de documents",
        "database_label": "Base de données :",
        "placeholder_db": "Sélectionner un chemin CSV de base de données...",

        "btn_select_db": "Sélectionner la base…",
        "btn_create_db": "Créer la base…",
        "btn_add_documents": "Ajouter des documents…",
        "btn_add_from_files": "Depuis des fichiers…",
        "btn_add_from_folder": "Depuis un dossier…",
        "btn_commit": "Enregistrer dans la base",
        "chk_pseudo_only": "Créer une copie pseudonymisée",

        "tip_theme_dark": "Passer en mode sombre",
        "tip_theme_light": "Passer en mode clair",
        "tip_lang_to_fr": "Passer en français",
        "tip_lang_to_en": "Switch to English",
        "tip_pseudo_only": (
            "Après chaque enregistrement, écrire un CSV frère nommé '<db>_pseudo_only.csv' "
            "contenant toutes les colonnes sauf DOCUMENT."
        ),

        "table_headers": ["Chemin du fichier", "IPP (auto)", "ORDER (auto)", "Aperçu"],

        "msg_title_info": "Info",
        "msg_title_warning": "Avertissement",
        "msg_title_error": "Erreur",

        "status_ready": "Prêt.",
        "status_not_ready": "Pas prêt (modèle de pseudonymisation non chargé).",

        "dlg_select_db_title": "Sélectionner une base CSV",
        "dlg_create_db_title": "Créer une base CSV",
        "dlg_select_pdf_title": "Sélectionner des documents PDF",
        "dlg_select_folder_title": "Sélectionner un dossier contenant des PDF",

        "progress_start_title": "Démarrage de l'application",
        "progress_loading_eds": "Chargement du modèle EDS-PSEUDO…",
        "progress_commit_title": "Enregistrement",
        "progress_extracting": "Extraction du texte…",
        "progress_pseudonymizing": "Pseudonymisation des textes extraits…",
        "progress_extracting_with_workers": "Extraction du texte…\nWorkers : {workers}",
        "progress_extracting_step": "Extraction du texte… ({done}/{total})\n{name}",
        "progress_pseudonymizing_step": "Pseudonymisation des textes extraits… ({done}/{total})\n{name}",

        "eds_loaded_ready": "Modèle EDS-PSEUDO chargé. Prêt.",
        "model_not_initialized": (
            "Le modèle de pseudonymisation n'est pas initialisé. "
            "Fournissez --eds_path pointant vers le dossier hf_cache (contenant artifacts), "
            "ou assurez-vous que hf_cache/artifacts existe relativement à ce script."
        ),
        "please_select_db_first": "Veuillez d'abord sélectionner un fichier de base de données.",
        "db_not_found": "Base introuvable. Créez-la d'abord ou sélectionnez un CSV existant.",
        "could_not_read_db": "Impossible de lire la base : {err}",
        "no_docs_loaded_click_docs": "Aucun document chargé. Cliquez d'abord sur 'Documents…'.",
        "loaded_files_enter_ipp": "{n} fichier(s) chargé(s). IPP/ORDRE seront détectés automatiquement lors de l'enregistrement.",
        "selected_db": "Base sélectionnée : {path}",
        "created_db": "Base créée : {path}",
        "no_docs_selected_db_not_updated": "Aucun document sélectionné. La base n'a pas été modifiée.{pseudo_msg}",
        "no_docs_selected": "Aucun document sélectionné.",
        "schema_mismatch": (
            "Incompatibilité du schéma de la base.\n"
            "Colonnes manquantes : {missing}\n"
            "Veuillez recréer la base avec DEFAULT_COLUMNS mis à jour."
        ),
        "salt_init_failed": "Impossible d'initialiser le sel de pseudonymisation pour cette base :\n{err}",
        "row_missing_file": "Ligne {row} : chemin de fichier manquant.",
        "row_only_pdf": "Ligne {row} : seuls les PDF sont autorisés :\n{fp}",
        "row_file_missing": "Ligne {row} : fichier introuvable :\n{fp}",
        "row_ipp_required": "Ligne {row} : IPP requis.",
        "row_order_required_existing_ipp": "Ligne {row} : ORDRE requis car IPP={ipp} existe déjà.",
        "row_order_int": "Ligne {row} : ORDRE doit être un entier.",
        "row_order_invalid": (
            "Ligne {row} : ORDRE={order} invalide pour IPP={ipp}. Doit être entre 1 et {max_order}."
        ),
        "row_extract_failed": "Ligne {row} : échec d'extraction de texte pour {name} :\n{err}",
        "parallel_extract_failed": "Échec de l'extraction parallèle :\n{err}",
        "row_empty_extracted": "Ligne {row} : texte extrait vide pour {name}.",
        "row_pseudo_failed": "Ligne {row} : échec de pseudonymisation pour {name} :\n{err}",
        "commit_failed": "Échec de l'enregistrement : {err}",
        "commit_ok": "{n} document(s) enregistré(s) dans la base.{pseudo_note}",

        "pseudo_only_copy_failed_title": "Échec de la copie pseudonymisée",
        "pseudo_only_copy_failed_no_docs": (
            "Aucun document sélectionné et la base n'a pas été modifiée, mais la copie pseudonymisée "
            "n'a pas pu être mise à jour.\n\nDétails : {err}"
        ),
        "pseudo_only_copy_failed_after_commit": (
            "Enregistrement réussi, mais la copie pseudonymisée n'a pas pu être mise à jour.\n\nDétails : {err}"
        ),
        "pseudo_only_copy_made": " Copie pseudonymisée créée.",
        "pseudo_only_copy_note": " Copie pseudonymisée : {path}",

        "eds_init_failed_title": "Échec d'initialisation EDS-PSEUDO",
        "eds_init_failed_body": (
            "Impossible d'initialiser le modèle eds-pseudo.\n\n"
            "Détails : {err}\n\n"
            "Vous pouvez fournir --eds_path pointant vers hf_cache, ou vérifier que hf_cache/artifacts "
            "est disponible relativement à ce script."
        ),

        "box_info": "Info",
        "box_error": "Erreur",

        # selection / removal
        "btn_select_mode": "Sélectionner fichier",
        "btn_cancel_select": "Annuler",
        "btn_remove_selected": "Supprimer la sélection",
        "chk_select_all": "Tout sélectionner",
        "dlg_confirm_remove_title": "Confirmer la suppression",
        "dlg_confirm_remove_body": "Supprimer {n} document(s) sélectionné(s) de la liste ?",
        "no_selection": "Aucun document sélectionné.",
    },
}


class EDSInitWorker(QObject):
    finished = Signal(object)  # emits TextPseudonymizer instance
    failed = Signal(str)  # emits error message

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


class MainWindow(QMainWindow):
    def __init__(self, *, eds_path: str | None = None):
        super().__init__()
        self.sleep_inhibitor = SleepInhibitor()
        self.settings = QSettings("ClinicalDatabase", "DocumentIntake")

        # Language (default EN)
        self.lang = self.settings.value("ui/lang", "en", type=str)
        if self.lang not in ("en", "fr"):
            self.lang = "en"

        self.theme = self.settings.value("ui/theme", "light", type=str)

        self.setWindowTitle(self.tr("window_title"))
        self.resize(980, 560)

        self.db_path_edit = QLineEdit()
        self.db_path_edit.setPlaceholderText(self.tr("placeholder_db"))

        self.extractor = TextExtractor()

        # Theme toggle button (icon-only, GitHub style)
        self.btn_theme = QPushButton()
        self.btn_theme.setCheckable(True)
        self.btn_theme.setFixedSize(36, 36)
        self.btn_theme.setIconSize(QSize(18, 18))
        self.btn_theme.clicked.connect(self._toggle_theme)

        icon_dir = Path(__file__).parent / "assets" / "icons"
        self.icon_sun = QIcon(str(icon_dir / "sun.svg"))
        self.icon_moon = QIcon(str(icon_dir / "moon.svg"))

        # Prominent in-app message banner (replaces tiny bottom-left status text)
        self.msg_frame = QFrame()
        self.msg_frame.setObjectName("MessageBanner")
        self.msg_frame.setFrameShape(QFrame.StyledPanel)

        self.msg_title = QLabel(self.tr("msg_title_info"))
        self.msg_title.setObjectName("MessageTitle")

        msg_layout = QVBoxLayout()
        msg_layout.setContentsMargins(12, 10, 12, 10)
        msg_layout.setSpacing(4)
        msg_layout.addWidget(self.msg_title)

        self.btn_commit = QPushButton(self.tr("btn_commit"))
        self.btn_commit.clicked.connect(self.commit)

        # EDS-PSEUDO loading
        self._loading_eds(eds_path)

        if self.pseudonymizer is None:
            self.btn_commit.setEnabled(False)

        self.msg_body = QLabel(
            self.tr("status_ready") if self.pseudonymizer else self.tr("status_not_ready")
        )
        self.msg_body.setObjectName("MessageBody")
        self.msg_body.setWordWrap(True)

        msg_layout.addWidget(self.msg_body)
        self.msg_frame.setLayout(msg_layout)

        self.btn_choose_db = QPushButton(self.tr("btn_select_db"))
        self.btn_choose_db.clicked.connect(self.choose_db)

        self.btn_init_db = QPushButton(self.tr("btn_create_db"))
        self.btn_init_db.clicked.connect(self.create_db)

        self.btn_add_docs = QPushButton(self.tr("btn_add_documents"))
        self.btn_add_docs.setMenu(QMenu(self))

        self.action_add_files = self.btn_add_docs.menu().addAction(
            self.tr("btn_add_from_files")
        )
        self.action_add_folder = self.btn_add_docs.menu().addAction(
            self.tr("btn_add_from_folder")
        )

        self.action_add_files.triggered.connect(self.add_documents_from_files)
        self.action_add_folder.triggered.connect(self.add_documents_from_folder)

        # Language toggle button (icon-only)
        self.btn_lang = QPushButton()
        self.btn_lang.setFixedSize(36, 36)
        self.btn_lang.setIconSize(QSize(22, 22))
        self.btn_lang.clicked.connect(self.toggle_language)

        self.icon_flag_uk = QIcon(str(icon_dir / "uk.svg"))
        self.icon_flag_fr = QIcon(str(icon_dir / "fr.svg"))

        # pseudo-only export option (checked by default)
        self.chk_pseudo_only = QCheckBox(self.tr("chk_pseudo_only"))
        self.chk_pseudo_only.setChecked(True)
        self.chk_pseudo_only.setToolTip(self.tr("tip_pseudo_only"))

        # Selection mode controls
        self._selection_mode = False

        self.btn_select_mode = QPushButton(self.tr("btn_select_mode"))
        self.btn_select_mode.clicked.connect(self._toggle_selection_mode)

        self.chk_select_all = QCheckBox(self.tr("chk_select_all"))
        self.chk_select_all.stateChanged.connect(self._toggle_select_all)
        self.chk_select_all.setVisible(False)

        self.btn_remove_selected = QPushButton(self.tr("btn_remove_selected"))
        self.btn_remove_selected.clicked.connect(self._remove_selected)
        self.btn_remove_selected.setVisible(False)

        # Table: one row per file, with IPP + ORDER entry
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(self.tr("table_headers"))
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 520)
        self.table.setColumnWidth(1, 140)
        self.table.setColumnWidth(2, 80)
        self.table.setColumnWidth(3, 180)

        # Status label (kept for compatibility; message banner is primary UX)
        self.status = QLabel(self.tr("status_ready") if self.pseudonymizer else self.tr("status_not_ready"))
        self.status.setStyleSheet("color: #333;")

        top = QGridLayout()
        top.addWidget(QLabel(self.tr("database_label")), 0, 0)
        top.addWidget(self.db_path_edit, 0, 1, 1, 3)
        top.addWidget(self.btn_choose_db, 0, 4)
        top.addWidget(self.btn_init_db, 0, 5)
        top.addWidget(self.btn_theme, 0, 6)
        top.addWidget(self.btn_lang, 0, 7)

        actions = QHBoxLayout()
        actions.addWidget(self.btn_add_docs)
        actions.addWidget(self.btn_select_mode)
        actions.addWidget(self.chk_select_all)
        actions.addWidget(self.btn_remove_selected)
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

        # Apply saved theme + translations last (ensures tooltips/icons and texts align)
        self.set_theme(self.theme)
        self._apply_translations()
        self._set_message("info", self.msg_body.text())

    def tr(self, key: str, **kwargs) -> str:
        pack = I18N.get(self.lang, I18N["en"])
        text = pack.get(key, I18N["en"].get(key, key))
        try:
            return text.format(**kwargs)
        except Exception:
            return text

    def _is_duplicate(self, df_db: pd.DataFrame, ipp: str, document: str) -> bool:
        if df_db.empty:
            return False
        return ((df_db["IPP"] == ipp) & (df_db["DOCUMENT"] == document)).any()

    def set_language(self, lang: str) -> None:
        lang = (lang or "").lower().strip()
        if lang not in ("en", "fr"):
            lang = "en"
        self.lang = lang
        self.settings.setValue("ui/lang", self.lang)
        self._apply_translations()

    def toggle_language(self) -> None:
        self.set_language("fr" if self.lang == "en" else "en")

    def _apply_translations(self) -> None:
        # Window + top label + placeholder
        self.setWindowTitle(self.tr("window_title"))
        self.db_path_edit.setPlaceholderText(self.tr("placeholder_db"))

        # Buttons / checkbox
        self.btn_choose_db.setText(self.tr("btn_select_db"))
        self.btn_init_db.setText(self.tr("btn_create_db"))
        self.btn_add_docs.setText(self.tr("btn_add_documents"))
        self.action_add_files.setText(self.tr("btn_add_from_files"))
        self.action_add_folder.setText(self.tr("btn_add_from_folder"))
        self.btn_commit.setText(self.tr("btn_commit"))
        self.chk_pseudo_only.setText(self.tr("chk_pseudo_only"))
        self.chk_pseudo_only.setToolTip(self.tr("tip_pseudo_only"))

        # Selection mode controls
        self.btn_select_mode.setText(
            self.tr("btn_cancel_select") if self._selection_mode else self.tr("btn_select_mode")
        )
        self.chk_select_all.setText(self.tr("chk_select_all"))
        self.btn_remove_selected.setText(self.tr("btn_remove_selected"))

        # Table headers
        self.table.setHorizontalHeaderLabels(self.tr("table_headers"))

        # Theme tooltips depend on theme
        if getattr(self, "theme", "light") == "dark":
            self.btn_theme.setToolTip(self.tr("tip_theme_light"))
        else:
            self.btn_theme.setToolTip(self.tr("tip_theme_dark"))

        # Flag icon + tooltip depends on current language (show the OTHER language as the action)
        if self.lang == "en":
            self.btn_lang.setIcon(self.icon_flag_fr)
            self.btn_lang.setToolTip(self.tr("tip_lang_to_fr"))
        else:
            self.btn_lang.setIcon(self.icon_flag_uk)
            self.btn_lang.setToolTip(self.tr("tip_lang_to_en"))

        # Ensure the banner title (Info/Warning/Error) is re-translated on language switch.
        # Re-apply current body text and the last severity to force title refresh.
        if not hasattr(self, "_last_msg_level"):
            self._last_msg_level = "info"
        self._set_message(
            self._last_msg_level,
            self.msg_body.text()
            or (self.tr("status_ready") if self.pseudonymizer else self.tr("status_not_ready")),
        )


    def _on_eds_ready(self, pseudo: object) -> None:
        self.pseudonymizer = pseudo  # type: ignore[assignment]
        if hasattr(self, "init_dlg") and self.init_dlg:
            self.init_dlg.close()
        self.btn_commit.setEnabled(True)
        self._set_message("info", self.tr("eds_loaded_ready"))

    def _on_eds_failed(self, err: str) -> None:
        self.pseudonymizer = None
        if hasattr(self, "init_dlg") and self.init_dlg:
            self.init_dlg.close()

        self.btn_commit.setEnabled(False)
        QMessageBox.critical(
            self,
            self.tr("eds_init_failed_title"),
            self.tr("eds_init_failed_body", err=err),
        )
        self._set_message("error", self.tr("status_not_ready"))

    def _loading_eds(self, eds_path):
        # EDS-PSEUDO model initialization (async so UI remains responsive)
        self.pseudonymizer: TextPseudonymizer | None = None
        self.btn_commit.setEnabled(False)  # will be enabled when model is ready

        self.init_dlg = self._make_progress(
            self.tr("progress_start_title"),
            self.tr("progress_loading_eds"),
            0,
        )
        self.init_dlg.setRange(0, 0)  # indeterminate busy animation
        self.init_dlg.setLabelText(self.tr("progress_loading_eds"))
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
            self.btn_theme.setToolTip(self.tr("tip_theme_light"))
            self.setStyleSheet(DARK_QSS + self._message_banner_qss(dark=True))
        else:
            self.btn_theme.setChecked(False)
            self.btn_theme.setIcon(self.icon_sun)
            self.btn_theme.setToolTip(self.tr("tip_theme_dark"))
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
        self._last_msg_level = level

        if level == "error":
            title = self.tr("msg_title_error")
        elif level == "warning":
            title = self.tr("msg_title_warning")
        else:
            title = self.tr("msg_title_info")

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
            self.tr("dlg_select_db_title"),
            start_dir,
            "CSV files (*.csv);;All files (*.*)",
        )
        if path:
            self.db_path_edit.setText(path)
            self._set_message("info", self.tr("selected_db", path=path))
            self._set_last_dir("paths/select_db", path)

    def create_db(self):
        start_dir = self._get_last_dir("paths/create_db")
        default_path = str(Path(start_dir) / "clinical_db.csv")

        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("dlg_create_db_title"),
            default_path,
            "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            init_db(path, columns=DEFAULT_COLUMNS)
            get_or_create_salt_file(path)
            self.db_path_edit.setText(path)
            self._set_message("info", self.tr("created_db", path=path))
            self._set_last_dir("paths/create_db", path)
        except Exception as e:
            self._error(str(e))

    def add_documents_from_folder(self):
        start_dir = self._get_last_dir("paths/documents")

        folder = QFileDialog.getExistingDirectory(
            self,
            self.tr("dlg_select_folder_title"),
            start_dir,
        )

        if not folder:
            return

        folder_path = Path(folder)
        pdf_files = sorted(folder_path.glob("*.pdf"))

        if not pdf_files:
            self._info(self.tr("no_docs_selected"))
            return

        self._set_last_dir("paths/documents", folder)

        existing = {
            self.table.item(r, 0).text().strip()
            for r in range(self.table.rowCount())
            if self.table.item(r, 0)
        }

        added = 0
        for p in pdf_files:
            fp = str(p.resolve())
            if fp in existing:
                continue
            self._add_row(fp)
            added += 1

        self._set_message("info", self.tr("loaded_files_enter_ipp", n=added))


    def add_documents_from_files(self):
        start_dir = self._get_last_dir("paths/documents")
        files, _ = QFileDialog.getOpenFileNames(
            self,
            self.tr("dlg_select_pdf_title"),
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
            self._add_row(file_path=f)
            added += 1

        self._set_message("info", self.tr("loaded_files_enter_ipp", n=added))

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
        df_out.to_csv(dst, index=True)

    def commit(self):
        self.sleep_inhibitor.enable()
        try:
            db_path = self.db_path_edit.text().strip()
            if not db_path:
                self._error(self.tr("please_select_db_first"))
                return

            if self.pseudonymizer is None:
                self._error(self.tr("model_not_initialized"))
                return

            try:
                df_db = load_db(db_path)
            except Exception as e:
                self._error(self.tr("could_not_read_db", err=e))
                return

            try:
                salt = get_or_create_salt_file(db_path)
                self.pseudonymizer.secret_salt = salt
            except Exception as e:
                self._error(self.tr("salt_init_failed", err=e))
                return

            n = self.table.rowCount()
            if n == 0:
                self._error(self.tr("no_docs_selected"))
                return

            # Logs
            errors: list[str] = []
            skipped: list[str] = []

            # -------------------------
            # Phase 1 — Extraction + IPP + duplicate check
            # -------------------------
            extract_dlg = self._make_progress(
                self.tr("progress_commit_title"),
                self.tr("progress_extracting"),
                n,
            )

            candidates: list[dict] = []

            for r in range(n):
                fp = self.table.item(r, 0).text().strip()
                p = Path(fp)

                try:
                    text = self.extractor.pdf_to_text(p)
                    if not text.strip():
                        raise ValueError("Empty extracted text")

                    try:
                        ipp = str(int(extract_IPP_from_document(text)))
                    except Exception:
                        ipp = str(int(extract_IPP_from_path(p)))

                    if self._is_duplicate(df_db, ipp, text):
                        skipped.append(p.name)
                    else:
                        candidates.append(
                            {
                                "path": p,
                                "ipp": ipp,
                                "document": text,
                            }
                        )

                except Exception as e:
                    errors.append(f"[Extraction] {p.name}: {e}")

                extract_dlg.setValue(r + 1)
                extract_dlg.setLabelText(
                    self.tr("progress_extracting_step", done=r + 1, total=n, name=p.name)
                )
                QApplication.processEvents()

            extract_dlg.close()

            if not candidates:
                self._info(
                    f"No document to commit.\n\n"
                    f"Skipped (duplicates): {len(skipped)}\n"
                    f"Errors: {len(errors)}"
                )
                return

            # -------------------------
            # Phase 2 — Pseudonymization + chunked commit
            # -------------------------
            pseudo_dlg = self._make_progress(
                self.tr("progress_commit_title"),
                self.tr("progress_pseudonymizing"),
                len(candidates),
            )

            pending_rows = []
            pending_names: list[str] = []
            committed_files: list[str] = []

            try:
                for i, item in enumerate(candidates, start=1):
                    p = item["path"]

                    try:
                        pseudo = self.pseudonymizer.pseudonymize(
                            item["document"],
                            ipp=item["ipp"],
                        )

                        pending_rows.append(
                            {
                                "IPP": item["ipp"],
                                "SOURCE_FILE": str(p.resolve()),
                                "DOCUMENT": item["document"],
                                "PSEUDO": pseudo,
                                "ORDER": 1,
                            }
                        )
                        pending_names.append(p.name)

                        if len(pending_rows) >= CHUNK_SIZE:
                            self._flush_pending(
                                db_path, pending_rows, pending_names,
                                committed_files, errors,
                            )
                            pending_rows.clear()
                            pending_names.clear()

                            self._set_message(
                                "info",
                                f"Committed: {len(committed_files)} | "
                                f"Skipped: {len(skipped)} | "
                                f"Errors: {len(errors)}"
                            )

                    except Exception as e:
                        errors.append(f"[Pseudonymization] {p.name}: {e}")

                    pseudo_dlg.setValue(i)
                    pseudo_dlg.setLabelText(
                        self.tr(
                            "progress_pseudonymizing_step",
                            done=i,
                            total=len(candidates),
                            name=p.name,
                        )
                    )
                    QApplication.processEvents()

                # Final flush
                if pending_rows:
                    self._flush_pending(
                        db_path, pending_rows, pending_names,
                        committed_files, errors,
                    )

            finally:
                pseudo_dlg.close()

            # -------------------------
            # Pseudo-only copy (if requested)
            # -------------------------
            pseudo_note = ""
            if self.chk_pseudo_only.isChecked():
                try:
                    self._write_pseudo_only_copy(db_path)
                    pseudo_path = self._pseudo_only_path(db_path)
                    pseudo_note = self.tr(
                        "pseudo_only_copy_note", path=str(pseudo_path)
                    )
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        self.tr("pseudo_only_copy_failed_title"),
                        self.tr(
                            "pseudo_only_copy_failed_after_commit",
                            err=e,
                        ),
                    )

            # Write commit log (always; serves as audit trail)
            log_note = ""
            try:
                log_path = self._write_commit_log(
                    db_path, committed_files, skipped, errors,
                )
                if errors:
                    log_note = f"\nCommit log: {log_path}"
            except Exception:
                pass  # Never fail the commit over a log write issue

            # Final report
            summary = (
                f"Commit finished.\n\n"
                f"Committed: {len(committed_files)}\n"
                f"Skipped (duplicates): {len(skipped)}\n"
                f"Errors: {len(errors)}\n"
                f"{pseudo_note}"
                f"{log_note}"
            )

            QMessageBox.information(
                self,
                self.tr("box_info"),
                summary,
            )

            self.table.setRowCount(0)
            self._set_message("info", summary)
        finally:
            self.sleep_inhibitor.disable()

    def _flush_pending(
        self,
        db_path: str,
        pending_rows: list[dict],
        pending_names: list[str],
        committed_files: list[str],
        errors: list[str],
    ) -> None:
        """
        Try to commit a chunk of rows to the DB.
        If the chunk fails, fall back to row-by-row commits so that
        one bad file does not prevent the rest from being saved.
        Modifies *committed_files* and *errors* in-place.
        """
        if not pending_rows:
            return

        try:
            append_rows_locked(db_path, pd.DataFrame(pending_rows))
            committed_files.extend(pending_names)
        except Exception:
            # Chunk failed — try rows individually to salvage what we can
            for row_dict, name in zip(pending_rows, pending_names):
                try:
                    append_rows_locked(db_path, pd.DataFrame([row_dict]))
                    committed_files.append(name)
                except Exception as row_err:
                    errors.append(f"[Commit] {name}: {row_err}")

    def _write_commit_log(
        self,
        db_path: str,
        committed: list[str],
        skipped: list[str],
        errors: list[str],
    ) -> Path:
        """Append a timestamped commit summary to ``<db_stem>_commit_log.txt``."""
        from datetime import datetime

        p = Path(db_path).expanduser().resolve()
        log_path = p.with_name(f"{p.stem}_commit_log.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"{'=' * 60}",
            f"Commit — {timestamp}",
            f"Database: {p}",
            f"{'=' * 60}",
            "",
            f"Committed: {len(committed)} file(s)",
        ]
        for f in committed:
            lines.append(f"  OK   {f}")
        lines.append("")

        lines.append(f"Skipped (duplicates): {len(skipped)} file(s)")
        for f in skipped:
            lines.append(f"  --   {f}")
        lines.append("")

        lines.append(f"Errors: {len(errors)}")
        for e in errors:
            lines.append(f"  ERR  {e}")
        lines.append("")
        lines.append("")

        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        return log_path

    # ------------------------------------------------------------------
    # Selection mode: select / deselect rows, then remove selected
    # ------------------------------------------------------------------

    def _toggle_selection_mode(self) -> None:
        """Enter or leave selection mode."""
        self._selection_mode = not self._selection_mode

        if self._selection_mode:
            # Entering selection mode — add checkboxes to every row's column-0 item
            self.btn_select_mode.setText(self.tr("btn_cancel_select"))
            self.chk_select_all.setVisible(True)
            self.btn_remove_selected.setVisible(True)
            self.chk_select_all.setChecked(True)

            for r in range(self.table.rowCount()):
                it = self.table.item(r, 0)
                if it is not None:
                    it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Checked)
        else:
            # Leaving selection mode — strip checkboxes
            self.btn_select_mode.setText(self.tr("btn_select_mode"))
            self.chk_select_all.setVisible(False)
            self.btn_remove_selected.setVisible(False)

            for r in range(self.table.rowCount()):
                it = self.table.item(r, 0)
                if it is not None:
                    it.setFlags(it.flags() & ~Qt.ItemIsUserCheckable)
                    it.setData(Qt.CheckStateRole, None)

    def _toggle_select_all(self, state: int) -> None:
        """Check or uncheck every row when the 'Select all' checkbox changes."""
        if not self._selection_mode:
            return
        target = Qt.Checked if state == Qt.Checked.value else Qt.Unchecked
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it is not None:
                it.setCheckState(target)

    def _remove_selected(self) -> None:
        """Remove rows whose checkbox is checked, after user confirmation."""
        selected_rows = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it is not None and it.checkState() == Qt.Checked:
                selected_rows.append(r)

        if not selected_rows:
            self._info(self.tr("no_selection"))
            return

        reply = QMessageBox.question(
            self,
            self.tr("dlg_confirm_remove_title"),
            self.tr("dlg_confirm_remove_body", n=len(selected_rows)),
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )

        if reply != QMessageBox.Yes:
            return

        # Remove from bottom to top so indices stay valid
        for r in reversed(selected_rows):
            self.table.removeRow(r)

        # Leave selection mode after removal
        self._toggle_selection_mode()

    # ------------------------------------------------------------------

    def _add_row(self, file_path: str):
        r = self.table.rowCount()
        self.table.insertRow(r)

        fp_item = QTableWidgetItem(file_path)
        fp_item.setFlags(fp_item.flags() ^ Qt.ItemIsEditable)
        self.table.setItem(r, 0, fp_item)

        # If currently in selection mode, give the new row a checkbox too
        if self._selection_mode:
            fp_item.setFlags(fp_item.flags() | Qt.ItemIsUserCheckable)
            fp_item.setCheckState(Qt.Unchecked)

        # IPP (auto, read-only)
        ipp_item = QTableWidgetItem("—")
        ipp_item.setFlags(ipp_item.flags() ^ Qt.ItemIsEditable)
        self.table.setItem(r, 1, ipp_item)

        # ORDER (auto, read-only)
        order_item = QTableWidgetItem("—")
        order_item.setFlags(order_item.flags() ^ Qt.ItemIsEditable)
        self.table.setItem(r, 2, order_item)

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
        QMessageBox.information(self, self.tr("box_info"), message)
        self._set_message("info", message)

    def _error(self, message: str):
        QMessageBox.critical(self, self.tr("box_error"), message)
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