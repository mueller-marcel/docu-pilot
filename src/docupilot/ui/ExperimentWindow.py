"""
The window that runs the factorial experiment over a corpus and shows the result.

A shell, not a calculator: every number comes from the evaluation package, which
stays runnable without any of this. The sections it shows are the report's own
(`report.sections`), so window and PDF cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QFont
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from docupilot.config import Settings, load_settings, save_settings
from docupilot.evaluation import artifacts, corpus, experiment
from docupilot.evaluation import report as rep
from docupilot.ui.formatting import format_bytes
from docupilot.ui.report_view import ReportCharts, apply_report
from docupilot.ui.ReportPdfWriter import write_report_pdf

_BACKGROUND = "#ffffff"

# Wait-spinner frames (Braille); Segoe UI on Windows renders these.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

_BUTTON_STYLE = (
    "QPushButton{background:#fff;color:#333;border:1px solid #ccc;"
    "border-radius:6px;padding:6px 16px;font-size:12px;}"
    "QPushButton:hover{background:#f0f0f0;}"
    "QPushButton:disabled{background:#f0f0f0;color:#aaa;border-color:#e0e0e0;}"
)

_NOTE_STYLE = ("color:#222; font-size:11px; padding:4px 6px; "
               "font-family:Consolas,'Courier New',monospace;")

_DELETION_TITLE = "Delete generated files"

_DELETION_CHOICES: tuple[tuple[str, frozenset[artifacts.ArtifactKind]], ...] = (
    ("Delete evidence curves and activity scans (.npz)…",
     frozenset({artifacts.ArtifactKind.LANES})),
    ("Delete Claude caches (.json)…",
     frozenset({artifacts.ArtifactKind.MODEL_CACHES})),
    ("Delete all generated files…",
     frozenset(artifacts.ArtifactKind)),
)


class _ExperimentWorker(QObject):
    """Runs the whole evaluation off the UI thread."""

    progress = Signal(str, int, int)
    failed = Signal(str)
    done = Signal(object)

    def __init__(self, directories: list[Path], settings: Settings) -> None:
        """:param settings: the run's settings."""
        super().__init__()
        self._directories = directories
        self._settings = settings
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _progress(self, phase: str):
        return lambda label, i, total: self.progress.emit(f"{phase}: {label}", i, total)

    def _is_cancelled(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        try:
            settings = self._settings
            data = experiment.load(
                self._directories, settings,
                on_progress=self._progress("Extraction"),
                is_cancelled=self._is_cancelled,
            )
            if self._cancelled or not data:
                self.failed.emit(
                    "Cancelled." if self._cancelled
                    else "No session found that can be evaluated."
                )
                return

            settings = experiment.measure_sesoi(data, settings)
            primary = experiment.run(data, settings,
                                     on_progress=self._progress("Primary run, fold"),
                                     is_cancelled=self._is_cancelled)
            if self._cancelled:
                self.failed.emit("Cancelled.")
                return
            self.done.emit(rep.analyse(data, primary))
        except Exception as exc:                      # noqa: BLE001 — shown, not hidden
            self.failed.emit(str(exc))


class ExperimentWindow(QDialog):
    """Runs the factorial experiment over a corpus and shows what it found."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Evaluation · Information contribution of the modalities")
        self.setMinimumSize(900, 700)
        self.resize(1180, 950)
        self.setModal(False)
        self.setStyleSheet(f"background:{_BACKGROUND};")

        self._directories: list[Path] = []
        self._corpus_root: Path | None = None
        self._settings_path: Path | None = None
        self._settings = Settings()
        self._can_run = False
        self._result: rep.Report | None = None
        self._thread: QThread | None = None
        self._worker: _ExperimentWorker | None = None
        self._deletion_actions = self._create_deletion_actions()

        self._build_ui()
        self._set_deletion_enabled(False)

    @property
    def deletion_actions(self) -> list[QAction]:
        """
        The actions that delete the loaded corpus' generated files.

        The main window puts these very objects into its menu bar, so the menu
        and this window are always enabled and disabled together.
        """
        return list(self._deletion_actions)

    def _button(self, text: str, slot, enabled: bool = True) -> QPushButton:
        button = QPushButton(text)
        button.setStyleSheet(_BUTTON_STYLE)
        button.setEnabled(enabled)
        button.clicked.connect(slot)
        return button

    def _create_deletion_actions(self) -> list[QAction]:
        actions = []
        for text, kinds in _DELETION_CHOICES:
            action = QAction(text, self)
            # `triggered` passes `checked`; the default argument binds this loop's kinds.
            action.triggered.connect(lambda _checked=False, k=kinds: self._delete_artifacts(k))
            actions.append(action)
        return actions

    def _deletion_button(self) -> QPushButton:
        """A button whose drop-down menu offers the deletion actions."""
        button = QPushButton(_DELETION_TITLE)
        button.setStyleSheet(_BUTTON_STYLE)
        menu = QMenu(button)
        menu.addActions(self._deletion_actions)
        button.setMenu(menu)
        return button

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        self._choose_button = self._button("Choose corpus …", self._on_choose_corpus)
        self._run_button = self._button("Start evaluation", self._on_run, enabled=False)
        self._cancel_button = self._button("Cancel", self._on_cancel, enabled=False)
        self._export_button = self._button("Export CSV …", self._on_export, enabled=False)
        self._pdf_button = self._button("Report (PDF) …", self._on_export_pdf, enabled=False)
        self._delete_button = self._deletion_button()
        self._settings_button = self._button(
            "Settings …", self._on_open_settings, enabled=False)

        top = QHBoxLayout()
        for button in (self._choose_button, self._run_button, self._cancel_button,
                       self._export_button, self._pdf_button, self._settings_button,
                       self._delete_button):
            top.addWidget(button)
        top.addStretch()
        root.addLayout(top)

        self._corpus_label = QLabel("No corpus chosen.")
        self._corpus_label.setWordWrap(True)
        self._corpus_label.setStyleSheet("color:#555; font-size:11px;")
        root.addWidget(self._corpus_label)

        self._settings_label = QLabel("")
        self._settings_label.setWordWrap(True)
        self._settings_label.setStyleSheet("color:#555; font-size:11px;")
        root.addWidget(self._settings_label)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["Session", "Begin (run 1)", "End", "Retest (run 2)", "Video cache", "Audio cache"])
        self._table.setMaximumHeight(150)
        self._table.setStyleSheet(
            "QTableWidget{background:#fff;color:#333;gridline-color:#ddd;"
            "font-size:11px;border:1px solid #ccc;}"
            "QHeaderView::section{background:#f0f0f0;color:#555;border:0;padding:4px;}"
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        root.addWidget(self._table)

        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(6)
        self._progress.setStyleSheet(
            "QProgressBar{background:#e5e7eb;border:0;border-radius:3px;}"
            "QProgressBar::chunk{background:#059669;border-radius:3px;}")
        root.addWidget(self._progress)

        self._spinner = QLabel("")
        self._spinner.setStyleSheet("color:#059669; font-size:15px; font-weight:bold;")
        self._status = QLabel("Bereit.")
        self._status.setStyleSheet("color:#555; font-size:11px;")
        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        status_row.addWidget(self._spinner)
        status_row.addWidget(self._status)
        status_row.addStretch()
        root.addLayout(status_row)

        self._spinner_index = 0
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(110)
        self._spinner_timer.timeout.connect(self._tick_spinner)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background:{_BACKGROUND}; border:0;")
        content = QWidget()
        content.setStyleSheet(f"background:{_BACKGROUND};")
        self._sections = QVBoxLayout(content)
        self._sections.setContentsMargins(0, 0, 0, 0)
        self._sections.setSpacing(4)

        self._charts = ReportCharts.create()
        self._placeholder = QLabel("No evaluation yet.")
        self._placeholder.setStyleSheet("color:#777; font-size:12px; padding:12px 6px;")
        self._sections.addWidget(self._placeholder)
        self._sections.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll, stretch=1)

    @staticmethod
    def _heading(title: str, subtitle: str) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 10, 4, 2)
        layout.setSpacing(1)

        heading = QLabel(title)
        font = QFont(heading.font())
        font.setBold(True)
        font.setPointSize(10)
        heading.setFont(font)
        heading.setStyleSheet("color:#222;")

        hint = QLabel(subtitle)
        hint.setStyleSheet("color:#777; font-size:10px;")
        hint.setWordWrap(True)

        layout.addWidget(heading)
        layout.addWidget(hint)
        return box

    def _on_choose_corpus(self) -> None:
        """Pick a folder of session directories and report what is usable."""
        chosen = QFileDialog.getExistingDirectory(self, "Choose corpus directory")
        if chosen:
            self.open_corpus(Path(chosen))

    def open_corpus(self, root: Path) -> None:
        """
        Load a corpus directory without evaluating it.

        The evaluation starts only from "Start evaluation", so choosing a corpus
        never launches the extraction by itself. Refused while a run is in
        progress: that run keeps its corpus.
        """
        if self._is_running():
            self._status.setText(
                "Evaluation running — the corpus can only be changed afterwards."
            )
            return
        self._load_corpus(root)

    def _load_corpus(self, root: Path) -> None:
        """Scan a corpus directory, read its run settings and fill the table."""
        self._corpus_root = root
        scanned = corpus.scan(root)
        self._directories = scanned.usable
        self._settings_path = scanned.settings_path

        self._table.setRowCount(len(scanned.sessions))
        for row, info in enumerate(scanned.sessions):
            for column, text in enumerate((
                info.name,
                str(info.n_begin) if info.annotated else "— no 'begin' GT —",
                str(info.n_end) if info.n_end else "–",
                str(info.n_retest) if info.n_retest else "–",
                "yes" if info.has_video_cache else "no",
                "yes" if info.has_audio_cache else "no",
            )):
                item = QTableWidgetItem(text)
                if not info.annotated:
                    item.setForeground(Qt.GlobalColor.red)
                self._table.setItem(row, column, item)

        self._corpus_label.setText(corpus.describe(scanned))
        self._can_run = scanned.can_evaluate and self._reload_settings()
        self._run_button.setEnabled(self._can_run)
        self._settings_button.setEnabled(True)
        self._set_deletion_enabled(True)

    def _reload_settings(self) -> bool:
        """Read the corpus' settings file; an invalid file blocks the run."""
        path = self._settings_path
        if path is None:
            return False
        try:
            self._settings = load_settings(path)
        except (ValueError, TypeError) as exc:
            self._settings_label.setText(f"⚠ {path.name} invalid: {exc}")
            return False
        s = self._settings
        self._settings_label.setText(
            f"Settings ({path.name}{'' if path.exists() else ', defaults'}): "
            f"{s.decider} · candidates {s.candidate_pool} · definition {s.boundary_definition} · "
            f"τ = {s.tau_s:g} s, SESOI = 1 boundary · intervals 95 %"
        )
        return True

    def _on_open_settings(self) -> None:
        """Write the defaults when there is no file yet, then open it for editing."""
        path = self._settings_path
        if path is None:
            return
        if not path.exists():
            save_settings(self._settings, path)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        self._status.setText(
            f"{path.name} opened — after saving, choose the corpus again to load it."
        )

    def _on_run(self) -> None:
        self._run_button.setEnabled(False)
        self._choose_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._set_deletion_enabled(False)
        self._progress.setRange(0, 0)
        self._status.setText("Starte …")
        self._start_spinner()

        self._thread = QThread(self)
        self._worker = _ExperimentWorker(self._directories, self._settings)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.done.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)

        self._thread.start()

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self._status.setText("Cancellation requested — finishing the current step …")

    def _on_progress(self, label: str, done: int, total: int) -> None:
        self._progress.setRange(0, total)
        self._progress.setValue(done)
        self._status.setText(f"{label}  ({done}/{total})")

    def _start_spinner(self) -> None:
        self._spinner_index = 0
        self._spinner.setText(_SPINNER_FRAMES[0])
        self._spinner_timer.start()

    def _stop_spinner(self) -> None:
        self._spinner_timer.stop()
        self._spinner.setText("")

    def _tick_spinner(self) -> None:
        self._spinner_index = (self._spinner_index + 1) % len(_SPINNER_FRAMES)
        self._spinner.setText(_SPINNER_FRAMES[self._spinner_index])

    def _on_failed(self, message: str) -> None:
        self._stop_spinner()
        self._reset_buttons()
        self._status.setText(f"Failed: {message.splitlines()[0]}")
        QMessageBox.warning(self, "Evaluation failed", message)

    def _on_done(self, result: rep.Report) -> None:
        self._stop_spinner()
        self._result = result
        self._reset_buttons()
        self._export_button.setEnabled(True)
        self._pdf_button.setEnabled(True)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._status.setText(rep.status_line(result))
        self._render(result)

    def _reset_buttons(self) -> None:
        self._run_button.setEnabled(self._can_run)
        self._choose_button.setEnabled(True)
        self._cancel_button.setEnabled(False)
        self._set_deletion_enabled(self._corpus_root is not None)

    def _is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def _set_deletion_enabled(self, enabled: bool) -> None:
        """Deleting needs a loaded corpus and no run that could be writing to it."""
        self._delete_button.setEnabled(enabled)
        for action in self._deletion_actions:
            action.setEnabled(enabled)

    def _delete_artifacts(self, kinds: frozenset[artifacts.ArtifactKind]) -> None:
        """
        Delete the loaded corpus' generated files of the given kinds, after asking.

        Recordings, event logs and ground truth are never touched (see
        `artifacts.delete`). The corpus is rescanned afterwards, so the cache
        columns of the table show what is left.
        """
        root = self._corpus_root
        if root is None or self._is_running():
            return
        paths = artifacts.find(root, kinds)
        if not paths:
            QMessageBox.information(
                self, _DELETION_TITLE, "The corpus contains no generated files of this kind."
            )
            return
        if not self._confirm_deletion(root, paths, kinds):
            return
        result = artifacts.delete(paths)
        self._load_corpus(root)
        self._report_deletion(result)

    def _confirm_deletion(
        self, root: Path, paths: list[Path], kinds: frozenset[artifacts.ArtifactKind]
    ) -> bool:
        """Ask before deleting; the full file list is one click away in the details."""
        sessions = len({path.parent for path in paths})
        text = (
            f"Delete {len(paths)} files ({format_bytes(artifacts.total_size(paths))}) "
            f"in {sessions} session(s)?\n\n"
            "Recording, events.json and ground truth stay unchanged. Deleted "
            "files are regenerated the next time they are opened or evaluated."
        )
        if artifacts.ArtifactKind.MODEL_CACHES in kinds:
            text += "\nDeleted Claude caches trigger new, paid model calls in the process."

        box = QMessageBox(
            QMessageBox.Icon.Warning,
            _DELETION_TITLE,
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            self,
        )
        box.setDefaultButton(QMessageBox.StandardButton.No)
        box.setDetailedText("\n".join(str(path.relative_to(root)) for path in paths))
        return box.exec() == QMessageBox.StandardButton.Yes

    def _report_deletion(self, result: artifacts.DeletionResult) -> None:
        self._status.setText(f"{len(result.deleted)} generated files deleted.")
        if result.failed:
            QMessageBox.warning(
                self,
                _DELETION_TITLE,
                "Not all files could be deleted:\n\n"
                + "\n".join(f"{path}: {reason}" for path, reason in result.failed.items()),
            )

    def _clear_sections(self) -> None:
        """Remove everything but the chart widgets, which are reused."""
        while self._sections.count():
            item = self._sections.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue
            if widget in self._charts.widgets():
                widget.setParent(None)
            else:
                widget.deleteLater()

    def _render(self, result: rep.Report) -> None:
        apply_report(self._charts, result)
        self._clear_sections()
        for block in rep.sections(result):
            self._sections.addWidget(self._heading(block.title, block.hint))
            if block.chart is not None:
                self._sections.addWidget(self._charts.by_name(block.chart))
            if block.text is not None:
                note = QLabel(block.text)
                note.setStyleSheet(_NOTE_STYLE)
                note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                self._sections.addWidget(note)
        self._sections.addStretch()

    def _on_export_pdf(self) -> None:
        """Write a paginated PDF report of the current result: metrics + charts."""
        if self._result is None:
            return
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save report", "evaluation.pdf", "PDF (*.pdf)")
        if not chosen:
            return
        if not chosen.lower().endswith(".pdf"):
            chosen += ".pdf"
        try:
            write_report_pdf(Path(chosen), self._result, self._corpus_root)
        except Exception as exc:                       # noqa: BLE001 — shown, not hidden
            QMessageBox.warning(self, "PDF export failed", str(exc))
            return
        self._status.setText(f"Report exported to {chosen}")

    def _on_export(self) -> None:
        """Write the tidy tables of every design run."""
        if self._result is None:
            return
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save results table", "results.csv", "CSV (*.csv)")
        if not chosen:
            return
        r = self._result
        experiment.write_csv(r.rows, Path(chosen))
        self._status.setText(f"Exported to {chosen}")

    def closeEvent(self, event) -> None:
        """Stop a running evaluation; cached model verdicts survive."""
        if self._worker is not None:
            self._worker.cancel()
        if self._is_running():
            self._thread.quit()
            self._thread.wait(120_000)
        super().closeEvent(event)
