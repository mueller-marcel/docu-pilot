from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from docupilot.recording.session import RecordingSession
from docupilot.ui.formatting import format_ms, run_name

_DEFINITION_TEXT = {"begin": "Begin", "end": "End  "}


class BoundaryDialog(QDialog):
    """
    Dialog for showing and deleting the ground-truth boundaries set in ONE run.

    The boundaries of other runs are neither shown nor changed: a repeat
    annotation must stay blind, otherwise the retest only measures how well
    the first run was copied.
    """

    def __init__(self, boundaries: list[dict], run: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Boundaries set · {run_name(run)}")
        self.setMinimumSize(460, 320)
        self.setModal(True)

        self._all = list(boundaries)
        self._run = run

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        self._header = QLabel()
        self._header.setStyleSheet("font-size:13px; font-weight:600; color:#222;")
        layout.addWidget(self._header)

        self._list = QListWidget()
        self._list.setStyleSheet(
            "QListWidget{border:1px solid #ddd; border-radius:6px; font-size:12px;}"
            "QListWidget::item{padding:6px 10px;}"
            "QListWidget::item:selected{background:#dbeafe; color:#222;}"
        )
        layout.addWidget(self._list)

        delete_btn = QPushButton("Delete selected")
        delete_btn.setStyleSheet(
            "QPushButton{background:#fff;color:#c0392b;border:1px solid #e74c3c;"
            "border-radius:6px;padding:5px 14px;font-size:12px;}"
            "QPushButton:hover{background:#fdf0ef;}"
        )
        delete_btn.clicked.connect(self._delete_selected)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            "QPushButton{background:#fff;color:#333;border:1px solid #ccc;"
            "border-radius:6px;padding:5px 14px;font-size:12px;}"
            "QPushButton:hover{background:#f0f0f0;}"
        )
        close_btn.clicked.connect(self.accept)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addWidget(delete_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._refresh_list()

    def _visible(self) -> list[int]:
        """Indices into the full list of the entries of this run, by time."""
        indices = [i for i, b in enumerate(self._all)
                   if RecordingSession.boundary_run(b) == self._run]
        return sorted(indices, key=lambda i: float(self._all[i].get("t_ms", 0.0)))

    def _refresh_list(self) -> None:
        visible = self._visible()
        self._header.setText(f"{len(visible)} boundary(-ies) · {run_name(self._run)}")
        self._list.clear()
        for n, i in enumerate(visible, start=1):
            b = self._all[i]
            stamp = str(b.get("annotated_at", b.get("created_at_utc", "")))[:19].replace("T", "  ")
            definition = _DEFINITION_TEXT.get(RecordingSession.boundary_definition(b), "?")
            item = QListWidgetItem(
                f"#{n}   {definition}   {format_ms(float(b.get('t_ms', 0.0)))}   —   {stamp} UTC"
            )
            item.setData(Qt.ItemDataRole.UserRole, i)
            self._list.addItem(item)

    def _delete_selected(self) -> None:
        selected = self._list.selectedItems()
        if not selected:
            return
        for idx in sorted({item.data(Qt.ItemDataRole.UserRole) for item in selected}, reverse=True):
            self._all.pop(idx)
        self._refresh_list()

    def get_boundaries(self) -> list[dict]:
        """All boundaries of all runs, with this run's deletions applied."""
        return self._all
