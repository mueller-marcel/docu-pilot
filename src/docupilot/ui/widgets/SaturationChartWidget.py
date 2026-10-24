from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from docupilot.ui.widgets.chart_style import (
    AXIS, BACKGROUND, MUTED, TEXT, decimal, paint_placeholder, small_fonts,
)

_LINE = "#7c7cf2"
_GAIN = "#4b5563"

_MARGIN_LEFT = 70
_MARGIN_RIGHT = 40
_MARGIN_TOP = 24
_MARGIN_BOTTOM = 48


class SaturationChartWidget(QWidget):
    """
    The saturation curve (TF3, Abb. 4.5): mean F1 of all combinations with k
    modalities against k = 0 … n, with the gain from one point to the next
    written between them — the shape and the step sizes in one picture.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._curve: dict[int, float] = {}
        self.setMinimumHeight(260)
        self.setStyleSheet(f"background:{BACKGROUND};")

    def set_values(self, curve: dict[int, float]) -> None:
        """:param curve: mean F1 per number of modalities, from 0."""
        self._curve = curve
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self._curve:
            paint_placeholder(painter, self.rect())
            return
        painter.fillRect(self.rect(), QColor(BACKGROUND))

        left, top = _MARGIN_LEFT, _MARGIN_TOP
        right = self.width() - _MARGIN_RIGHT
        bottom = self.height() - _MARGIN_BOTTOM
        counts = sorted(self._curve)
        steps = max(len(counts) - 1, 1)

        # The y-axis runs from 0 to the next round value above the curve, so
        # small F1 values are not squeezed against the bottom of a 0…1 scale.
        tick = _tick_step(max(self._curve.values()))
        y_max = tick * math.ceil(max(self._curve.values()) * 1.15 / tick)

        def point_of(k: int) -> QPointF:
            inset = 36             # the outer points keep their labels clear of the axes
            x = left + inset + (right - left - 2 * inset) * counts.index(k) / steps
            return QPointF(x, bottom - (bottom - top) * self._curve[k] / y_max)

        small, tiny = small_fonts(self.font())

        painter.setPen(QPen(QColor(AXIS), 1))
        painter.drawLine(left, top, left, bottom)
        painter.drawLine(left, bottom, right, bottom)
        painter.setFont(tiny)
        value = 0.0
        while value <= y_max + 1e-9:
            y = bottom - (bottom - top) * value / y_max
            painter.setPen(QPen(QColor(AXIS), 1, Qt.PenStyle.DotLine))
            painter.drawLine(left, int(y), right, int(y))
            painter.setPen(QColor(MUTED))
            painter.drawText(QRectF(left - 44, y - 8, 38, 16),
                             int(Qt.AlignmentFlag.AlignRight), decimal(value, 2))
            value += tick

        painter.setFont(small)
        painter.setPen(QColor(TEXT))
        painter.save()
        painter.translate(14, (top + bottom) / 2)
        painter.rotate(-90)
        painter.drawText(QRectF(-80, -9, 160, 18), int(Qt.AlignmentFlag.AlignHCenter),
                         "mittlerer F1-Wert")
        painter.restore()
        painter.drawText(QRectF(left, bottom + 26, right - left, 18),
                         int(Qt.AlignmentFlag.AlignHCenter), "Anzahl Modalitäten")

        painter.setPen(QPen(QColor(_LINE), 2))
        for previous, current in zip(counts, counts[1:]):
            painter.drawLine(point_of(previous), point_of(current))

        # The gain of each step, written beside the middle of its segment.
        painter.setFont(tiny)
        painter.setPen(QColor(_GAIN))
        for previous, current in zip(counts, counts[1:]):
            a, b = point_of(previous), point_of(current)
            gain = self._curve[current] - self._curve[previous]
            painter.drawText(QRectF((a.x() + b.x()) / 2 - 30, (a.y() + b.y()) / 2 + 4, 60, 14),
                             int(Qt.AlignmentFlag.AlignHCenter),
                             ("+" if gain >= 0 else "") + decimal(gain, 3))

        for k in counts:
            centre = point_of(k)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(_LINE))
            painter.drawEllipse(centre, 4.5, 4.5)
            painter.setFont(small)
            painter.setPen(QColor(TEXT))
            painter.drawText(QRectF(centre.x() - 34, centre.y() - 22, 68, 16),
                             int(Qt.AlignmentFlag.AlignHCenter), decimal(self._curve[k], 3))
            painter.setFont(tiny)
            painter.setPen(QColor(MUTED))
            painter.drawText(QRectF(centre.x() - 20, bottom + 6, 40, 14),
                             int(Qt.AlignmentFlag.AlignHCenter), str(k))


def _tick_step(top_value: float) -> float:
    """A round grid distance giving four to eight lines up to `top_value`."""
    for step in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25):
        if top_value * 1.15 / step <= 8:
            return step
    return 0.25
