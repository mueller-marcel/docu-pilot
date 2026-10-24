from __future__ import annotations

import math

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from docupilot.ui.widgets.chart_style import (
    AXIS, BACKGROUND, MUTED, TEXT, decimal, paint_placeholder, small_fonts,
)

_LABEL_WIDTH = 160
_VALUE_WIDTH = 170
_ROW_HEIGHT = 38
_TOP_PLAIN = 8
_TOP_BAND = 24          # room for the ±SESOI labels above the plot
_BOTTOM = 46            # tick labels and axis title
_BAND = "#ebebeb"


class BarChartWidget(QWidget):
    """
    One horizontal bar per row with its 95 % confidence interval as whisker,
    against a zero line — the figures of TF1, TF2 and the TF3 gains. With
    `band`, a grey area marks ±band (the SESOI), labelled above the plot as in
    the thesis: a whisker that stays inside it is a gain too small to matter.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[tuple[str, float, float, float, str]] = []
        self._band: float | None = None
        self._axis_title = ""
        self.setMinimumHeight(140)
        self.setStyleSheet(f"background:{BACKGROUND};")

    def set_values(
        self,
        rows: list[tuple[str, float, float, float, str]],
        axis_title: str = "",
        band: float | None = None,
    ) -> None:
        """
        :param rows: (label, value, ci_low, ci_high, hex colour), top to bottom.
        :param axis_title: the x-axis label under the ticks.
        :param band: half-width of the ±band drawn around zero, or None.
        """
        self._rows = rows
        self._axis_title = axis_title
        self._band = band
        top = _TOP_BAND if band is not None else _TOP_PLAIN
        self.setMinimumHeight(top + _ROW_HEIGHT * len(rows) + _BOTTOM)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self._rows:
            paint_placeholder(painter, self.rect())
            return
        painter.fillRect(self.rect(), QColor(BACKGROUND))

        plot_left = _LABEL_WIDTH
        plot_right = max(plot_left + 40, self.width() - _VALUE_WIDTH)
        plot_top = _TOP_BAND if self._band is not None else _TOP_PLAIN
        plot_bottom = plot_top + _ROW_HEIGHT * len(self._rows)

        band = self._band or 0.0
        low = min(0.0, -band, min(r[2] for r in self._rows))
        high = max(0.0, band, max(r[3] for r in self._rows))
        pad = max((high - low) * 0.06, 0.005)
        low, high = low - pad, high + pad

        def x_of(value: float) -> float:
            return plot_left + (plot_right - plot_left) * (value - low) / (high - low)

        small, tiny = small_fonts(self.font())

        if self._band is not None:
            painter.fillRect(QRectF(x_of(-band), plot_top, x_of(band) - x_of(-band),
                                    plot_bottom - plot_top), QColor(_BAND))
            painter.setFont(small)
            for edge, label in ((-band, "−SESOI"), (band, "+SESOI")):
                painter.setPen(QPen(QColor(MUTED), 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(x_of(edge)), plot_top, int(x_of(edge)), plot_bottom)
                painter.setPen(QColor(TEXT))
                painter.drawText(QRectF(x_of(edge) - 40, 2, 80, 18),
                                 int(Qt.AlignmentFlag.AlignHCenter), label)

        # Axes with ticks and title, then the zero line.
        painter.setPen(QPen(QColor(AXIS), 1))
        painter.drawLine(plot_left, plot_bottom, plot_right, plot_bottom)
        painter.drawLine(plot_left, plot_top, plot_left, plot_bottom)
        painter.setFont(tiny)
        step = _tick_step(high - low)
        k = math.ceil(low / step - 1e-9)
        while k * step <= high + 1e-9:
            x = x_of(k * step)
            painter.setPen(QPen(QColor(AXIS), 1))
            painter.drawLine(int(x), plot_bottom, int(x), plot_bottom + 3)
            painter.setPen(QColor(MUTED))
            painter.drawText(QRectF(x - 28, plot_bottom + 4, 56, 14),
                             int(Qt.AlignmentFlag.AlignHCenter), decimal(k * step, 2))
            k += 1
        if self._axis_title:
            painter.setFont(small)
            painter.setPen(QColor(TEXT))
            painter.drawText(QRectF(plot_left, plot_bottom + 20, plot_right - plot_left, 18),
                             int(Qt.AlignmentFlag.AlignHCenter), self._axis_title)
        painter.setPen(QPen(QColor(TEXT), 1))
        painter.drawLine(int(x_of(0.0)), plot_top, int(x_of(0.0)), plot_bottom)

        for index, (name, value, ci_low, ci_high, colour) in enumerate(self._rows):
            top = plot_top + index * _ROW_HEIGHT
            middle = top + _ROW_HEIGHT / 2

            painter.setFont(small)
            painter.setPen(QColor(TEXT))
            painter.drawText(
                QRectF(4, top, _LABEL_WIDTH - 12, _ROW_HEIGHT),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), name,
            )

            bar_left, bar_right = sorted((x_of(0.0), x_of(value)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(colour))
            painter.drawRect(QRectF(bar_left, middle - 8, max(bar_right - bar_left, 1.0), 16))

            # The whisker in black over the bar, so a whisker reaching zero or
            # the band is never hidden behind its bar.
            painter.setPen(QPen(QColor(TEXT), 1.2))
            painter.drawLine(int(x_of(ci_low)), int(middle), int(x_of(ci_high)), int(middle))
            for edge in (ci_low, ci_high):
                painter.drawLine(int(x_of(edge)), int(middle - 6), int(x_of(edge)), int(middle + 6))

            painter.setFont(tiny)
            painter.setPen(QColor(MUTED))
            painter.drawText(
                QRectF(plot_right + 8, top, _VALUE_WIDTH - 12, _ROW_HEIGHT),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                f"{decimal(value, 3)}  [{decimal(ci_low, 3)}; {decimal(ci_high, 3)}]",
            )


def _tick_step(span: float) -> float:
    """A round tick distance giving at most ten ticks over `span`."""
    for step in (0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0):
        if span / step <= 10:
            return step
    return 1.0
