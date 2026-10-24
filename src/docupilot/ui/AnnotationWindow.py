from __future__ import annotations

from PySide6.QtCore import QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

from docupilot.config import GUIDELINE_VERSION
from docupilot.recording.session import RecordingSession
from docupilot.ui.formatting import format_ms
from docupilot.ui.widgets.BoundaryDialog import BoundaryDialog
from docupilot.ui.widgets.EventsPanelWidget import EventsPanelWidget
from docupilot.ui.widgets.FeatureDialog import FeatureDialog


class _MarkerSlider(QSlider):
    """Horizontal slider with coloured marker dots drawn over the groove — the
    input events, so the annotator sees where the user acted while scrubbing."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._markers: list[float] = []

    def set_markers(self, fractions: list[float]) -> None:
        self._markers = [f for f in fractions if 0.0 <= f <= 1.0]
        self.update()

    def paintEvent(self, paint_event) -> None:
        super().paintEvent(paint_event)
        if not self._markers:
            return

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            opt,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#D85A30"))
        painter.setPen(Qt.PenStyle.NoPen)
        cy = groove.center().y()
        for frac in self._markers:
            cx = groove.left() + int(frac * groove.width())
            painter.drawEllipse(cx - 3, cy - 3, 6, 6)
        painter.end()


class AnnotationWindow(QWidget):
    """
    The annotation page shown after a recording stops or a session is opened.

    Orchestrates player, toolbar, bottom bar, EventsPanelWidget and FeatureDialog.
    Ground-truth boundaries live exclusively in session.ground_truth_data; this
    page never shows model output while annotating, so the ground truth cannot
    end up confirming the detector.

    Every boundary belongs to an annotation RUN. Run 1 is the annotation; run 2
    is its blind, time-shifted repetition: while it is selected, nothing of run
    1 is shown (count, list, feature lanes), so the retest measures the
    annotator rather than their memory of run 1.
    """

    back_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._session: RecordingSession | None = None
        self._events: list[dict] = []
        self._duration_ms: float = 0.0
        self._feature_dialog: FeatureDialog | None = None

        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)

        self._build_ui()

        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)
        self._player.mediaStatusChanged.connect(self._on_media_status_changed)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._on_tick)

    def load_session(self, session: RecordingSession) -> None:
        self._session = session
        self._duration_ms = 0.0

        session.load_ground_truth()

        self._events = session.read_events()
        self._events_panel.set_events(self._events)

        self._player.setSource(QUrl.fromLocalFile(str(session.recording_path)))

        self._dir_label.setText(session.session_dir.name)
        self._update_boundary_count()
        self._timer.start()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_toolbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle{background:#ddd; width:1px;}")

        self._video_widget = QVideoWidget()
        self._video_widget.setStyleSheet("background:#111;")
        self._video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._player.setVideoOutput(self._video_widget)
        splitter.addWidget(self._video_widget)

        self._events_panel = EventsPanelWidget()
        self._events_panel.setFixedWidth(280)
        self._events_panel.jumped.connect(self._seek_to)
        splitter.addWidget(self._events_panel)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 1)

        root.addWidget(splitter, stretch=1)
        root.addWidget(self._build_bottom_bar())

    def _build_toolbar(self) -> QWidget:
        toolbar = QWidget()
        toolbar.setFixedHeight(46)
        toolbar.setStyleSheet("background:#f7f7f7; border-bottom:1px solid #ddd;")

        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(10, 0, 12, 0)
        layout.setSpacing(8)

        back_btn = QPushButton("← Back")
        back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        back_btn.setStyleSheet(
            "QPushButton{background:none;border:none;color:#555;font-size:13px;}"
            "QPushButton:hover{color:#111;}"
        )
        back_btn.clicked.connect(self._on_back)

        reload_btn = QPushButton("⟳")
        reload_btn.setToolTip("Reload video")
        reload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reload_btn.setFixedSize(28, 28)
        reload_btn.setStyleSheet(
            "QPushButton{background:none;border:1px solid #ddd;border-radius:5px;"
            "color:#888;font-size:14px;}"
            "QPushButton:hover{border-color:#aaa;color:#333;}"
        )
        reload_btn.clicked.connect(self._reload_video)

        dir_icon = QLabel("📁")
        dir_icon.setStyleSheet("font-size:13px;")
        self._dir_label = QLabel()
        self._dir_label.setStyleSheet("font-family:monospace; font-size:12px; color:#666;")

        dir_container = QWidget()
        dir_layout = QHBoxLayout(dir_container)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        dir_layout.setSpacing(4)
        dir_layout.addWidget(dir_icon)
        dir_layout.addWidget(self._dir_label)

        self._boundary_count_label = QPushButton("0 boundaries set")
        self._boundary_count_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._boundary_count_label.setStyleSheet(
            "QPushButton{background:none;border:none;color:#1a6fc4;"
            "font-size:12px;font-weight:600;}"
            "QPushButton:hover{color:#0a4f9c;text-decoration:underline;}"
        )
        self._boundary_count_label.clicked.connect(self._show_boundary_dialog)

        run_label = QLabel("Run")
        run_label.setStyleSheet("color:#555; font-size:12px;")
        self._run_box = QSpinBox()
        self._run_box.setRange(1, 9)
        self._run_box.setValue(1)
        self._run_box.setToolTip(
            "1 = annotation · 2 = blind repetition (at least 3 weeks later); "
            "boundaries of other runs stay hidden"
        )
        self._run_box.valueChanged.connect(self._on_run_changed)

        guideline_label = QLabel("Guideline")
        guideline_label.setStyleSheet("color:#555; font-size:12px;")
        self._guideline_edit = QLineEdit(GUIDELINE_VERSION)
        self._guideline_edit.setFixedWidth(48)
        self._guideline_edit.setToolTip("Version of the annotation guideline, saved with every boundary")

        self._features_button = QPushButton("♪  Features")
        self._features_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._features_button.setToolTip(
            "Show detected action boundaries per modality on the timeline"
        )
        self._features_button.setStyleSheet(
            "QPushButton{background:#e8f0fe;color:#1a6fc4;border:1px solid #c5d8f6;"
            "border-radius:5px;font-size:12px;font-weight:600;padding:0 12px;}"
            "QPushButton:hover{background:#d0e4ff;border-color:#4da3ff;}"
            "QPushButton:pressed{background:#bdd4f8;}"
        )
        self._features_button.clicked.connect(self._show_feature_dialog)

        layout.addWidget(back_btn)
        layout.addWidget(reload_btn)
        layout.addSpacing(8)
        layout.addWidget(dir_container)
        layout.addStretch()
        layout.addWidget(self._features_button)
        layout.addSpacing(10)
        layout.addWidget(run_label)
        layout.addWidget(self._run_box)
        layout.addSpacing(6)
        layout.addWidget(guideline_label)
        layout.addWidget(self._guideline_edit)
        layout.addSpacing(10)
        layout.addWidget(self._boundary_count_label)
        return toolbar

    def _build_bottom_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(90)
        bar.setStyleSheet("background:#f7f7f7; border-top:1px solid #ddd;")

        layout = QVBoxLayout(bar)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(6)

        self._time_label = QLabel("00:00.000")
        self._time_label.setStyleSheet(
            "color:#555; font-size:12px; font-family:monospace; min-width:62px;"
        )
        self._slider = _MarkerSlider()
        self._slider.setRange(0, 1000)
        self._slider.setStyleSheet(
            "QSlider::groove:horizontal{height:4px;background:#ddd;border-radius:2px;}"
            "QSlider::sub-page:horizontal{background:#4da3ff;border-radius:2px;}"
            "QSlider::handle:horizontal{width:12px;height:12px;margin:-4px 0;"
            "background:#4da3ff;border-radius:6px;border:2px solid #f7f7f7;}"
        )
        self._slider.sliderMoved.connect(self._on_slider_moved)
        self._duration_label = QLabel("00:00.000")
        self._duration_label.setStyleSheet(
            "color:#999; font-size:12px; font-family:monospace; min-width:62px;"
        )

        progress_row = QHBoxLayout()
        progress_row.addWidget(self._time_label)
        progress_row.addWidget(self._slider)
        progress_row.addWidget(self._duration_label)
        layout.addLayout(progress_row)

        btn_start = self._make_transport_btn("⏮", "To start")
        btn_back  = self._make_transport_btn("◀ 5s", "5 s back")
        self._play_button = self._make_transport_btn("▶", "Play / Pause")
        btn_fwd   = self._make_transport_btn("5s ▶", "5 s forward")
        btn_end   = self._make_transport_btn("⏭", "To end")
        self._mute_button = self._make_transport_btn("🔊", "Mute")
        self._mute_button.setCheckable(True)

        btn_start.clicked.connect(lambda: self._seek_to(0))
        btn_back.clicked.connect(lambda: self._seek_relative(-5000))
        self._play_button.clicked.connect(self._toggle_play)
        btn_fwd.clicked.connect(lambda: self._seek_relative(5000))
        btn_end.clicked.connect(lambda: self._seek_to(self._duration_ms))
        self._mute_button.clicked.connect(self._toggle_mute)

        self._boundary_button = QPushButton("✂  Set boundary (begin)")
        self._boundary_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._boundary_button.setFixedHeight(30)
        self._boundary_button.setToolTip(
            "Begin of the first observable action of the new step — "
            "the definition of the annotation guideline"
        )
        self._apply_boundary_idle_style()
        self._boundary_button.clicked.connect(lambda: self._set_boundary("begin"))

        self._end_button = QPushButton("■  Set end")
        self._end_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._end_button.setFixedHeight(30)
        self._end_button.setToolTip(
            "Result of the step visible and lasting — for the sensitivity analysis only"
        )
        self._end_button.setStyleSheet(
            "QPushButton{background:#f5f5f5;color:#555;border:1.5px solid #bbb;"
            "border-radius:5px;font-size:13px;font-weight:600;padding:0 16px;}"
            "QPushButton:hover{background:#eee;}"
            "QPushButton:pressed{background:#e0e0e0;}"
        )
        self._end_button.clicked.connect(lambda: self._set_boundary("end"))

        transport_row = QHBoxLayout()
        transport_row.setSpacing(6)
        for b in (btn_start, btn_back, self._play_button, btn_fwd, btn_end):
            transport_row.addWidget(b)
        transport_row.addSpacing(10)
        transport_row.addWidget(self._mute_button)
        transport_row.addStretch()
        transport_row.addWidget(self._end_button)
        transport_row.addSpacing(6)
        transport_row.addWidget(self._boundary_button)
        layout.addLayout(transport_row)

        return bar

    @staticmethod
    def _make_transport_btn(text: str, tooltip: str) -> QPushButton:
        b = QPushButton(text)
        b.setToolTip(tooltip)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setFixedHeight(28)
        b.setStyleSheet(
            "QPushButton{background:#fff;color:#333;border:1px solid #ccc;"
            "border-radius:5px;font-size:12px;padding:0 10px;}"
            "QPushButton:hover{background:#f0f0f0;border-color:#aaa;}"
            "QPushButton:pressed{background:#e8e8e8;}"
        )
        return b

    def _on_tick(self) -> None:
        pos_ms = self._player.position()
        self._time_label.setText(format_ms(float(pos_ms)))
        if self._duration_ms > 0:
            self._slider.blockSignals(True)
            self._slider.setValue(int(pos_ms / self._duration_ms * 1000))
            self._slider.blockSignals(False)
        self._events_panel.highlight(float(pos_ms))

    def _on_duration_changed(self, duration_ms: int) -> None:
        self._duration_ms = float(duration_ms)
        self._duration_label.setText(format_ms(float(duration_ms)))
        if self._duration_ms > 0 and self._session is not None:
            # Filter the log already in memory rather than reading events.json
            # again for the same content.
            fractions = [
                ev.get("t_ms", 0.0) / self._duration_ms
                for ev in RecordingSession.input_events_of(self._events)
            ]
            self._slider.set_markers(fractions)

    def _on_playback_state_changed(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play_button.setText("⏸" if playing else "▶")

    def _on_media_status_changed(self, status) -> None:
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            self._player.play()
            self._player.pause()

    def _toggle_play(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _toggle_mute(self) -> None:
        """Mute or unmute the session playback."""
        muted = not self._audio.isMuted()
        self._audio.setMuted(muted)
        self._mute_button.setText("🔇" if muted else "🔊")
        self._mute_button.setToolTip("Unmute" if muted else "Mute")

    def _seek_to(self, ms: float) -> None:
        self._player.setPosition(int(ms))

    def _seek_relative(self, delta_ms: int) -> None:
        target = max(0, min(self._player.position() + delta_ms, int(self._duration_ms)))
        self._player.setPosition(target)

    def _on_slider_moved(self, value: int) -> None:
        if self._duration_ms > 0:
            self._player.setPosition(int(value / 1000 * self._duration_ms))
            if self._player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                self._player.play()
                self._player.pause()

    def _reload_video(self) -> None:
        if self._session is not None:
            self._player.setSource(QUrl())
            self._player.setSource(QUrl.fromLocalFile(str(self._session.recording_path)))

    @property
    def _run(self) -> int:
        return int(self._run_box.value())

    def _set_boundary(self, definition: str = "begin") -> None:
        if self._session is None:
            return
        # The MP4 carries real timestamps, so the player position already is the
        # common clock the events and video use — no conversion needed.
        pos_ms = float(self._player.position())
        self._session.add_ground_truth_boundary(
            pos_ms, label=format_ms(pos_ms), definition=definition, run=self._run,
            guideline_version=self._guideline_edit.text().strip(),
        )
        self._update_boundary_count()
        if definition == "begin":
            self._flash_boundary_button()

    def _on_run_changed(self, _run: int) -> None:
        # Run 2 must stay blind: the feature lanes would show run 1's boundaries.
        self._features_button.setEnabled(self._run == 1)
        self._update_boundary_count()

    def _show_boundary_dialog(self) -> None:
        if self._session is None:
            return
        dlg = BoundaryDialog(self._session.ground_truth_data, self._run, parent=self)
        dlg.exec()
        self._session.set_ground_truth_boundaries(dlg.get_boundaries())
        self._update_boundary_count()

    def _update_boundary_count(self) -> None:
        if self._session is None:
            return
        n = self._session.count_boundaries("begin", self._run)
        n_end = self._session.count_boundaries("end", self._run)
        self._boundary_count_label.setText(
            f"Run {self._run}: {n} boundar{'ies' if n != 1 else 'y'} set"
            + (f"  ·  {n_end} end" if n_end else "")
        )

    def _flash_boundary_button(self) -> None:
        self._boundary_button.setStyleSheet(
            "QPushButton{background:#e8f5e9;color:#2e7d32;border:1.5px solid #4caf50;"
            "border-radius:5px;font-size:13px;font-weight:600;padding:0 16px;}"
        )
        QTimer.singleShot(600, self._apply_boundary_idle_style)

    def _apply_boundary_idle_style(self) -> None:
        self._boundary_button.setStyleSheet(
            "QPushButton{background:#fff3ee;color:#D85A30;border:1.5px solid #D85A30;"
            "border-radius:5px;font-size:13px;font-weight:600;padding:0 16px;}"
            "QPushButton:hover{background:#ffe8de;}"
            "QPushButton:pressed{background:#ffd6c4;}"
        )

    def _show_feature_dialog(self) -> None:
        if self._session is None or self._run != 1:
            return
        if self._feature_dialog is not None:
            self._feature_dialog.close()
            self._feature_dialog = None

        self._feature_dialog = FeatureDialog(
            player=self._player,
            session=self._session,
            duration_ms=self._duration_ms,
            parent=self,
        )
        self._feature_dialog.show()

    def _on_back(self) -> None:
        self._player.stop()
        self._timer.stop()
        self.back_requested.emit()
