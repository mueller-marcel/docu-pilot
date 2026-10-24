# Recording

Module: {py:mod}`docupilot.recording`.

## Building blocks

`RecorderService`
: Starts and stops a recording. Creates a {py:class}`~docupilot.recording.session.RecordingSession`,
  opens the {py:class}`~docupilot.recording.session.EventWriter` and
  starts both recorders. Writes the life-cycle events
  `recording_started`, `recording_stopping` and `recording_stopped`.

`AvRecorder`
: A single `ffmpeg` process records screen and microphone into
  `recording.mp4`. The screen is grabbed with `mss` 10 times per second
  and passed to `ffmpeg` as a raw frame; `ffmpeg` reads the microphone
  itself (`dshow` on Windows, `avfoundation` on macOS).

`InputRecorder`
: Logs via `pynput` mouse clicks (`mouse_click`, pressed and
  released), scrolling (`mouse_scroll`) as well as key presses and releases
  (`key_press`, `key_release`).

`RecordingSession`
: One recording: file paths, metadata, the shared clock and the
  reference annotation. It is the only place that reads and writes
  `ground_truth.json`.

`EventWriter`
: Appends events thread-safely to a JSON array, one per line.

## One clock for all data streams

All timestamps in `events.json` are on the video clock. The
`AvRecorder` sets the session origin (`RecordingSession.arm`) at the
moment `ffmpeg` has accepted the first frame – exactly this frame is
time 0 in the MP4. Every later event receives `t_ms` relative to this origin
(`session_time_ms`). Video, audio and input thus lie on one time axis without
subsequent synchronisation; the video modality later reads the frame times
directly from the container with `ffprobe`.

```{important}
The mouse pointer is **not** drawn into the frames. It would carry the
event stream into the video modality, whose independence is a
prerequisite for the Shapley decomposition. The vision model therefore never
gets to see a pointer.
```

## Session directory

Each recording lives in its own directory. The inputs
(`recording.mp4`, `events.json`, `ground_truth.json`) are never modified
automatically; everything else in this directory is derived and can be
deleted. Structure and formats: {doc}`../reference/data-formats`.

A saved session is opened with
{py:meth}`RecordingSession.from_directory <docupilot.recording.session.RecordingSession.from_directory>`;
screen and microphone are then placeholders, because there is no hardware
behind them.
