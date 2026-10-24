# Aufnahme

Modul: {py:mod}`docupilot.recording`.

## Bausteine

`RecorderService`
: Startet und stoppt eine Aufnahme. Legt eine {py:class}`~docupilot.recording.session.RecordingSession`
  an, öffnet den {py:class}`~docupilot.recording.session.EventWriter` und
  startet beide Rekorder. Schreibt die Lebenszyklus-Ereignisse
  `recording_started`, `recording_stopping` und `recording_stopped`.

`AvRecorder`
: Ein einziger `ffmpeg`-Prozess nimmt Bildschirm und Mikrofon in
  `recording.mp4` auf. Der Bildschirm wird mit `mss` 10-mal pro Sekunde
  abgegriffen und als Rohbild an `ffmpeg` übergeben; das Mikrofon liest `ffmpeg`
  selbst (`dshow` unter Windows, `avfoundation` unter macOS).

`InputRecorder`
: Protokolliert über `pynput` Mausklicks (`mouse_click`, gedrückt und
  losgelassen), Scrollen (`mouse_scroll`) sowie Tastendrücke und -freigaben
  (`key_press`, `key_release`).

`RecordingSession`
: Eine Aufnahme: Dateipfade, Metadaten, die gemeinsame Uhr und die
  Referenzannotation. Sie ist die einzige Stelle, die `ground_truth.json` liest
  und schreibt.

`EventWriter`
: Hängt Ereignisse threadsicher an ein JSON-Array an, eines pro Zeile.

## Eine Uhr für alle Datenströme

Alle Zeitstempel in `events.json` stehen auf der Uhr des Videos. Der
`AvRecorder` setzt den Ursprung der Session (`RecordingSession.arm`) in dem
Moment, in dem `ffmpeg` das erste Bild angenommen hat – genau dieses Bild ist
Zeit 0 im MP4. Jedes spätere Ereignis erhält `t_ms` relativ zu diesem Ursprung
(`session_time_ms`). Damit liegen Video, Ton und Eingaben ohne nachträgliche
Synchronisation auf einer Zeitachse; die Bildzeiten liest die Video-Modalität
später mit `ffprobe` direkt aus dem Container.

```{important}
Der Mauszeiger wird **nicht** in die Bilder gezeichnet. Er würde den
Ereignisstrom in die Video-Modalität tragen; deren Unabhängigkeit ist
Voraussetzung für die Shapley-Zerlegung. Das Bildmodell bekommt deshalb nie
einen Zeiger zu sehen.
```

## Session-Verzeichnis

Jede Aufnahme liegt in einem eigenen Verzeichnis. Die Eingaben
(`recording.mp4`, `events.json`, `ground_truth.json`) werden nie automatisch
verändert; alles andere in diesem Verzeichnis ist abgeleitet und kann gelöscht
werden. Aufbau und Formate: {doc}`../reference/data-formats`.

Eine gespeicherte Session wird mit
{py:meth}`RecordingSession.from_directory <docupilot.recording.session.RecordingSession.from_directory>`
geöffnet; Bildschirm und Mikrofon sind dann Platzhalter, weil keine Hardware
dahintersteht.
