# Datenformate

## Korpus

Ein Korpus ist ein Verzeichnis mit Session-Verzeichnissen. Eine Session zählt
dazu, wenn ihr Verzeichnis `recording.mp4` enthält
({py:func}`~docupilot.evaluation.corpus.session_directories`). Optional liegt im
Korpus `docupilot_settings.json` mit den Einstellungen eines Laufs
({doc}`configuration`).

```text
korpus/
├── docupilot_settings.json        optional
├── session_01_…/
│   ├── recording.mp4              Eingabe
│   ├── events.json                Eingabe
│   ├── ground_truth.json          Eingabe (Annotation)
│   ├── video_ssim.npz             abgeleitet: SSIM-Scan
│   ├── audio_raw.json             abgeleitet: Transkript + Satzurteile
│   ├── gui_vlm_cache.json         abgeleitet: Urteile des Bildmodells
│   ├── audio_llm_cache.json       abgeleitet: Urteile des Sprachmodells
│   └── <modalität>_evidence.npz   abgeleitet: Kurve für den Merkmalsdialog
└── session_02_…/
```

**Eingaben** werden nie automatisch verändert oder gelöscht. **Abgeleitete
Dateien** entstehen beim nächsten Öffnen oder Auswerten neu und lassen sich über
das Menü **Auswertung** gezielt löschen
({py:mod}`~docupilot.evaluation.artifacts`):

| Gruppe | Dateien | Folge des Löschens |
|---|---|---|
| Kurven | `*_evidence.npz`, `video_ssim.npz` | Neuberechnung ohne Modellaufrufe |
| Modell-Caches | `gui_vlm_cache.json`, `audio_llm_cache.json`, `audio_raw.json` | neue Modellaufrufe beim nächsten Lauf; Whisper transkribiert neu |

Andere Dateien im Session-Verzeichnis (etwa aus früheren Versionen) liest die
Pipeline nicht.

## `recording.mp4`

H.264-Video (`libx264`, yuv420p) mit 10 Bildern pro Sekunde und AAC-Tonspur vom Mikrofon. Zeit 0 ist das erste
Bild; dieselbe Uhr gilt für `events.json`.

## `events.json`

Ein JSON-Array, ein Ereignis pro Zeile, jedes mit `type` und `t_ms`
(Millisekunden seit dem ersten Videobild).

| `type` | weitere Felder |
|---|---|
| `recording_started` | `timestamp_utc`, `metadata` (Session-ID, Pfade, Bildschirm, Mikrofon) |
| `av_started` | `fps`, `width`, `height` |
| `input_started` | – |
| `mouse_click` | `x`, `y`, `button` (`Button.left`, …), `pressed` |
| `mouse_scroll` | `x`, `y`, `dx`, `dy` |
| `key_press`, `key_release` | `key` – Zeichen oder `Key.<name>` |
| `recording_stopping`, `av_stopped`, `input_stopped`, `recording_stopped` | – |

```json
{"type": "mouse_click", "x": 1087, "y": 1417, "button": "Button.left", "pressed": true, "t_ms": 1507.704}
```

## `ground_truth.json`

Ein JSON-Array von Grenzen:

| Feld | Bedeutung |
|---|---|
| `t_ms` | Zeitpunkt der Grenze in Millisekunden |
| `label` | Anzeigename, z. B. `"00:09.600"` |
| `definition` | `"begin"` (primär) oder `"end"` |
| `run` | Annotationslauf: 1 = Referenz, 2, 3, … = Wiederholungen |
| `annotated_at` | Zeitpunkt der Annotation (UTC, ISO 8601) |
| `guideline_version` | Version des Annotationsleitfadens |

Weitere Felder werden mitgespeichert und ignoriert. Einträge älterer Versionen
ohne `definition` werden über ein Feld `kind` gelesen (`"start"` → `begin`,
sonst `end`); ohne `run` gehören sie zu Lauf 1.

```json
{"t_ms": 9600.0, "label": "00:09.600", "definition": "begin", "run": 1,
 "annotated_at": "2026-10-02T11:52:41+00:00", "guideline_version": "1"}
```

## Zwischenspeicher

Alle abgeleiteten Dateien tragen einen Schlüssel aus allem, wovon sie abhängen;
ein abweichender Schlüssel gilt als Fehlschlag.

`video_ssim.npz`
: `key`, `times_s` (Bildzeiten aus `ffprobe`), `sim` (kleinste Kachel-SSIM je
  Bild zum Vorgänger). Schlüssel: Format, Kachelraster, SSIM-Parameter,
  Bildbreite.

`audio_raw.json`
: `{"key": …, "payload": {"duration_s", "model", "retrieved_at", "sentences": [{"start", "end", "text", "category", "p"}]}}`.
  Schlüssel: Whisper-Modell, Ausrichtung, spaCy-Modelle, Sprachmodell,
  Prompt-Version.

`gui_vlm_cache.json`, `audio_llm_cache.json`
: Abbildung Cache-Schlüssel → Urteil (Kategorie, Wahrscheinlichkeit,
  Begründung) mit Modell und Abrufzeitpunkt. Schlüssel: Hash der Frage
  (Kompositbild bzw. alle Sätze) + Modell + Prompt-Version.

`<modalität>_evidence.npz`
: eine gezeichnete Kurve für die Anzeige, geschlüsselt über Kurvenformat,
  Rohergebnis-Schlüssel, Grenzdefinition und Darstellungsparameter.

## Ergebnistabelle (CSV-Export)

Eine Zeile je (Session, Startwert, Koalition):

| Spalte | Bedeutung |
|---|---|
| `session`, `seed` | Testsession und Startwert |
| `subset`, `n_modalities` | Koalition (`audio+events`, `{}` für die leere) und ihre Größe |
| `tau_s` | Toleranz der Bewertung |
| `threshold` | gewählte Entscheidungsschwelle (`nan` bei leerer Koalition und Regel-Entscheider) |
| `audio_peak_offset_s`, `video_theta`, `video_min_rest_s` | im Fold kalibrierte Parameter |
| `n_predicted`, `tp`, `fp`, `fn` | Zahl der Vorhersagen und Zuordnung |
| `precision`, `recall`, `f1` | Kennzahlen der Session |
