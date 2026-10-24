# Einstieg

## Voraussetzungen

| Komponente | Zweck |
|---|---|
| Python 3.13 oder 3.14 | Laufzeit (`requires-python = ">=3.13,<3.15"`); `.python-version` legt lokal 3.14 fest, wie in der CI |
| [uv](https://docs.astral.sh/uv/) | Abhängigkeiten, virtuelle Umgebung und bei Bedarf Python selbst |
| `ffmpeg` und `ffprobe` im `PATH` | Aufnahme von Bildschirm und Mikrofon, Bildzeiten und Dauer der Aufnahmen |
| Zugang zur Anthropic-API | Urteile des Sprach- und des Bildmodells in der Segmentierung |

Die Aufnahme selbst braucht keinen API-Zugang. Er wird erst benötigt, wenn eine
Aufnahme segmentiert oder ein Korpus ausgewertet wird und die Urteile noch nicht
im Cache der Session liegen.

## Installation

```bash
uv sync
```

`uv sync` installiert die Laufzeitabhängigkeiten und die Gruppe `dev`
(Tests, Linting) aus `uv.lock`. Für diese Dokumentation zusätzlich:

```bash
uv sync --group docs
```

Das deutsche spaCy-Modell `de_core_news_lg` wird als Abhängigkeit mitinstalliert;
Whisper lädt sein Modell (`small`) beim ersten Transkribieren herunter.

### Zugang zum Modell

DocuPilot liest Zugangsdaten so, wie das Anthropic-SDK sie auflöst: aus
`ANTHROPIC_API_KEY` oder einem mit `ant auth login` angelegten Profil. Eine
Datei `.env` im Projektverzeichnis wird beim Start eingelesen, ohne bereits
gesetzte Variablen zu überschreiben:

```ini
ANTHROPIC_API_KEY=...
# optional: anderes Modell für beide Urteilsstufen
DOCUPILOT_MODEL=claude-sonnet-5
```

Beide Urteilsstufen – Sprache und Video – verwenden bewusst **dasselbe** Modell,
damit der Modalitätsvergleich nicht die Fähigkeiten zweier Modelle vergleicht
(siehe {doc}`concepts/reproducibility`).

## Starten

```bash
uv run docupilot
```

oder `python -m docupilot`. Es öffnet sich das Hauptfenster.

## Arbeitsablauf

### 1. Aufnehmen

Im Hauptfenster Bildschirm und Mikrofon wählen und die Aufnahme starten.
DocuPilot schreibt in ein Session-Verzeichnis:

- `recording.mp4` – Bildschirm (10 fps) und Mikrofon,
- `events.json` – jede Maus- und Tastatureingabe mit Zeitstempel auf der Uhr
  des Videos.

Details: {doc}`architecture/recording`.

### 2. Annotieren

Nach dem Stoppen – oder über **Datei → Öffnen…** für eine gespeicherte
Session – zeigt das Annotationsfenster das Video. **Grenze setzen (Beginn)**
und **Ende setzen** markieren die aktuelle Position als Grenze. Die Grenzen
landen in `ground_truth.json`, getrennt nach Grenzdefinition (`begin`, `end`)
und Annotationslauf (1, 2, …). Welche Grenze wo zu setzen ist, legt der
Annotationsleitfaden fest ({doc}`concepts/boundaries`).

**Features** öffnet den Merkmalsdialog: eine Zeitleiste mit der Evidenzkurve
jeder Modalität neben den annotierten Grenzen.

### 3. Auswerten

**Auswertung → Korpus auswählen…** wählt ein Verzeichnis mit
Session-Verzeichnissen; **Auswertung → Auswertungsfenster öffnen…** zeigt es.
Die Tabelle nennt je Session, wie viele Grenzen annotiert sind und ob
Modellurteile schon im Cache liegen. Ein Lauf

1. extrahiert jede Modalität einmal je Session,
2. berechnet alle acht Modalitätskombinationen leave-one-session-out,
3. zeigt den Bericht mit den Abbildungen zu den drei Teilfragen.

Bericht und Ergebnistabelle lassen sich als PDF bzw. CSV exportieren. Ablauf
und Rechenweg: {doc}`architecture/evaluation`.

```{note}
Ein Lauf ruft die Modelle nur für Urteile auf, die noch nicht im Cache der
jeweiligen Session liegen. Ein zweiter Lauf über denselben Korpus kostet keine
Modellaufrufe, solange Modell und Prompt-Version gleich bleiben. Über das Menü
des Auswertungsfensters lassen sich die Caches gezielt löschen
({doc}`reference/data-formats`).
```

## Kommandozeile

Einige Prüfungen laufen ohne Oberfläche:

```bash
# Wie stark weichen Whisper- und WhisperX-Satzanfänge voneinander ab?
uv run python -m docupilot.evaluation.alignment <korpus>

# Wie stabil sind die Modellurteile ohne Cache? (verursacht Modellaufrufe)
uv run python -m docupilot.evaluation.repeatability <korpus>
```
