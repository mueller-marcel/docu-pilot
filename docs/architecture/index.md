# Architekturüberblick

DocuPilot ist eine Desktop-Anwendung (PySide6) mit einer Rechen-Pipeline, die
vollständig ohne Oberfläche läuft. Das Paket `docupilot` gliedert sich in vier
Teilpakete und eine zentrale Konfiguration:

| Paket | Aufgabe | Abhängig von |
|---|---|---|
| {py:mod}`docupilot.recording` | Aufnahme von Bildschirm, Mikrofon und Eingaben; Session-Verzeichnis und Referenzannotation | – |
| {py:mod}`docupilot.segmentation` | je Modalität eine Evidenzkurve für Handlungsgrenzen | `recording`, `config` |
| {py:mod}`docupilot.evaluation` | Experiment über einen Korpus: Fusion, Kalibrierung, Metriken, Statistik, Bericht | `segmentation`, `recording`, `config` |
| {py:mod}`docupilot.ui` | Fenster, Dialoge und Diagramme | alle anderen |
| {py:mod}`docupilot.config` | jede Einstellung mit ihrer Herkunft | – |

```{mermaid}
flowchart TB
    subgraph ui[ui – PySide6]
        MW[MainWindow] --> AW[AnnotationWindow]
        MW --> EW[ExperimentWindow]
        AW --> FD[FeatureDialog]
        EW --> RV[report_view · Diagramme · PDF]
    end
    subgraph recording
        RS[RecorderService] --> AV[AvRecorder · ffmpeg]
        RS --> IR[InputRecorder · pynput]
        RS --> SES[RecordingSession]
    end
    subgraph segmentation
        PL[pipeline] --> V[video]
        PL --> A[audio]
        PL --> E[events]
        V --> VS[video_scoring · VLM]
        A --> AS[audio_scoring · LLM]
        V & A & E --> EV[evidence]
    end
    subgraph evaluation
        EXP[experiment] --> CAL[calibration]
        EXP --> FU[fusion]
        EXP --> MET[metrics]
        REP[report] --> STAT[statistics]
        REP --> AN[analysis]
    end
    MW --> RS
    FD --> PL
    EW --> EXP
    EXP --> PL
    RV --> REP
```

## Leitende Entwurfsentscheidungen

**Rechnen ohne Oberfläche.** Alles, was eine Zahl der Auswertung erzeugt, liegt
in `segmentation` und `evaluation` und importiert kein Qt. Fenster und PDF
lesen dieselben Berichtsobjekte ({py:mod}`docupilot.evaluation.report`);
dadurch können Fenster und PDF nicht auseinanderlaufen, und die Ergebnisse
lassen sich per Skript reproduzieren.

**Unabhängige Modalitäten.** Die Module `video`, `audio` und `events` lesen je
nur ihre eigene Datei und importieren einander nicht. Jede Querverbindung würde
in der Shapley-Zerlegung als Beitrag einer Modalität erscheinen, der in
Wahrheit von einer anderen stammt ({doc}`../concepts/reproducibility`).

**Zwei Stufen je Modalität.** `extract_raw` erledigt das Teure (Dekodieren,
Transkribieren, Modellaufrufe) einmal je Session; `render` zeichnet daraus
billig eine Kurve für die Parameter eines Folds. Kalibrieren kostet so keine
erneute Extraktion ({doc}`segmentation`).

**Keine Information aus der bewerteten Session.** Jeder Parameter, der aus
Daten bestimmt wird – Schwelle, Audio-Vorlauf, θ und d des Videos, Zahl der
Bäume – wird je äußerem Fold nur auf den Trainingssessions bestimmt
({doc}`evaluation`).

**Zwischenergebnisse liegen neben der Aufnahme.** Teure Ergebnisse werden im
Session-Verzeichnis gespeichert, jeweils unter einem Schlüssel, der alles nennt,
wovon sie abhängen. Ein Schlüssel, der nicht passt, ist ein Fehlschlag, nie
eine veraltete Antwort ({doc}`../reference/data-formats`).

## Datenfluss einer Auswertung

```{mermaid}
sequenceDiagram
    participant W as ExperimentWindow
    participant X as experiment
    participant P as segmentation.pipeline
    participant R as report
    W->>X: load(Korpus)
    X->>P: extract_raws je Session
    P-->>X: Rohergebnisse (Video-Scan, Transkript+Urteile, Bursts)
    W->>X: measure_sesoi
    W->>X: run
    loop jede Session als Testsession
        X->>X: calibrate (Trainingssessions)
        X->>P: render (Fold-Parameter)
        X->>X: Merkmale · Wald · Schwelle · NMS
        X->>X: match gegen Referenz (±τ)
    end
    X-->>W: Ergebnistabelle
    W->>R: analyse
    R-->>W: Abbildungen 4.2–4.6
```
