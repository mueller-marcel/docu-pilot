# Segmentierungs-Pipeline

Modul: {py:mod}`docupilot.segmentation`.

Die Segmentierung beantwortet für eine Aufnahme und je Modalität die Frage
„Wo liegen Handlungsgrenzen?“ – nicht mit einer Liste von Zeitpunkten, sondern
mit einer **Evidenzkurve**: einem Wert zwischen 0 und 1 für jeden Zeitpunkt
({doc}`../concepts/evidence-curves`). Erst die Fusion in der Auswertung
entscheidet, wo tatsächlich Grenzen gesetzt werden.

## Vertrag einer Modalität

Jede Modalität ist ein Modul mit denselben vier Namen:

`MODALITY`
: Name der Modalität (`"video"`, `"audio"`, `"events"`).

`raw_key(settings) -> str`
: Alles, wovon das Rohergebnis abhängt, außer der Aufnahme selbst – etwa
  Modell, Prompt-Version, Messkonstanten. Bestimmt, ob ein gespeichertes
  Rohergebnis wiederverwendet werden darf.

`extract_raw(session, settings, *, use_cache, on_progress, is_cancelled)`
: Die teure Stufe: dekodieren, transkribieren, Modelle befragen. Unabhängig
  von jedem Fold-Parameter, deshalb einmal je Session.

`render(raw, settings, params) -> BoundaryEvidence`
: Die billige Stufe: zeichnet aus dem Rohergebnis die Kurve für die
  Parameter eines Folds (z. B. den Audio-Vorlauf). Beim Video entscheiden θ und
  d, welche Bildschirmänderungen existieren; die zugehörigen Modellurteile
  werden deshalb hier eingeholt – einmal je Bildpaar, danach aus dem Cache.

Eine vierte Modalität bräuchte nur ein Modul mit diesen Namen und einen Eintrag
in `_EXTRACTORS` in {py:mod}`docupilot.segmentation.pipeline`; kein Aufrufer
müsste sich ändern.

## Ergebnis: `BoundaryEvidence`

{py:class}`~docupilot.segmentation.evidence.BoundaryEvidence` enthält

- `times_s` – das Abtastraster mit 50 Hz (`GRID_HZ`),
- `score` – die Evidenzkurve in [0, 1],
- `boundaries_s` – die Grenzen, die die Modalität allein setzen würde
  (Kurve ≥ 0,5; nur zur Anzeige und für den Regel-Entscheider),
- `channels` – optionale Teilkurven derselben Modalität. Die Fusion liest sie
  als getrennte Merkmalsblöcke; für die Shapley-Zerlegung bleiben sie **eine**
  Modalität.

## Ablauf

```{mermaid}
flowchart LR
    S[RecordingSession] --> X1[events.extract_raw<br/>Bursts aus events.json]
    S --> X2[video.extract_raw<br/>SSIM-Scan]
    S --> X3[audio.extract_raw<br/>Whisper · spaCy · LLM]
    X1 --> R1[events.render]
    X2 --> R2[video.render<br/>Änderungen unter θ, d · VLM]
    X3 --> R3[audio.render<br/>Glocke je Satz]
    R1 & R2 & R3 --> B[BoundaryEvidence je Modalität]
```

{py:func}`~docupilot.segmentation.pipeline.extract_raws` führt die Extraktion
in der Reihenfolge Ereignisse → Video → Audio aus, also vom billigsten zum
teuersten; scheitert eine Modalität, laufen die anderen weiter und der Fehler
wird gemeldet. {py:func}`~docupilot.segmentation.pipeline.segment` zeichnet
für den Merkmalsdialog alle drei Kurven mit Standardparametern und speichert
sie als „Lane“ in der Session.

## Zwischenspeicher

{py:mod}`docupilot.segmentation.store` hält zwei Schichten:

| Schicht | Datei | Inhalt |
|---|---|---|
| Rohergebnis | `audio_raw.json`, `video_ssim.npz` | Transkript mit Satzurteilen; SSIM-Scan des Videos |
| Lane | `<modalität>_evidence.npz` | eine gezeichnete Kurve, nur für die Anzeige |
| Urteils-Cache | `gui_vlm_cache.json`, `audio_llm_cache.json` | jedes Modellurteil, geschlüsselt über die Frage selbst |

Jeder Eintrag trägt einen Schlüssel aus allem, wovon er abhängt. Ändert sich
etwa die Prompt-Version, passt der Schlüssel nicht mehr, und der Eintrag wird
neu berechnet. Schreibvorgänge sind atomar: Ein abgebrochener Lauf hinterlässt
nie eine halbe Datei. Die Auswertung liest die Lanes nicht; sie zeichnet die
Kurven je Fold selbst.

## Modellzugang

{py:mod}`docupilot.segmentation.backend` lädt `.env`, hält den einen
Anthropic-Client und die gemeinsame Teilziel-Definition beider Prompts. Es
transportiert keine Daten zwischen den Modalitäten – Audio und Video teilen sich
Modell, Definition und Cache-Format, nie Evidenz. Prompts und Versionen:
{doc}`../reference/model-prompts`.
