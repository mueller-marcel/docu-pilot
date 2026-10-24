# DocuPilot

DocuPilot zeichnet die Arbeit einer Person in einer Desktop-Anwendung auf –
Bildschirm, Mikrofon und alle Maus- und Tastaturereignisse – und zerlegt die
Aufnahme automatisch in **Handlungsschritte**. Eine Handlungsgrenze liegt dort,
wo das Teilziel der Person wechselt.

Neben der Segmentierung selbst beantwortet DocuPilot eine Forschungsfrage:
**Welchen Informationsbeitrag liefert jede der drei Modalitäten zur
Segmentierung, und ab wann bringt eine weitere Modalität keinen relevanten
Gewinn mehr?** Dazu wird jede der acht Kombinationen der Modalitäten
leave-one-session-out gegen eine Referenzannotation ausgewertet und der Beitrag
jeder Modalität mit Shapley-Werten zerlegt.

Wer DocuPilot zum ersten Mal einsetzt, beginnt mit {doc}`getting-started`;
wer den Code verstehen oder erweitern will, mit {doc}`architecture/index`.

```{mermaid}
flowchart LR
    A[Aufnahme<br/>recording.mp4 · events.json] --> B[Annotation<br/>ground_truth.json]
    A --> C[Segmentierung<br/>Video · Sprache · Ereignisse]
    C --> D[Fusion & Auswertung<br/>8 Koalitionen, LOSO]
    B --> D
    D --> E[Bericht<br/>Abb. 4.2–4.6]
```

## Inhalt

```{toctree}
:maxdepth: 2
:caption: Einstieg

getting-started
```

```{toctree}
:maxdepth: 2
:caption: Architektur

architecture/index
architecture/recording
architecture/segmentation
architecture/modalities
architecture/evaluation
architecture/ui
```

```{toctree}
:maxdepth: 2
:caption: Konzepte

concepts/boundaries
concepts/evidence-curves
concepts/fusion
concepts/metrics
concepts/statistics
concepts/reproducibility
```

```{toctree}
:maxdepth: 2
:caption: Referenz

reference/data-formats
reference/configuration
reference/model-prompts
glossary
reference/api
```

```{toctree}
:maxdepth: 1
:caption: Entwicklung

development
```
