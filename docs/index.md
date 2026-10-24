# DocuPilot

DocuPilot records a person's work in a desktop application –
screen, microphone and all mouse and keyboard events – and automatically splits the
recording into **action steps**. An action boundary lies where
the person's sub-goal changes.

Besides the segmentation itself, DocuPilot answers a research question:
**What information contribution does each of the three modalities make to
segmentation, and from which point on does a further modality no longer bring a
relevant gain?** To this end, each of the eight combinations of the modalities is
evaluated leave-one-session-out against a reference annotation, and the contribution
of each modality is decomposed with Shapley values.

If you are using DocuPilot for the first time, start with {doc}`getting-started`;
if you want to understand or extend the code, start with {doc}`architecture/index`.

```{mermaid}
flowchart LR
    A[Recording<br/>recording.mp4 · events.json] --> B[Annotation<br/>ground_truth.json]
    A --> C[Segmentation<br/>Video · Speech · Events]
    C --> D[Fusion & evaluation<br/>8 coalitions, LOSO]
    B --> D
    D --> E[Report<br/>figures]
```

## Contents

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started
```

```{toctree}
:maxdepth: 2
:caption: Architecture

architecture/index
architecture/recording
architecture/segmentation
architecture/modalities
architecture/evaluation
architecture/ui
```

```{toctree}
:maxdepth: 2
:caption: Concepts

concepts/boundaries
concepts/evidence-curves
concepts/fusion
concepts/metrics
concepts/statistics
concepts/reproducibility
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/data-formats
reference/configuration
reference/model-prompts
glossary
reference/api
```

```{toctree}
:maxdepth: 1
:caption: Development

development
```
