# Konfiguration

Modul: {py:mod}`docupilot.config`. Jede Einstellung steht dort einmal, mit
ihrer Herkunft.

## Einstellungen eines Laufs

{py:class}`~docupilot.config.Settings` hält die Schalter eines Laufs. Sie
werden als `docupilot_settings.json` im Korpusverzeichnis gespeichert; das
Auswertungsfenster legt die Datei mit Standardwerten an und öffnet sie zum
Bearbeiten.

| Feld | Standard | Werte | Bedeutung |
|---|---|---|---|
| `boundary_definition` | `"begin"` | `begin`, `end` | welche Annotation die Referenz ist |
| `candidate_pool` | `"grid"` | `grid`, `own`, `union` | wo Grenzen geprüft werden |
| `decider` | `"forest"` | `forest`, `logreg`, `rule` | Entscheider der Fusion |
| `burst_pause_s` | `2.0` | > 0 | Pause, die einen neuen Eingabe-Burst beginnt |
| `event_channels` | `"rest"` | `rest`, `all` | Teilkurven der Ereignis-Modalität |
| `asr_alignment` | `"whisper"` | `whisper`, `whisperx` | Wortzeitstempel der Transkription |
| `tau_primary_s` | `1.0` | > 0 | Toleranz τ der Bewertung |
| `tau_source` | `"a priori, GEBD-Protokoll"` | Text | Herkunft von τ im Bericht |
| `seed` | `0` | ganzzahlig | erster Startwert |
| `n_seeds` | `3` | ≥ 1 | Startwerte; F1 je Session wird über sie gemittelt |

Die SESOI (`sesoi`, `sesoi_source`) wird bei jedem Lauf aus der Annotation
bestimmt und lässt sich nicht setzen. Unbekannte Felder in der Datei werden
abgelehnt.

```json
{
  "boundary_definition": "begin",
  "candidate_pool": "grid",
  "decider": "forest",
  "burst_pause_s": 2.0,
  "event_channels": "rest",
  "asr_alignment": "whisper",
  "tau_primary_s": 1.0,
  "tau_source": "a priori, GEBD-Protokoll",
  "seed": 0,
  "n_seeds": 3
}
```

## Umgebungsvariablen

| Variable | Bedeutung |
|---|---|
| `ANTHROPIC_API_KEY` | API-Schlüssel (alternativ ein Profil aus `ant auth login`) |
| `DOCUPILOT_MODEL` | Modell beider Urteilsstufen, Standard `claude-sonnet-5` |

Beide können in `.env` im Projektverzeichnis stehen; gesetzte Variablen haben
Vorrang.

## Feste Werte

| Konstante | Wert | Herkunft |
|---|---|---|
| `TOLERANCE_S` | 1,0 s | a priori: menschliche Präzision, Granularität ({doc}`../concepts/metrics`) |
| `NMS_S` | 2,0 s | a priori: kleinster Schrittabstand der Annotation |
| `GRID_HZ` | 50 Hz | technisch: Abtastung der Evidenzkurven |
| `CAND_PER_TAU` | 5 | Kandidatenraster τ/5 |
| `FEATURE_WINDOWS_TAU` | 0,5 · 1 · 2 | Merkmalsfenster in Vielfachen von τ |
| `THRESHOLD_GRID` | 0,05 … 0,95 | Kandidaten der Entscheidungsschwelle |
| `INNER_FOLDS` | 5 | innere GroupKFold der Schwellenwahl |
| `FOREST_TREES_STEP`, `FOREST_TREES_MAX` | 50, 300 | Wachstum des Waldes |
| `TREES_HOLDOUT_SHARE`, `TREES_STABLE_TOL`, `TREES_STABLE_STEPS` | 0,2 · 0,001 · 2 | Abbruch des Wachstums |
| `FOREST_MIN_SAMPLES_LEAF`, `FOREST_MAX_SAMPLES` | 5, 0,3 | Regularisierung und Laufzeit |
| `AUDIO_MIN_PAIRS` | 10 | darunter Rückfall auf Vorlauf 0 |
| `VIDEO_THETA_CANDIDATES` | 0,85 · 0,9 · 0,95 · 0,98 | Rauschschwelle θ |
| `VIDEO_MIN_REST_CANDIDATES_S` | 0,5 · 1 · 1,5 · 2 · 3 s | Mindestruhe d |
| `EVENT_CHANNEL_WEIGHTS` | 1 : 1 : 0,5 | Gewichte der Ereignis-Teilkurven in der Anzeige und für A0 |
| `GT_EDGE_S`, `GT_MERGE_S` | 0,3 s, 0,1 s | Randregeln der Annotation |
| `DOUBLE_CLICK_S` | 0,5 s | Windows-Standard |
| `KEY_CLASSES`, `MODIFIER_KEYS` | Tabellen | Tastenklassen der Ereignis-Modalität |

Die Tabelle `PARAMETERS` in {py:mod}`docupilot.config` ordnet die Parameter
zusätzlich nach Klasse – a priori, technisch, gesetzt, gemessen, in-fold.
