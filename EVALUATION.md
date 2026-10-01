# Evaluationsplan

Wie aus der implementierten Segmentierung die drei Teilfragen der Masterarbeit
beantwortet werden. Stand: Änderungsspezifikation v2 (30.09.2026). Ergebnisse,
die vor diesem Stand erzeugt wurden, gelten nur noch als **Pilot**.

**Forschungsfrage:** Wie verändert sich die Qualität der automatisierten
Segmentierung in Abhängigkeit von Art, Anzahl und Kombination der Modalitäten —
und ab welchem Punkt liefert eine weitere Modalität keinen relevanten Beitrag
mehr?

| | Teilfrage | beantwortet durch |
|---|---|---|
| **TF1** | Welche Modalität leistet isoliert am meisten? | Tabelle der 8 Koalitionen |
| **TF2** | Welchen marginalen Beitrag leistet jede Modalität? | Shapley-Werte |
| **TF3** | Ab wann tritt Sättigung / Redundanz ein? | Schritte Δ(k), k*, Interaktionsindex |

Datensatz: **25 selbst aufgenommene und annotierte Sessions**, ein Annotator.
Es gibt **keine Entwicklungssession** außerhalb des Korpus.

---

## Grenzdefinition

Grenze = **Beginn der ersten beobachtbaren Handlung des neuen Schritts**
(Annotationsleitfaden). Alle drei Extraktoren setzen ihre Evidenz dorthin
(`Settings.boundary_definition = "begin"`); „Ende“ (Ergebnis sichtbar und
dauerhaft) bleibt nur als Sensitivitätsvariante und wird nur gegen eine
„Ende“-Annotation ausgewertet.

| Modalität | Urteil (unverändert) | Zeitpunkt der Evidenz |
|---|---|---|
| Ereignisse | – | erstes Ereignis eines Bursts; Teilkurven Ruhe davor, Abschluss-Taste im vorigen Burst, Klassenwechsel |
| Video | VLM: eingerasteter Zustand gegen den Anchor | erstes Frame nach dem Dwell mit Aktivität ≥ ACTIVITY_QUIET; letzter Dwell ohne Folgeaktivität: keine Evidenz |
| Audio | LLM: kündigt der Satz eine Operation an? | Peak = Anker (Satzbeginn/-ende) + Versatz d mit Vorzeichen, begrenzt auf [t_i − 2 s, t_i + 0,9·Fenster] |

Randregeln des Leitfadens werden einmal zentral angewandt (`dataset.py`):
Grenzen in den ersten/letzten 0,3 s entfallen, Grenzen mit < 0,1 s Abstand
werden zusammengeführt.

---

## Schicht 1 — Extraktion (`segmentation/`)

Jede Modalität hat zwei Stufen:

- `extract_raw` — Dekodieren und Modellaufrufe, gecacht (`<modalität>_raw.json`,
  `video_activity.npz`, Urteilscaches). Unabhängig von Definition und
  Fold-Parametern.
- `render` — die Kurve, billig. Hängt von Definition, Peak-Modus und den in-fold
  kalibrierten Parametern ab.

Eine Parameteränderung löst daher nie einen Modellaufruf aus. Jeder
Cache-Eintrag trägt Modell-ID und Abrufdatum.

## Schicht 2 — Kandidaten, Merkmale, Entscheidung (`fusion.py`)

**Kandidaten:** ein neutrales **5-Hz-Raster** über die Aufnahme, für alle 8
Koalitionen einer Session bitgleich. Schon *wo* geprüft wird, ist Information;
das Raster trägt keine. „own“ und „union“ (Peaks der eigenen bzw. aller Kurven)
bleiben als Sensitivität.

**Merkmale je Kurve** (Ereignisse: je Teilkurve): Punktwert (lineare
Interpolation), Maximum in ±0,5/1/2 s, Sessionrang des ±1-s-Maximums. Eine
Koalition sieht nur Spalten ihrer eigenen Modalitäten.

**Label:** positiv, wenn eine Referenzgrenze in ±1 s liegt. Bei 5 Hz sind je
Grenze etwa 11 Rasterpunkte positiv — Klassenverhältnis **≈ 1 : 15**
(`class_weight="balanced"`).

**Entscheidung:** p ≥ Schwelle → lokale Maxima von p entlang der Zeit → NMS 1 s.
Identisch bei Schwellenwahl und Testvorhersage.

**Entscheider:** Random Forest (primär), logistische Regression, und die
untrainierte Regel **A0** (Maximum der Score-Kurven, Peaks ≥ 0,5, Abstand ≥ 1 s).

**Leere Koalition:** `v(∅)` ist die informationslose Baseline — Grenzen im
festen Abstand der mittleren Segmentlänge der Trainingssessions. Das
Zufallsniveau (gleichverteilte Grenzen) bleibt als zweite Referenz im Bericht.

## Schicht 3 — Leave-One-Session-Out (`experiment.py`)

Je äußerem Fold, ausschließlich auf den 24 Trainingssessions:

1. **Kalibrieren** (`calibration.py`): REST_FULL_S ∈ {4, 6, 8, 10, 12} s über A0
   auf {Ereignisse}; AUDIO_ANCHOR und PEAK_OFFSET_S aus den Abständen
   Referenzgrenze − Anker der OPERATION-Sätze (< 10 Paare → Satzbeginn, d = 0).
   Zuordnungsquote und Abstandshistogramm gehen in den Bericht.
2. **Rendern** aller Kurven mit diesen Parametern.
3. Je Koalition **Schwelle über eine innere GroupKFold** (5 Folds) wählen,
   Endmodell auf allen Trainingssessions fitten, Testsession vorhersagen. Die
   OOB-Schwelle wird nur als Vergleichsspalte geloggt.

Das Ergebnis-CSV hat eine Zeile je (Session × Koalition × τ) mit Schwelle,
OOB-Schwelle und den kalibrierten Parametern des Folds.

## Schicht 4 — Attribution (`analysis.py`)

Exakte Shapley-Werte über 8 Koalitionen, Interaktionsindex (Grabisch & Roubens
1999), Sättigungskurve. Selbsttest: Σφ = v(N) − v(∅).

## Schicht 5 — Statistik (`statistics.py`)

Shapley-Werte, Interaktionen und Schritte Δ(k) sind linear in den v(S) und
werden **je Session** berechnet. Je Größe: Mittel, sd, t-Intervall — **naiv**
und **korrigiert nach Nadeau & Bengio (2003)** (Faktor 1/n + 1/(n−1) statt 1/n;
bei n = 25 ≈ 1,43-fach breiter). Beide werden immer berichtet.

| Klasse | Bedingung | Bedeutung |
|---|---|---|
| Ä | CI vollständig in [−SESOI, +SESOI] | nachweislich vernachlässigbar |
| R | CI schließt 0 aus und \|Schätzer\| ≥ SESOI | statistisch und praktisch relevant |
| S | CI schließt 0 aus, sonst | statistisch gesichert, Größe offen |
| U | sonst | unentscheidbar; n_eq berichten |

Sättigungspunkt k* = kleinstes k, ab dem alle späteren Δ(j) in Ä liegen; ist ein
späterer Schritt U, lautet das Ergebnis „nicht entscheidbar“. MDE, n_req und
n_eq werden je Kontrast aus dessen eigener gepaarter Streuung berechnet.
Cluster-Bootstrap nur als Kontrolle (`bootstrap_check`), Bonferroni per
`multiplicity`.

## Reliabilität (`reliability.py`)

Blinde, zeitversetzte Wiederholungsannotation (Lauf 2, ≥ 3 Wochen, ≥ 8
Sessions). Daraus: Abstandsstatistik (Mittel, Median, P90, P95), Konsistenz-F1
(τ = 0,2 … 1,0 s) und SESOI-Kandidaten (F1-Differenz des vollen Modells
zwischen Lauf 1 und 2). Welche Kennzahl τ und SESOI definiert, ist festgelegt in
`Settings.tau_rule` / `sesoi_rule`; übernommen werden sie mit Herkunftsvermerk
in `docupilot_settings.json`.

---

## Parameter ohne Dev-Session

Jeder Parameter ist a priori fest, eingefroren oder in-fold kalibriert — die
vollständige Tabelle steht in `config.PARAMETERS` und im Anhang des Berichts.
**ACTIVITY_QUIET = 0,08 und _MIN_DWELL_S = 0,5 s sind eingefroren**: Sie
bestimmen die VLM-Aufrufe und wurden laut Code auf einer Entwicklungssession
gesweept, die Teil des Korpus ist. Sie werden nicht neu getunt; in der Arbeit
als Limitation benennen.

## Konfiguration und Methodenvergleich

`docupilot_settings.json` im Korpusverzeichnis hält alle Schalter eines Laufs
(`config.Settings`) und die Liste der Varianten. Je Variante läuft der volle
Versuch über 8 Koalitionen; berichtet werden nur Rangfolge der φ und
Sättigungsklasse. Standard: A0, Kandidatenpool „union“ (Zeitpunkt-Kredit
v_union({m}) − v_grid({m})), Ereignisse nur mit Teilkurve „Ruhe“ (Beitrag der
Merkmale nach Rebmann & van der Aa). Weitere Achsen: `burst_pause_s`,
`audio_peak_mode`, `boundary_definition`, `activity_quiet` (neue VLM-Aufrufe),
`asr_alignment` (WhisperX, optional).

## Wiederholbarkeit der Modellurteile

```bash
python -m docupilot.evaluation.repeatability <korpus> --sessions 5 --runs 3
```

3 Läufe ohne Cache auf 5 Sessions: Anteil identischer Kategorien, Spanne von
p_boundary, Spanne des F1 der Einzelmodalität. Kostet bis zu 400 VLM-Aufrufe je
Session und Lauf.

---

## Modulaufbau

```
evaluation/
  metrics.py         Matching (Kuhn–Munkres), Precision/Recall/F1, Zufallsniveau
  dataset.py         Ground Truth je Definition und Lauf, Randregeln, Dauer
  media.py           ein ffprobe je Aufnahme, geteilt
  corpus.py          Inhalt eines Korpusverzeichnisses
  fusion.py          Raster, Merkmale, v(∅), A0, Forest/LogReg, Schwelle
  calibration.py     in-fold kalibrierte Parameter
  experiment.py      LOSO über alle Sessions × 8 Koalitionen
  analysis.py        Shapley, Interaktionsindex, Sättigungskurve
  statistics.py      Intervalle je Session, Klassenregel, n je Kontrast
  reliability.py     Test-Retest: τ und SESOI
  synchronization.py Uhrenversatz, Drift, Satzstart-Vergleich
  repeatability.py   Stabilität der Modellurteile ohne Cache
  report.py          fertige Auswertung + Texte, ohne Qt
```

Die Tests in `tests/` prüfen Eigenschaften statt Goldens — u. a. Raster-
Identität, Spaltenisolation, Invarianz gegen eine verrauschte fremde Kurve und
den Leckage-Test mit NaN-Labels der Testsession.

## Offene Punkte außerhalb des Codes

- **N1** Klären, nach welcher Definition die 25 Sessions annotiert sind; ggf.
  nach „Beginn“ neu annotieren. Einträge ohne Definition gelten als „Ende“,
  der alte Typ „start“ als „Beginn“.
- **N2** Wiederholungsannotation (Lauf 2) durchführen.
- **N3** Leitfaden ergänzen und versionieren (`config.GUIDELINE_VERSION`).
- **N4** Satzanfänge stichprobenartig prüfen (Schwelle für WhisperX).
- **N5** Entscheidungen festlegen: τ-Regel, SESOI-Regel, Konfidenzniveau,
  Mehrfachvergleich, primäres Intervall, Early Fusion / TCN.
- **N7** Voll-Lauf auf fixiertem Commit; Einstellungen, Caches und CSV archivieren.

---

## Literatur

- Shapley, L. S. (1953): *A Value for n-Person Games.*
- Grabisch, M.; Roubens, M. (1999): *An axiomatic approach to the concept of
  interaction among players in cooperative games.*
- Nadeau, C.; Bengio, Y. (2003): *Inference for the Generalization Error.*
- Rebmann, A.; van der Aa, H. (2023/2024): *Unsupervised task recognition from
  user interaction streams.*
- Lakens, D. (2022): *Sample Size Justification.*
- Lakens, D. (2017): *Equivalence Tests.*
- Wengelin, Å. (2006): *Examining pauses in writing.*
