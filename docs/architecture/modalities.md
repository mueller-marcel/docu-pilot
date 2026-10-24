# Die drei Modalitäten

Jede Modalität liest genau eine Quelle der Aufnahme und keine Ausgabe einer
anderen Modalität.

| | Video | Sprache | Ereignisse |
|---|---|---|---|
| Quelle | Bildspur von `recording.mp4` | Tonspur von `recording.mp4` | `events.json` |
| Kandidaten | Beginn einer Bildschirmänderung | Satzanfang | Beginn eines Eingabe-Bursts |
| Semantisches Urteil | Bildmodell (VORHER\|NACHHER) | Sprachmodell (ganze Erzählung) | – |
| Kurvenform | Gauß-Glocke am Änderungsbeginn | Raised-Cosine über dem Satz | Gauß-Glocke am Burst-Beginn |
| In-Fold-Parameter | Rauschschwelle θ, Mindestruhe d | Vorlauf Satzbeginn → Grenze | – |
| Modul | {py:mod}`docupilot.segmentation.video` | {py:mod}`docupilot.segmentation.audio` | {py:mod}`docupilot.segmentation.events` |

## Video

1. **Ähnlichkeitssignal.** Jedes Bild wird auf 896 px Breite verkleinert und in
   Graustufen umgerechnet. Für jedes Bild wird die SSIM (Wang et al. 2004) zum
   Vorgänger berechnet, getrennt für 8 × 8 Kacheln; zählt die **kleinste**
   Kachel. So verschwindet eine kleine Änderung – eine neue Zeile Text – nicht
   im Mittel über den ganzen Bildschirm, während Kodierrauschen die SSIM kaum
   bewegt. Die Bildzeiten kommen per `ffprobe` direkt aus dem Container. Der
   Scan wird als `video_ssim.npz` gespeichert.
2. **Ruhe und Änderung.** Ein Bild ist ruhig, wenn seine SSIM ≥ θ ist. Eine
   Folge ruhiger Bilder von mindestens d Sekunden ist eine **Ruhephase**.
   Zwischen je zwei Ruhephasen liegt eine **Änderung**; ihr Beginn ist das
   erste bewegte Bild. VORHER und NACHHER sind die Zustandsbilder der beiden
   Ruhephasen, jeweils 0,2 s nach deren Beginn (damit keine ausblendende
   Animation beurteilt wird). Auch die erste Änderung einer Aufnahme zählt –
   sie beginnt das erste Teilziel, und die Annotation markiert diesen Beginn.
3. **Urteil.** Unterscheiden sich VORHER und NACHHER in keiner Kachel um mehr
   als einen Grauwert im Mittel, entfällt die Änderung ohne Modellaufruf.
   Sonst bekommt das Bildmodell beide Bilder als **ein** Kompositbild (links
   VORHER, rechts NACHHER, rot umrandet der geänderte Bereich) und antwortet mit
   `NEW_SUBGOAL`, `SAME_SUBGOAL` oder `NOT_USER` samt Konfidenz.
   P(Grenze) ist die Konfidenz bei `NEW_SUBGOAL`, sonst ihr Komplement. Höchstens
   400 Aufrufe je Session und (θ, d); ein Erreichen der Grenze wird gemeldet.
4. **Kurve.** Je Urteil eine Gauß-Glocke der Höhe P(Grenze) am
   Änderungsbeginn.

θ und d werden je Fold auf den Trainingssessions gewählt: Für jedes Paar aus
θ ∈ {0,85; 0,9; 0,95; 0,98} und d ∈ {0,5; 1; 1,5; 2; 3} s werden die
Änderungsbeginne ohne Modellaufruf gegen die Referenz gematcht; das Paar mit dem
höchsten gepoolten F1-Wert gewinnt ({py:func}`~docupilot.evaluation.calibration.calibrate_video`).

```{note}
Jedes Paar (θ, d), das irgendein Fold wählt, wird für alle Sessions beurteilt,
weil die Trainingssessions dieses Folds ihre Kurven damit brauchen. Wählen die
Folds verschiedene Paare, steigt die Zahl der Modellaufrufe entsprechend.
```

## Sprache

1. **Transkript.** Whisper (`small`, Deutsch, Temperatur 0 für ein
   deterministisches Ergebnis) transkribiert die Tonspur mit Wortzeitstempeln.
   Optional richtet WhisperX die Wörter per Forced Alignment aus
   (`asr_alignment = "whisperx"`).
2. **Sätze.** spaCy (`de_core_news_lg`) zerlegt das Transkript in Sätze; ein
   Satz beginnt mit dem Zeitstempel seines ersten Wortes und endet mit dem
   seines letzten.
3. **Urteil.** **Ein** Aufruf übergibt dem Sprachmodell alle Sätze der Session
   nummeriert, damit jeder Satz im Zusammenhang beurteilt wird. Kategorien:
   `NEW_SUBGOAL`, `SAME_SUBGOAL`, `OTHER`.
4. **Kurve.** Jeder Satz zeichnet eine Raised-Cosine-Glocke über sein Intervall
   [Beginn, Ende]. Ihr Gipfel liegt beim **Vorlauf** nach dem Satzbeginn: dem
   Median des Abstands Satzbeginn → erste Referenzgrenze im Satz, je Fold auf
   den Trainingssessions gemessen. Zwei Teilkurven:
   - `judged` – Höhe = P(Grenze) aus dem Modellurteil,
   - `announced` – Höhe 1 für jeden Satz: Die Teilnehmenden sollten jeden
     Schritt ankündigen, also ist schon der Satzbeginn ein Hinweis.

Sprache weiß das **Intervall**, in dem eine Grenze liegt, nicht den genauen
Zeitpunkt. Der Vorlauf übersetzt das Intervall in einen erwarteten Zeitpunkt,
ohne Information aus einer anderen Modalität.

## Ereignisse

1. **Klassifizieren.** Jedes Eingabeereignis bekommt eine Klasse aus
   Aktionstyp und Tastenklasse (Zeichen, Navigation, Abschluss wie Enter oder
   Strg+S, Abbruch wie Esc oder Strg+Z, Shortcut; Klick, Doppelklick,
   Rechtsklick, Scrollen). Modifikatortasten und Freigaben sind keine eigene
   Aktion. Zwei Linksklicks innerhalb von 0,5 s sind ein Doppelklick.
2. **Bursts.** Eine Pause von mehr als 2 s beginnt einen neuen Burst
   (Pausenschwelle der Schreibprozessforschung, Wengelin 2006).
3. **Übergänge.** Am ersten Ereignis jedes Bursts werden drei Teilkurven
   gezeichnet:
   - `rest` – Länge der Pause davor, geteilt durch die längste Pause der Session,
   - `completion` – der vorige Burst endete mit einer Abschluss-Taste ohne
     folgendes Abbruch-Ereignis,
   - `class_change` – 1 − Kosinus-Ähnlichkeit der Klassenhistogramme beider Bursts.

Der Primärlauf nutzt nur `rest` (`event_channels = "rest"`); alle drei sind die
Variante `"all"`. Die Bildschirmposition von Klicks wird nicht ausgewertet.
