# Modellprompts

Beide semantischen Stufen fragen ein Modell dasselbe: **Ändert sich das Teilziel
der Person?** Die Definition des Teilziels steht einmal in
`SUBGOAL_DEFINITION` ({py:mod}`docupilot.segmentation.backend`) und wird in
beide Prompts eingesetzt, damit sie nicht auseinanderlaufen. Sie folgt Kurby &
Zacks (2008), wie die Annotation ({doc}`../concepts/boundaries`). Beispiele in
den Prompts sind erfunden und an keine Anwendung gebunden – nie aus einer
Aufnahme übernommen.

| | Bildmodell | Sprachmodell |
|---|---|---|
| Modul | {py:mod}`docupilot.segmentation.video_scoring` | {py:mod}`docupilot.segmentation.audio_scoring` |
| Eingabe | ein Kompositbild VORHER \| NACHHER mit rotem Rahmen um den geänderten Bereich | alle Sätze einer Session, nummeriert, in einem Aufruf |
| Kategorien | `NEW_SUBGOAL`, `SAME_SUBGOAL`, `NOT_USER` | `NEW_SUBGOAL`, `SAME_SUBGOAL`, `OTHER` |
| Antwort | JSON: Beobachtung, Kategorie, Konfidenz, Begründung | JSON: je Satz Kategorie, Konfidenz, Begründung |
| Prompt-Version | `PROMPT_VERSION` in `video_scoring` | `PROMPT_VERSION` in `audio_scoring` |

**Von der Antwort zur Wahrscheinlichkeit.** P(Grenze) = Konfidenz, wenn die
Kategorie `NEW_SUBGOAL` ist, sonst 1 − Konfidenz. Die Stufen liefern damit eine
abgestufte Evidenz und keine Entscheidung.

**Aufruf.** Adaptives Thinking mit Effort `low` und strukturierter Ausgabe
(JSON-Schema). Eine Ablehnung des Modells ergibt für diese Frage keine Evidenz;
eine abgeschnittene Antwort bricht den Lauf mit einer Fehlermeldung ab, statt
still eine leere Kurve zu liefern.

## Prompt ändern

1. Text in `_SYSTEM` bzw. `SUBGOAL_DEFINITION` anpassen.
2. **`PROMPT_VERSION` erhöhen** – bei einer Änderung an `SUBGOAL_DEFINITION` in
   beiden Modulen. Die Version ist Teil jedes Cache-Schlüssels; ohne neue
   Version würden alte Urteile wiederverwendet.
3. In einem Kommentar über `PROMPT_VERSION` festhalten, was sich geändert hat.

Der nächste Lauf fragt alle Urteile neu ab. Sprache kostet einen Aufruf je
Session; Video einen je Bildschirmänderung unter jedem gewählten (θ, d).
