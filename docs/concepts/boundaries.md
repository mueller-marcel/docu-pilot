# Handlungsgrenzen und Referenzannotation

## Teilziel und Grenze

DocuPilot folgt der Ereignissegmentierung nach **Kurby & Zacks (2008)**:
Menschen nehmen eine laufende Tätigkeit hierarchisch wahr. Die Aufgabe gliedert
sich in **Teilziele**, und jedes Teilziel wird durch eine Folge feinerer
Handlungen ausgeführt, die alle ihm dienen.

- In der Aufgabe „Bericht vorbereiten“ wären „Diagramm einfügen“, „Überschrift
  formatieren“ oder „Datei unter neuem Namen speichern“ Teilziele.
- Klicken, Tippen, ein Menü oder einen Dialog öffnen, Scrollen oder ein
  Ergebnis prüfen sind feinere Handlungen *innerhalb* eines Teilziels.
- Einen Fehler zu korrigieren, etwa einen Tippfehler, ändert das Teilziel nicht.

Eine **Handlungsgrenze** liegt dort, wo das Teilziel wechselt: wo die Person ein
Teilziel nicht mehr verfolgt und das nächste beginnt. Ein Handlungsschritt ist
das Segment zwischen zwei Grenzen.

Die Segmentierung sucht Grenzen, keine benannten Abschnitte – wie *Generic Event
Boundary Detection* (Shou et al. 2021). Vordefinierte Aktionslabels ließen sich
über Anwendungen hinweg nicht sinnvoll festlegen.

## Wo genau die Grenze liegt

Die Referenzgrenze liegt am **ersten wahrnehmbaren Hinweis** auf das neue
Teilziel, so wie ein Beobachter es sieht, der Bild, Ton und Eingaben zugleich
wahrnimmt:

- ein Satz, der den Schritt ankündigt („Jetzt füge ich …“),
- die erste Bildschirmänderung, die dem neuen Teilziel dient.

Ein Menü oder Dialog zu öffnen ist für sich kein Wechsel; es markiert einen
nur, wenn es die erste Handlung für ein neues Teilziel ist. Eine Denkpause ohne
wahrnehmbare Änderung setzt keine Grenze. Auch das erste Teilziel einer
Aufnahme beginnt mit einer Grenze – es folgt auf den ruhigen Beginn.

Diese Definition ist in Code die Grenzdefinition `begin`. Die Definition `end`
(Grenze am Ende eines Schritts) wird mitgeführt, ist aber nicht die primäre.

Dieselbe Definition steht – als `SUBGOAL_DEFINITION` – in beiden
Modellprompts, damit die Modelle dieselbe Frage beantworten wie die Annotation
({doc}`../reference/model-prompts`).

## Referenzannotation

Die Annotation entsteht im Annotationsfenster und liegt in
`ground_truth.json` ({doc}`../reference/data-formats`). Jeder Eintrag nennt
seine Definition und seinen **Lauf**: Lauf 1 ist die Referenz, Lauf 2, 3, …
sind Wiederholungen. Alle Einträge zählen gleich; es wird nicht nach Herkunft
der Daten unterschieden.

Beim Lesen gelten zwei Randregeln des Leitfadens
({py:func}`~docupilot.evaluation.dataset.apply_edge_rules`):

- Grenzen in den ersten oder letzten 0,3 s entfallen.
- Grenzen, die weniger als 0,1 s auseinanderliegen, werden zu ihrem Mittel
  zusammengelegt – ein doppelter Klick auf den Annotationsknopf ist eine Grenze.

## Zeitliche Unschärfe

Eine Grenze ist kein exakt reproduzierbarer Zeitpunkt: Zwischen dem Hinweis und
dem Tastendruck liegt die Reaktionszeit, und verschiedene Hinweise auf
denselben Wechsel liegen Sekundenbruchteile auseinander. Die Auswertung misst
deshalb nicht punktgenau, sondern mit einer Toleranz τ
({doc}`metrics`).
