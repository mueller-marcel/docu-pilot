# Glossar

```{glossary}
Annotationslauf
  Eine vollständige Annotation einer Session. Lauf 1 ist die Referenz, weitere
  Läufe sind Wiederholungen.

Baseline v(∅)
  Grenzen im festen Abstand der mittleren Segmentlänge, ohne jede Modalität.
  Nullpunkt der Shapley-Werte.

Burst
  Folge von Eingabeereignissen ohne Pause länger als 2 s.

Evidenzkurve
  Wert in [0, 1] je Zeitpunkt, wie stark eine Modalität dort eine Grenze
  anzeigt. Siehe {doc}`concepts/evidence-curves`.

Fold
  Ein Durchgang des Leave-one-session-out: eine Session ist Testsession, alle
  anderen sind Trainingssessions.

Handlungsgrenze
  Zeitpunkt, an dem das Teilziel der Person wechselt; liegt am ersten
  wahrnehmbaren Hinweis auf das neue Teilziel.

Koalition
  Teilmenge der Modalitäten, die gemeinsam bewertet wird; mit drei Modalitäten
  gibt es acht.

Lane
  Gespeicherte Evidenzkurve einer Modalität für die Anzeige im Merkmalsdialog.

Non-Maximum-Suppression (NMS)
  Von zwei vorhergesagten Grenzen, die näher als 2 s beieinander liegen, bleibt
  nur die stärkere.

Rohergebnis
  Teures Zwischenergebnis einer Modalität vor dem Zeichnen der Kurve: SSIM-Scan,
  Transkript mit Urteilen, Bursts.

Ruhephase
  Folge von Videobildern mit SSIM ≥ θ zum Vorgänger, mindestens d Sekunden lang.

SESOI
  Smallest effect size of interest: eine zusätzlich richtig erkannte Grenze, im
  Mittel 1/n über die Sessions.

Shapley-Wert
  Mittlerer marginaler Beitrag einer Modalität über alle Reihenfolgen, in denen
  die Modalitäten hinzukommen können.

Sättigungspunkt k*
  Letzte Anzahl an Modalitäten, deren Hinzunahme noch einen relevanten Zugewinn
  bringt.

Teilziel
  Was eine Person mit einer Folge feinerer Handlungen erreichen will (Kurby &
  Zacks 2008); siehe {doc}`concepts/boundaries`.

Toleranz τ
  Höchster Abstand, mit dem eine Vorhersage noch als Treffer einer
  Referenzgrenze zählt; 1 s.

Vorlauf
  Abstand vom Satzbeginn bis zur Referenzgrenze, je Fold gemessen.

Zugewinn Δ(k)
  Differenz der Sättigungskurve von k − 1 zu k Modalitäten.
```
