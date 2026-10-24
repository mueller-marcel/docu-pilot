# Unabhängigkeit und Reproduzierbarkeit

## Unabhängigkeit der Modalitäten

Die Shapley-Zerlegung schreibt jeder Modalität zu, was sie zur Segmentierung
beiträgt. Das ist nur gültig, wenn jede Modalität ausschließlich aus ihrer
eigenen Quelle schöpft. DocuPilot sichert das an mehreren Stellen:

- `video`, `audio` und `events` lesen je genau eine Quelle und importieren
  einander nicht.
- Der Mauszeiger wird nicht in das Video gezeichnet; er würde den
  Ereignisstrom ins Bild tragen.
- Die Ereignis-Modalität nimmt die Dauer des Protokolls aus dessen eigenen
  Lebenszyklus-Ereignissen, nicht aus dem Video.
- Sprach- und Bildmodell teilen sich nur Transport, Modell, Cache-Format und die
  Teilziel-Definition – nie Evidenz ({py:mod}`docupilot.segmentation.backend`).
- In der Fusion liest eine Koalition nur die Merkmalsspalten ihrer eigenen
  Modalitäten; das wird beim Auswählen geprüft.

**Gleiches Modell für Sprache und Bild.** Beide Urteilsstufen nutzen dasselbe
Modell (`DOCUPILOT_MODEL`). Zwei verschiedene Modelle würden dieselbe Frage mit
verschiedenen Fähigkeiten beantworten; der Vergleich würde dann die Modelle
messen, nicht die Modalitäten.

## Keine Information aus der Testsession

Jeder Parameter, der aus Daten bestimmt wird, kommt je äußerem Fold nur aus den
Trainingssessions:

| Parameter | Bestimmung |
|---|---|
| Audio-Vorlauf | Median Satzbeginn → Referenzgrenze |
| Video θ, d | bestes gepooltes F1 der Änderungsbeginne |
| Baumzahl | stabiler Brier-Score auf zurückgehaltenen Trainingssessions |
| Entscheidungsschwelle | innere GroupKFold über die Trainingssessions |
| Baseline v(∅) | mittlere Segmentlänge der Trainingssessions |

Was vor dem Lauf festliegt, hängt nicht vom Ergebnis ab: die Toleranz τ (a
priori), die SESOI (nur aus der Zahl der Referenzgrenzen), alle Konstanten in
{py:mod}`docupilot.config`.

## Deterministische Rechnung

- Whisper transkribiert mit Temperatur 0 (deterministischer Decoder).
- Wald, innere Splits und Zufallsniveau verwenden feste Startwerte; der
  Primärlauf mittelt über mehrere Startwerte, bevor Statistik gebildet wird.
- Die Video-Kandidaten hängen nur vom SSIM-Scan und von (θ, d) ab, nicht von
  einem Modell.

## Modellurteile im Cache

Modellantworten sind nicht deterministisch. Damit ein Ergebnis trotzdem
reproduzierbar ist, wird **jedes Urteil** neben der Aufnahme gespeichert,
geschlüsselt über die Frage selbst:

- Bild: Hash des Kompositbildes + Modell + Prompt-Version,
- Sprache: Hash aller Sätze + Modell + Prompt-Version.

Dasselbe Bildpaar oder dieselbe Erzählung kostet daher nie einen zweiten Aufruf,
egal in welchem Fold oder Lauf sie auftaucht. Eine neue Prompt-Version oder ein
anderes Modell erzeugt neue Schlüssel und damit neue Urteile; die alten bleiben
liegen, werden aber nicht mehr gelesen. Jedes Urteil trägt Modell und
Abrufzeitpunkt.

Wie stabil die Urteile ohne Cache wären, misst
`python -m docupilot.evaluation.repeatability`.

## Testdaten

Die Pipeline unterscheidet nicht zwischen echten, Test- oder synthetischen
Daten: Jeder Eintrag einer Annotation zählt. Ob ein Korpus für eine Aussage
geeignet ist, entscheidet, wer ihn auswertet – nicht der Code.
