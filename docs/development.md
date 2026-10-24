# Entwicklung

## Umgebung

```bash
poetry install --with dev,docs
```

| Gruppe | Inhalt |
|---|---|
| (Haupt) | Laufzeitabhängigkeiten |
| `dev` | `ruff`, `pytest` |
| `docs` | `sphinx`, `furo`, `myst-parser`, `sphinx-autoapi`, `sphinx-copybutton`, `sphinxcontrib-mermaid` |

## Prüfen

```bash
poetry run ruff check .
poetry run pytest
```

Die Tests in `tests/` sichern die Rechenwege, auf denen die Ergebnisse beruhen:
Zuordnung der Grenzen, Randregeln, 95-%-Intervalle, SESOI, Klassen,
Sättigungspunkt, Shapley-Werte, Sättigungskurve und der Weg von der
Ergebnistabelle zu den Abbildungen. Ein Test rechnet die veröffentlichten
Shapley-Werte und Zugewinne aus den Koalitionswerten eines Laufs nach.

Modellaufrufe, Whisper und das Dekodieren von Videos sind nicht getestet: Sie
brauchen Netz, Zugangsdaten oder Aufnahmen und wären in der CI nicht
reproduzierbar.

## Dokumentation bauen

```bash
poetry run sphinx-build -W --keep-going -b html docs docs/_build/html
```

Die API-Referenz entsteht dabei aus den Docstrings; das Paket wird nicht
importiert. `-W` macht jede Warnung zum Fehler – ein toter Querverweis bricht
den Build ab. Live-Vorschau beim Schreiben, z. B. mit `sphinx-autobuild`
(nicht Teil der Abhängigkeiten):

```bash
poetry run pip install sphinx-autobuild
poetry run sphinx-autobuild docs docs/_build/html
```

Die Seiten sind Markdown (MyST); Diagramme sind Mermaid-Blöcke.

## Kontinuierliche Integration

`.github/workflows/pull-request.yml` läuft bei jedem Pull Request und Push auf
`main`:

| Job | Schritte |
|---|---|
| Lint and package | `poetry check --lock`, `ruff check .`, `poetry build` |
| Test | `poetry install --with dev`, `pytest tests` |
| Docs | `poetry install --only docs --no-root`, `sphinx-build -W` |

## Konventionen

- Docstrings und Bezeichner auf Englisch, Texte der Oberfläche und des Berichts
  auf Deutsch.
- Docstrings im Sphinx-Feldstil (`:param x:`, `:return:`, `:raises E:`).
- Zeilenlänge 100 (`ruff`).
- Rechenwege importieren kein Qt; die Oberfläche rechnet nichts selbst.
- Jede Konstante in {py:mod}`docupilot.config` nennt ihre Herkunft.
- Eine Änderung an einem Modellprompt erhöht dessen `PROMPT_VERSION`.
- Kein Code unterscheidet zwischen echten, Test- oder synthetischen Daten.
