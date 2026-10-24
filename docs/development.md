# Entwicklung

## Umgebung

```bash
uv sync --all-groups
```

| Gruppe | Inhalt |
|---|---|
| (Haupt) | Laufzeitabhängigkeiten |
| `dev` | `ruff`, `pytest`; installiert `uv sync` standardmäßig mit |
| `docs` | `sphinx`, `furo`, `myst-parser`, `sphinx-autoapi`, `sphinx-copybutton`, `sphinxcontrib-mermaid` |

## Prüfen

```bash
uv run ruff check .
uv run pytest
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
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

Die API-Referenz entsteht dabei aus den Docstrings; das Paket wird nicht
importiert. `-W` macht jede Warnung zum Fehler – ein toter Querverweis bricht
den Build ab. Live-Vorschau beim Schreiben, z. B. mit `sphinx-autobuild`
(nicht Teil der Abhängigkeiten):

```bash
uv run --with sphinx-autobuild sphinx-autobuild docs docs/_build/html
```

Die Seiten sind Markdown (MyST); Diagramme sind Mermaid-Blöcke.

## Kontinuierliche Integration

`.github/workflows/pull-request.yml` läuft bei jedem Pull Request und Push auf
`main`:

| Job | Schritte |
|---|---|
| Lint and package | `uv lock --check`, `ruff check .`, `uv build` |
| Test | `uv sync --locked`, `pytest tests` |
| Docs | `uv sync --locked --only-group docs`, `sphinx-build -W` |

## Konventionen

- Docstrings und Bezeichner auf Englisch, Texte der Oberfläche und des Berichts
  auf Deutsch.
- Docstrings im Sphinx-Feldstil (`:param x:`, `:return:`, `:raises E:`).
- Zeilenlänge 100 (`ruff`).
- Rechenwege importieren kein Qt; die Oberfläche rechnet nichts selbst.
- Jede Konstante in {py:mod}`docupilot.config` nennt ihre Herkunft.
- Eine Änderung an einem Modellprompt erhöht dessen `PROMPT_VERSION`.
- Kein Code unterscheidet zwischen echten, Test- oder synthetischen Daten.
