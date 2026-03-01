# Data Directory Layout

Place downloaded/extracted dataset files under this directory.

Original source:
- BDD-X dataset: https://github.com/JinkyuKimUCB/BDD-X-dataset

Compliance reminder:
- This folder may contain derived data from BDD-X.
- Keep original attribution and copyright/disclaimer notice when redistributing.
- Use for educational/research/not-for-profit purposes unless you have separate commercial permission.

Expected structure:

```text
data/
  description_sensor/
    *.json
  textual_kg/
    *.json
  id_mapping.json
  queries_gemini.json
```
- `data/description_sensor`, `data/textual_kg`, and `data/id_mapping.json` are used by `build_kg.py`.
- `data/description_sensor`, `data/textual_kg`, and `data/id_mapping.json` are used by `build_kg.py`.
- `data/queries_gemini.json` is used by `do_query_kg.py` by default.
