# DriveMMKG: Context Engineering for Multimodal Query Agents over Driving Data

This repository contains reproducible code for DriveMMKG, including building a driving-scene knowledge graph in Memgraph and querying it with an LLM agent.

## Included Files
- `build_kg.py`: Loads BDD-X-style data into Memgraph as a knowledge graph
- `do_query_kg.py`: Runs query benchmarks and saves predicted trip IDs
- `memgraph_tools.py`: Tool wrappers used by `do_query_kg.py`

## 1) Install and Run Memgraph (Docker)
Run Memgraph locally with:

```bash
docker run -p 7687:7687 -p 7444:7444 --name memgraph memgraph/memgraph-mage
```

If you restart later:

```bash
docker start memgraph
```

## 2) Create Conda Environment
Use a dedicated conda environment named `drive-mmkg`:

```bash
conda create -n drive-mmkg python=3.11 -y
conda activate drive-mmkg
pip install -r requirements.txt
```

## 3) Install and Run Ollama
Install Ollama first (Linux/macOS/Windows):

https://ollama.com/download

Download the default model used in this repo:

```bash
ollama pull gpt-oss:120b
```

Run the Ollama server in a separate terminal and keep it running:

```bash
ollama serve
```

Then use another terminal for `build_kg.py` and `do_query_kg.py` commands.

## 4) Download Dataset Files
You can download the data from [Google Drive](https://drive.google.com/file/d/1V16vDW0bANHlMxSiL7I1DX0Hr8ga9aVq/view?usp=sharing) and place files under `data/`.

Original dataset source:
- BDD-X dataset repository: https://github.com/JinkyuKimUCB/BDD-X-dataset

License/compliance note:
- This project uses data derived from BDD-X.
- Follow the original BDD-X license terms when using or redistributing any derived dataset.
- Keep the original copyright/disclaimer notice in your redistribution package.
- Commercial use may require a separate license from the original rights holder.
- For this repository, dataset redistribution is intended for educational, research, and not-for-profit use.
- See `THIRD_PARTY_NOTICES.md` for the copied third-party notice text.

Expected layout:

```text
data/
  description_sensor/
    *.json
  textual_kg/
    *.json
  id_mapping.json
  queries_gemini.json
```

Notes:
- `data/description_sensor`, `data/textual_kg`, and `data/id_mapping.json` are required by `build_kg.py`.
- `data/queries_gemini.json` is required by `do_query_kg.py` (default path).
- `queries_gemini.json` can include `matching_videos` for recall/precision reporting.

## 5) Memgraph Connection Configuration (Optional)
Set these only if your Memgraph connection differs from defaults:

- `MEMGRAPH_URI` (default: `bolt://localhost:7687`)
- `MEMGRAPH_USERNAME` (default: empty string)
- `MEMGRAPH_PASSWORD` (default: empty string)
- `MEMGRAPH_DATABASE` (default: `memgraph`)

Example:

```bash
export MEMGRAPH_URI="bolt://localhost:7687"
export MEMGRAPH_USERNAME=""
export MEMGRAPH_PASSWORD=""
export MEMGRAPH_DATABASE="memgraph"
```

## 6) Build Knowledge Graph (Run First)
Load graph data into Memgraph before running query evaluation:

```bash
python build_kg.py --base-dir data
```

## 7) Run Query Evaluation
Then run KG query execution:

```bash
python do_query_kg.py \
  --model gpt-oss:120b \
  --queries data/queries_gemini.json \
  --output-dir pred \
  --verbose
```

## Output
- Prediction results are saved to `pred/results_kg_<model>.json`.

## Dataset Attribution
If you redistribute a modified dataset, include:
- The original source link: https://github.com/JinkyuKimUCB/BDD-X-dataset
- A statement that your files are modified/derived data
- The original copyright and disclaimer notice
