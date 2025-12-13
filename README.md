# CSE 6521: Multimodal Data Management with Large Language Models

This guide will walk you through setting up the dataset, building the DriveMMKG, and performing natural language queries using our agentic framework.

### 📋 Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.8+
- Memgraph (install via `curl -sSf "https://install.memgraph.com" | sh`)
- One agent interface:
	- Lemonade Server + Tiny Agents CLI (for local model hosting), or
	- Claude Desktop (if you prefer Claude; optional)

Install the required Python packages using pip (the Neo4j Python driver is used to talk to Memgraph over Bolt):
```
pip install numpy tqdm neo4j
```

### 🚀 Step-by-Step Reproduction Guide
Follow these steps to set up the environment and run the experiments.

#### Step 1: Download Dataset
Our experiments utilize the HRI Driving Dataset (HDD). Please download the dataset from the official website:

- Link: https://usa.honda-ri.com/hdd

*Note: After downloading the dataset, open the `config.py` file and set the `dataset_root` variable to the path of your HDD data's root directory.*

#### Step 2: Prepare Cause Annotations
For our demonstration, we use the validation set of the HDD. The first step is to parse the raw dataset and extract the cause annotations for each driving session into a structured format.

We provide a script to automate this process. Run the following command from the project's root directory:
```
python cause_annotations.py
```

This script, `cause_annotations.py`, will process the raw HDD files and save the cause annotations as `.npy` files in the specified output directory (`hdd_data/cause`).

#### Step 3: Build and Ingest DriveMMKG into Memgraph
Process the prepared annotations and raw driving data to construct DriveMMKG and ingest it into a Memgraph instance.

1. Install Memgraph:
```
curl -sSf "https://install.memgraph.com" | sh
```

2. Start Memgraph (ensure Bolt is exposed on 7687; `--schema-info-enabled=True` helps downstream schema introspection):
```
docker run -p 7687:7687 memgraph/memgraph-mage --schema-info-enabled=True
```

3. Set your Memgraph connection settings (defaults assume no auth):
```
export MEMGRAPH_URI="bolt://localhost:7687"
export MEMGRAPH_USERNAME=""
export MEMGRAPH_PASSWORD=""
```

4. Run the provided ingestion script:
```
python build_drivemmkg.py
```

The `build_drivemmkg.py` script connects to Memgraph over Bolt, builds the graph structure as defined in our paper, and populates it with the HDD validation set data.

#### Step 4: Configure the Agentic Interface
To enable an LLM agent to communicate with your DriveMMKG instance in Memgraph, configure an MCP client. Two options:

- Memgraph MCP reference (general MCP setup): https://github.com/memgraph/ai-toolkit/tree/main/integrations/mcp-memgraph
- Lemonade Server + Tiny Agents (local model hosting): see Step 5 below.

#### Step 5: Run Natural Language Queries with Lemonade Server (Tiny Agents)
Lemonade Server lets you host the model locally and connect MCP servers (like the provided Memgraph MCP server) via Tiny Agents. Based on the [Lemonade Server MCP guide](https://huggingface.co/learn/mcp-course/en/unit2/lemonade-server):

1. Install Lemonade Server (Windows/Linux installers at https://github.com/lemonade-sdk/lemonade/releases) and launch it. The UI runs at `http://localhost:8000`.
2. Add a model in Lemonade (e.g., `Qwen3-8B-GGUF`) via **Model Management**. Ensure the API endpoint is available at `http://localhost:8000/api/`.
3. In this repo, create an `agent.json` to point Tiny Agents to Lemonade for inference and to the Memgraph MCP server over stdio:
```
{
	"model": "Qwen3-8B-GGUF",
	"endpointUrl": "http://localhost:8000/api/",
	"servers": [
		{
			"type": "stdio",
			"command": "python",
			"args": ["path/to/mcp-memgraph/server.py"]
		}
	]
}
```
4. Run the agent locally (Tiny Agents CLI):
```
tiny-agents run agent.json
```
5. Query the Memgraph graph via natural language, for example:
```
Find left turn scenes where maximum speed exceeds 25mph. Find scenes from the Memgraph graph.
```

If you prefer Claude Desktop instead of Lemonade, follow the general MCP instructions above and point Claude to the Memgraph MCP server.
