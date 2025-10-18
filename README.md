# Interpreting the Drive: Agent-driven Exploratory Multimodal Querying for Complex Driving Scenarios
This repository contains the official implementation and instructions to reproduce the experiments presented in our paper, "Interpreting the Drive: Agent-driven Exploratory Multimodal Querying for Complex Driving Scenarios"

This guide will walk you through setting up the dataset, building the DriveMMKG, and performing natural language queries using our agentic framework.

### 📋 Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.8+
- Claude Desktop

Install the required Python packages using pip:
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

#### Step 3: Build and Ingest DriveMMKG into Neo4j
Next, we will process the prepared annotations and raw driving data to construct our DriveMMKG and ingest it into a Neo4j graph database.

1. Log in to your [Neo4j Aura Console](https://console.neo4j.io/).

2. Create a new, empty database instance.

3. Set your Neo4j Connection URI and password as environment variables.
```
export NEO4J_URI="neo4j+s://your-aura-instance-uri.databases.neo4j.io"
export NEO4J_PASSWORD="your_password"
```

4. Run the provided ingestion script:
```
python build_drivemmkg.py
```

The `build_drivemmkg.py` script will connect to your local Neo4j instance, build the graph structure as defined in our paper, and populate it with the HDD validation set data.

#### Step 4: Configure the Agentic Interface
To enable the Claude LLM agent to communicate with your DriveMMKG instance in Neo4j, you will need to configure the connection using the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). For detailed instructions on setting up this protocol, please refer to the official Neo4j MCP repository:

- Official Guide: https://github.com/neo4j/mcp

Following the guide will allow the agent to interact with the database, enabling it to explore the schema and execute queries as described in our paper.

#### Step 5: Perform Natural Language Queries
You are now ready to perform complex driving scene retrieval using natural language.

1. Open your Claude Desktop application or interface.

2. Input a prompt instructing the agent to find scenes from the Neo4j graph. Use the following prompt format as an example:
```
Find left turn scenes where maximum speed exceeds 25mph. Find scenes from the Neo4j graph.
```
