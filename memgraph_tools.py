"""
Memgraph tools for ollama tool calling.

This module wraps memgraph_toolbox functions as ollama-compatible tools
for interacting with Memgraph database.
"""

import os

from memgraph_toolbox.api.memgraph import Memgraph
from memgraph_toolbox.tools.betweenness_centrality import BetweennessCentralityTool
from memgraph_toolbox.tools.config import ShowConfigTool
from memgraph_toolbox.tools.constraint import ShowConstraintInfoTool
from memgraph_toolbox.tools.cypher import CypherTool
from memgraph_toolbox.tools.index import ShowIndexInfoTool
from memgraph_toolbox.tools.node_neighborhood import NodeNeighborhoodTool
from memgraph_toolbox.tools.page_rank import PageRankTool
from memgraph_toolbox.tools.schema import ShowSchemaInfoTool
from memgraph_toolbox.tools.storage import ShowStorageInfoTool
from memgraph_toolbox.tools.trigger import ShowTriggersTool

# --- Memgraph Configuration ---
URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
USERNAME = os.getenv("MEMGRAPH_USERNAME", "")
PASSWORD = os.getenv("MEMGRAPH_PASSWORD", "")
DATABASE = os.getenv("MEMGRAPH_DATABASE", "memgraph")

# Initialize Memgraph client
DB = Memgraph(url=URI, username=USERNAME, password=PASSWORD, database=DATABASE)


def run_cypher_query(query: str) -> str:
    """Execute a Cypher query on the Memgraph database."""
    try:
        result = CypherTool(db=DB).call({"query": query})
        if not result:
            return "Query executed successfully but returned no results."

        output_lines = [f"Found {len(result)} result(s):"]
        for index, record in enumerate(result[:50], start=1):
            formatted = {}
            for key, value in record.items():
                if isinstance(value, list) and len(value) > 10:
                    formatted[key] = f"[{len(value)} items]"
                elif isinstance(value, str) and len(value) > 100:
                    formatted[key] = value[:100] + "..."
                else:
                    formatted[key] = value
            output_lines.append(f"  {index}. {formatted}")

        if len(result) > 50:
            output_lines.append(f"  ... and {len(result) - 50} more results")

        return "\n".join(output_lines)
    except Exception as error:
        return f"Error executing query: {str(error)}"


def get_schema_info() -> str:
    """Get schema information of the Memgraph database."""
    try:
        result = ShowSchemaInfoTool(db=DB).call({})
        if not result:
            return "No schema information available."
        return str(result)
    except Exception as error:
        return f"Error fetching schema: {str(error)}"


def get_configuration() -> str:
    """Get Memgraph server configuration information."""
    try:
        result = ShowConfigTool(db=DB).call({})
        if not result:
            return "No configuration information available."
        return str(result)
    except Exception as error:
        return f"Error fetching configuration: {str(error)}"


def get_index_info() -> str:
    """Get information about indexes in the Memgraph database."""
    try:
        result = ShowIndexInfoTool(db=DB).call({})
        if not result:
            return "No indexes found."
        return str(result)
    except Exception as error:
        return f"Error fetching index info: {str(error)}"


def get_constraint_info() -> str:
    """Get information about constraints in the Memgraph database."""
    try:
        result = ShowConstraintInfoTool(db=DB).call({})
        if not result:
            return "No constraints found."
        return str(result)
    except Exception as error:
        return f"Error fetching constraint info: {str(error)}"


def get_storage_info() -> str:
    """Get storage information of the Memgraph database."""
    try:
        result = ShowStorageInfoTool(db=DB).call({})
        if not result:
            return "No storage information available."
        return str(result)
    except Exception as error:
        return f"Error fetching storage info: {str(error)}"


def get_triggers_info() -> str:
    """Get information about triggers in the Memgraph database."""
    try:
        result = ShowTriggersTool(db=DB).call({})
        if not result:
            return "No triggers found."
        return str(result)
    except Exception as error:
        return f"Error fetching triggers info: {str(error)}"


def get_betweenness_centrality() -> str:
    """Calculate betweenness centrality for nodes in the graph."""
    try:
        result = BetweennessCentralityTool(db=DB).call({})
        if not result:
            return "No betweenness centrality results."
        return str(result)
    except Exception as error:
        return f"Error calculating betweenness centrality: {str(error)}"


def get_page_rank() -> str:
    """Calculate PageRank scores for nodes in the graph."""
    try:
        result = PageRankTool(db=DB).call({})
        if not result:
            return "No PageRank results."
        return str(result)
    except Exception as error:
        return f"Error calculating PageRank: {str(error)}"


def get_node_neighborhood(node_id: str, max_distance: int = 1, limit: int = 100) -> str:
    """Find nodes within a specified distance from a given node."""
    try:
        result = NodeNeighborhoodTool(db=DB).call(
            {"node_id": node_id, "max_distance": max_distance, "limit": limit}
        )
        if not result:
            return "No neighbors found."
        return str(result)
    except Exception as error:
        return f"Error finding node neighborhood: {str(error)}"


AVAILABLE_TOOLS = {
    "run_cypher_query": run_cypher_query,
    "get_schema_info": get_schema_info,
    "get_configuration": get_configuration,
    "get_index_info": get_index_info,
    "get_constraint_info": get_constraint_info,
    "get_storage_info": get_storage_info,
    "get_triggers_info": get_triggers_info,
    "get_betweenness_centrality": get_betweenness_centrality,
    "get_page_rank": get_page_rank,
    "get_node_neighborhood": get_node_neighborhood,
}

TOOL_FUNCTIONS = [
    run_cypher_query,
    get_schema_info,
    get_configuration,
    get_index_info,
    get_constraint_info,
    get_storage_info,
    get_triggers_info,
    get_betweenness_centrality,
    get_page_rank,
    get_node_neighborhood,
]
