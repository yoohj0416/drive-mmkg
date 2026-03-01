"""
Driving Scene Query Execution Script

This script processes driving-scene queries,
executes them using the retrieval agent,
and saves predicted trip IDs to JSON.
"""

import argparse
import json
import os
import re
from datetime import datetime

from ollama import ChatResponse, chat

from memgraph_tools import AVAILABLE_TOOLS, TOOL_FUNCTIONS


# --- Configuration ---
DEFAULT_MODEL = "gpt-oss:120b"
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QUERIES_PATH = os.path.join(REPO_ROOT, "data", "queries_gemini.json")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "pred")


# --- System Prompt ---
SYSTEM_PROMPT = """You are an expert driving scene retrieval assistant. Your task is to help users find specific driving scenes from a knowledge graph stored in Memgraph database.

## Available Tools

You have access to the following tools:

### Database Exploration Tools
- `get_schema_info()`: Get the graph schema (node labels, relationship types, properties). Use this FIRST to understand the database structure.
- `get_configuration()`: Get Memgraph server configuration settings.
- `get_index_info()`: Get information about existing indexes.
- `get_constraint_info()`: Get information about constraints.
- `get_storage_info()`: Get storage usage metrics.
- `get_triggers_info()`: List all database triggers.

### Query Tool
- `run_cypher_query(query)`: Execute any Cypher query against the database. This is your primary tool for data retrieval.

### Graph Analysis Tools
- `get_betweenness_centrality()`: Calculate betweenness centrality for nodes.
- `get_page_rank()`: Calculate PageRank scores for nodes.
- `get_node_neighborhood(node_id, max_distance, limit)`: Find nodes within a specified distance from a given node.

## Your Workflow

You MUST follow this systematic approach for every query:

### Step 1: Understand the Knowledge Graph Structure
Before writing any search query, use `get_schema_info()` to understand:
- What node types exist (labels)
- What relationships connect them
- What properties each node type has

### Step 2: Explore the Data
Use `run_cypher_query()` to run exploratory queries:
- Check what action types exist: `MATCH (a:Action) RETURN DISTINCT a.actionType, count(*) ORDER BY count(*) DESC`
- Check what cause types exist: `MATCH (c:Cause) RETURN DISTINCT c.causeType, count(*) ORDER BY count(*) DESC`
- Check sensor types: `MATCH (s:Sensor) RETURN DISTINCT s.name, s.unit LIMIT 10`
- Understand data patterns before constructing complex queries

### Step 3: Construct the Final Query
Based on your exploration, write a precise Cypher query to find the requested scenes.

## Matching Strategy - FLEXIBLE MATCHING

When searching for trips/scenes based on user queries, use a **flexible matching approach**:

1. **Trip-level matching**: If a query mentions multiple conditions (e.g., \"left turn then stop\"), find Trips where ALL conditions appear somewhere within the trip, in the correct ORDER, but NOT necessarily in immediately consecutive scenes.
   - Example: \"left turn followed by stop\" → Trip is valid if scene 2 has \"left turn\" and scene 5 has \"stop\" (order preserved, but not consecutive)

2. **Order matters, strict adjacency doesn't**: When the query implies a sequence (e.g., \"A then B\", \"A followed by B\"), ensure A appears before B within the same trip, but allow other scenes in between.

3. **Scene-level matching**: If the query is about a single action/condition (e.g., \"left turn with speed > 25 mph\"), match individual scenes that satisfy all conditions.

4. **Use sceneNumber for ordering**: Scenes have `sceneNumber` property. Use this to verify order:
   ```cypher
   MATCH (t:Trip)-[:HAS_SCENE]->(s1:Scene)-[:HAS_ACTION]->(a1:Action),
         (t)-[:HAS_SCENE]->(s2:Scene)-[:HAS_ACTION]->(a2:Action)
   WHERE a1.actionType CONTAINS \"left\" AND a2.actionType CONTAINS \"stop\"
     AND s1.sceneNumber < s2.sceneNumber
   RETURN DISTINCT t.id
   ```

5. **Partial text matching**: Use `CONTAINS` or `toLower()` for flexible text matching rather than exact matches.

## Expected Knowledge Graph Structure (for driving scenes)

The graph typically contains:
- **Trip**: A video clip (properties: id, videoName)
  - **Trip ID format**: Trip IDs are 4-digit strings like \"0001\", \"0002\", ..., \"0523\"
- **Scene**: A segment within a trip (properties: sceneId, sceneNumber, startTimeInSec, endTimeInSec)
- **Action**: Driving action performed (properties: actionId, actionType)
- **Cause**: Reason for the action (properties: causeId, causeType)
- **Sensor**: Time-series sensor data (properties: id, name, unit, values[], description)

Common Relationships:
- (Trip)-[:HAS_SCENE]->(Scene)
- (Scene)-[:HAS_ACTION]->(Action)
- (Scene)-[:HAS_CAUSE]->(Cause)
- (Action)-[:CAUSED_BY]->(Cause)
- (Scene)-[:HAS_SENSOR]->(Sensor)

## Sensor Data Details
- **Speed**: Unit is m/s. To convert to mph, multiply by 2.237 (1 m/s ≈ 2.237 mph)
- **Gyro Yaw**: Unit is rad/s. Negative = right turn, Positive = left turn
- **Accelerator Pedal**: Percentage (0-100%)
- **Brake Pedal**: Pressure in kPa
- **Turn Signal**: 0=off, 1=left, 2=right

## Cypher Tips

1. **IMPORTANT - String Quotes**: Use DOUBLE QUOTES (") for string literals in Cypher, NOT single quotes.
   - Correct: `WHERE a.actionType CONTAINS \"left\"`
   - Wrong: `WHERE a.actionType CONTAINS 'left'`
2. **Sensor array processing**: Sensor values are stored as arrays. Use Cypher functions like:
   - `reduce(max = 0, x IN sensor.values | CASE WHEN x > max THEN x ELSE max END)` for max value
   - `reduce(sum = 0.0, x IN sensor.values | sum + x) / size(sensor.values)` for average
3. **Unit conversions**: Always convert units when needed (e.g., m/s to mph: multiply by 2.237)
4. **Case sensitivity**: Use `toLower()` for case-insensitive matching if needed.
5. **Pattern matching**: Use `CONTAINS` or `=~` for partial string matching.

## IMPORTANT: Final Response Format

After finding results, you MUST provide the list of matching Trip IDs.
Trip IDs are 4-digit strings like \"0001\", \"0042\", \"0523\".
Return ONLY the Trip IDs that match the query criteria."""


def format_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%H:%M:%S")


def get_output_path(model_name: str, output_dir: str) -> str:
    """Get the output file path based on model name."""
    safe_model_name = model_name.replace(":", "-").replace("/", "-")
    return os.path.join(output_dir, f"results_kg_{safe_model_name}.json")


def load_existing_results(model_name: str, output_dir: str) -> dict:
    """Load existing results if file exists."""
    output_path = get_output_path(model_name, output_dir)
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {"model": model_name, "results": {}}


def save_results(results: dict, model_name: str, output_dir: str):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = get_output_path(model_name, output_dir)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def extract_trip_ids_from_response(content: str) -> list[str]:
    """
    Extract trip IDs from model response text.
    Trip IDs are 4-digit strings like "0001", "0042", "0523".
    """
    if not content:
        return []

    pattern = r"\b(0[0-5][0-9]{2})\b"
    matches = re.findall(pattern, content)

    seen = set()
    unique_ids = []
    for match in matches:
        if match not in seen:
            seen.add(match)
            unique_ids.append(match)

    return unique_ids


def run_query_agent(
    user_query: str,
    query_id: str,
    model_name: str,
    max_iterations: int = 50,
    verbose: bool = False,
) -> tuple[list[str], dict]:
    """
    Run the agent loop for a single query and return predicted trip IDs and usage metrics.

    Returns:
        Tuple of (predicted trip IDs list, usage dict)
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"[{format_timestamp()}] Processing Query: {query_id}")
        print(f"Query: {user_query[:100]}...")
        print(f"{'=' * 60}")

    enhanced_query = f"""{user_query}

After finding the matching trips, please list all the Trip IDs that match this query.
Remember: Trip IDs are 4-digit strings (e.g., "0001", "0042", "0523").
List only the Trip IDs in your final response."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": enhanced_query},
    ]

    iteration = 0
    consecutive_errors = 0
    max_consecutive_errors = 3
    final_content = ""

    total_prompt_tokens = 0
    total_eval_tokens = 0
    total_duration_ns = 0
    prompt_eval_duration_ns = 0
    eval_duration_ns = 0

    while iteration < max_iterations:
        iteration += 1
        if verbose:
            print(f"\n[{format_timestamp()}] Iteration {iteration}/{max_iterations}")

        try:
            response: ChatResponse = chat(
                model=model_name,
                messages=messages,
                tools=TOOL_FUNCTIONS,
                options={"temperature": 0},
            )
            consecutive_errors = 0
        except Exception as error:
            consecutive_errors += 1
            if verbose:
                print(
                    f"[ERROR] Failed to call model "
                    f"(attempt {consecutive_errors}/{max_consecutive_errors}): {error}"
                )

            if consecutive_errors >= max_consecutive_errors:
                if verbose:
                    print("[ERROR] Max consecutive errors reached. Stopping.")
                break

            error_feedback = (
                "The previous tool call failed due to JSON parsing error. "
                "Please ensure your Cypher query uses double quotes for strings instead of single quotes. "
                "Try again with a corrected query."
            )
            messages.append({"role": "user", "content": error_feedback})
            continue

        message = response.message
        thinking = getattr(message, "thinking", None)
        content = message.content
        tool_calls = message.tool_calls

        total_prompt_tokens += getattr(response, "prompt_eval_count", 0) or 0
        total_eval_tokens += getattr(response, "eval_count", 0) or 0
        total_duration_ns += getattr(response, "total_duration", 0) or 0
        prompt_eval_duration_ns += getattr(response, "prompt_eval_duration", 0) or 0
        eval_duration_ns += getattr(response, "eval_duration", 0) or 0

        if verbose and thinking:
            if len(thinking) > 100:
                print(f"  🧠 Thinking: {thinking[:100]}...")
            else:
                print(f"  🧠 Thinking: {thinking}")

        if content:
            if verbose:
                if len(content) > 200:
                    print(f"  💬 Response: {content[:200]}...")
                else:
                    print(f"  💬 Response: {content}")
            final_content = content

        assistant_message = {"role": "assistant", "content": content or ""}
        if thinking:
            assistant_message["thinking"] = thinking
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)

        if tool_calls:
            if verbose:
                print(f"  🔧 Tool calls: {len(tool_calls)}")

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments

                if verbose:
                    print(f"    - {tool_name}({str(tool_args)[:50]}...)")

                if tool_name in AVAILABLE_TOOLS:
                    try:
                        result = AVAILABLE_TOOLS[tool_name](**tool_args)
                        if verbose:
                            result_preview = result[:200] + "..." if len(result) > 200 else result
                            print(f"      ✅ {result_preview}")
                    except Exception as error:
                        result = f"Error executing tool: {str(error)}"
                        if verbose:
                            print(f"      ❌ {error}")
                else:
                    result = f"Unknown tool: {tool_name}"
                    if verbose:
                        print("      ❌ Unknown tool")

                messages.append(
                    {"role": "tool", "tool_name": tool_name, "content": result}
                )
        else:
            if verbose:
                print(f"\n[{format_timestamp()}] ✅ Agent completed")
            break

    trip_ids = extract_trip_ids_from_response(final_content)
    if verbose:
        print(f"\n[{format_timestamp()}] Extracted Trip IDs: {trip_ids}")

    usage = {
        "total_prompt_tokens": total_prompt_tokens,
        "total_eval_tokens": total_eval_tokens,
        "total_tokens": total_prompt_tokens + total_eval_tokens,
        "total_duration_ms": total_duration_ns / 1_000_000,
        "prompt_eval_duration_ms": prompt_eval_duration_ns / 1_000_000,
        "eval_duration_ms": eval_duration_ns / 1_000_000,
        "iterations": iteration,
    }

    if verbose:
        print(f"\n[{format_timestamp()}] Usage Metrics:")
        print(
            f"  Total tokens: {usage['total_tokens']} "
            f"(prompt: {usage['total_prompt_tokens']}, eval: {usage['total_eval_tokens']})"
        )
        print(f"  Total duration: {usage['total_duration_ms']:.1f} ms")
        print(f"  Iterations: {usage['iterations']}")

    return trip_ids, usage


def main():
    """Main function to process all queries."""
    parser = argparse.ArgumentParser(description="Execute driving scene queries")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-q",
        "--queries",
        type=str,
        default=DEFAULT_QUERIES_PATH,
        help=f"Path to query JSON file (default: {DEFAULT_QUERIES_PATH})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save prediction results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum agent iterations per query (default: 50)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed agent progress (thinking, tool calls, etc.)",
    )
    args = parser.parse_args()

    model_name = args.model
    queries_path = os.path.abspath(args.queries)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(queries_path):
        raise FileNotFoundError(
            "Queries file not found: "
            f"{queries_path}\n"
            "Please place your query file there or pass --queries /path/to/queries.json"
        )

    print("\n" + "=" * 80)
    print("  DRIVING SCENE QUERY EXECUTION")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Queries file: {queries_path}")
    print(f"Output: {get_output_path(model_name, output_dir)}")
    print(f"Verbose: {args.verbose}")
    print("=" * 80 + "\n")

    with open(queries_path, "r", encoding="utf-8") as file:
        queries_data = json.load(file)

    queries = queries_data.get("queries", [])
    print(f"Total queries: {len(queries)}")

    results = load_existing_results(model_name, output_dir)

    already_processed = len(results.get("results", {}))
    print(f"Already processed: {already_processed}")

    for index, query_item in enumerate(queries):
        query_id = query_item.get("id", f"q_{index:04d}")
        query_text = query_item.get("query", "")
        level = query_item.get("level", "")

        if query_id in results.get("results", {}):
            print(f"\n[{index + 1}/{len(queries)}] Skipping {query_id} (already processed)")
            continue

        print(f"\n[{index + 1}/{len(queries)}] Processing {query_id} (Level: {level})")

        predicted_trip_ids, usage = run_query_agent(
            query_text,
            query_id,
            model_name,
            max_iterations=args.max_iterations,
            verbose=args.verbose,
        )

        results["results"][query_id] = {
            "query": query_text,
            "level": level,
            "predicted_trip_ids": predicted_trip_ids,
            "usage": usage,
            "timestamp": datetime.now().isoformat(),
        }

        save_results(results, model_name, output_dir)
        print(f"[{format_timestamp()}] Saved results for {query_id}")

        gt_trip_ids = query_item.get("matching_videos", [])
        predicted_set = set(predicted_trip_ids)
        gt_set = set(gt_trip_ids)

        if gt_set:
            recall = len(predicted_set & gt_set) / len(gt_set) * 100
            precision = len(predicted_set & gt_set) / len(predicted_set) * 100 if predicted_set else 0
            print(f"  Ground truth: {len(gt_set)} trips, Predicted: {len(predicted_set)} trips")
            print(f"  Recall: {recall:.1f}%, Precision: {precision:.1f}%")

    print("\n" + "=" * 80)
    print("  EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Total processed: {len(results.get('results', {}))}")
    print(f"Results saved to: {get_output_path(model_name, output_dir)}")


if __name__ == "__main__":
    main()
