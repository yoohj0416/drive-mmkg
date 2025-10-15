import os
import numpy as np
from tqdm import tqdm
from neo4j import GraphDatabase, Driver, Session

from cfg import GeneralCfg

# --- Environment and Database Configuration ---
URI = os.getenv("NEO4J_URI")
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD"))

# --- Helper Functions for Neo4j Operations ---

def create_scene_node(session: Session, scene_id: str, scene_number: int, start_time: float, end_time: float):
    """Creates a Scene node in the graph."""
    query = (
        "CREATE (s:Scene {sceneId: $scene_id, sceneNumber: $scene_number, "
        "startTimeInSec: $start_time, endTimeInSec: $end_time})"
    )
    session.run(query, scene_id=scene_id, scene_number=scene_number, start_time=start_time, end_time=end_time)

def create_action_node(session: Session, scene_id: str, action_id: str, label: str, action_type: str):
    """Creates an action node and links it to a scene."""
    query = (
        f"MATCH (s:Scene {{sceneId: $scene_id}}) "
        f"CREATE (a:{label} {{actionId: $action_id, actionType: $action_type}}) "
        f"CREATE (s)-[:HAS_ACTION]->(a)"
    )
    session.run(query, scene_id=scene_id, action_id=action_id, action_type=action_type)

def create_cause_node(session: Session, scene_id: str, cause_id: str, cause_type: str, stimulus_action_id: str):
    """Creates a Cause node and links it to an action and its scene."""
    session.run("CREATE (c:Cause {causeId: $cause_id, causeType: $cause_type})", cause_id=cause_id, cause_type=cause_type)
    session.run(
        "MATCH (a:StimulusDrivenAction {actionId: $action_id}), (c:Cause {causeId: $cause_id}) CREATE (a)-[:CAUSED_BY]->(c)",
        action_id=stimulus_action_id, cause_id=cause_id
    )
    session.run(
        "MATCH (s:Scene {sceneId: $scene_id}), (c:Cause {causeId: $cause_id}) CREATE (s)-[:HAS_CAUSE]->(c)",
        scene_id=scene_id, cause_id=cause_id
    )

def create_sensor_node(session: Session, scene_id: str, sensor_key: str, data: dict):
    """Creates a Sensor node with metadata and data, then links it to a scene."""
    query = (
        "MATCH (s:Scene {sceneId: $scene_id}) "
        "CREATE (sensor:Sensor $data) "
        "CREATE (s)-[:HAS_SENSOR]->(sensor)"
    )
    session.run(query, scene_id=scene_id, data=data)

def link_to_previous_scene(session: Session, current_scene_id: str, prev_scene_id: str):
    """Creates directional relationships between two scenes."""
    query = (
        "MATCH (s1:Scene {sceneId: $prev_scene_id}), (s2:Scene {sceneId: $current_scene_id}) "
        "CREATE (s1)-[:NEXT_SCENE]->(s2), (s2)-[:PREVIOUS_SCENE]->(s1)"
    )
    session.run(query, prev_scene_id=prev_scene_id, current_scene_id=current_scene_id)

# --- Helper Function for Data Processing ---

def segment_actions_into_scenes(goal_action: np.ndarray, cause_action: np.ndarray, target_map: dict, cause_map: dict) -> list:
    """Segments action arrays into scenes based on changes in either action."""
    scenes = []
    if len(goal_action) == 0: return scenes
    start_time, current_goal_val, current_cause_val = 0.0, goal_action[0], cause_action[0]
    for i in range(1, len(goal_action)):
        if goal_action[i] != current_goal_val or cause_action[i] != current_cause_val:
            end_time = float(i)
            scenes.append({
                "startTimeInSec": start_time, "endTimeInSec": end_time,
                "goalAction": target_map[current_goal_val].title(),
                "stimulusDrivenAction": cause_map[current_cause_val].title() if current_cause_val != 0 else None,
            })
            start_time, current_goal_val, current_cause_val = end_time, goal_action[i], cause_action[i]
    scenes.append({
        "startTimeInSec": start_time, "endTimeInSec": float(len(goal_action)),
        "goalAction": target_map[current_goal_val].title(),
        "stimulusDrivenAction": cause_map[current_cause_val].title() if current_cause_val != 0 else None,
    })
    return scenes

# --- Session Processing Function ---

def process_and_load_session(driver: Driver, cfg: GeneralCfg, session_name: str):
    """Loads all data for a single session, processes it, and populates the Neo4j graph."""
    print(f"\nProcessing session: {session_name}")
    
    # --- 1. Load All Data Types ---
    try:
        goal_action = np.load(os.path.join(cfg.target_root, f"{session_name}.npy"))
        cause = np.load(os.path.join(cfg.cause_root, f"{session_name}.npy"))
        sensor_data = np.load(os.path.join(cfg.sensor_root, f"{session_name}.npy"))
    except FileNotFoundError as e:
        print(f"Error: Could not find data files for session '{session_name}'. Skipping. Details: {e}")
        return

    # --- 2. Process Actions and Segment Scenes ---
    scenes = segment_actions_into_scenes(goal_action[::cfg.sample_rate], cause[::cfg.sample_rate], cfg.target_int_to_str, cfg.cause_int_to_str)
    print(f"Segmented session '{session_name}' into {len(scenes)} scenes.")

    # --- 3. Extract Full Sensor Vectors ---
    sensor_vectors = {}
    for key, meta in cfg.sensor_metadata.items():
        sensor_vectors[key] = sensor_data[:, meta["col"]]

    # --- 4. Populate Neo4j Graph ---
    trip_id = session_name
    with driver.session(database="neo4j") as session:
        # Clear existing data for this trip_id
        session.run("MATCH (t:Trip {id: $trip_id}) DETACH DELETE t", trip_id=trip_id)
        print(f"Cleared existing data for Trip ID: {trip_id}")
        session.run("CREATE (t:Trip {id: $trip_id})", trip_id=trip_id)

    action_counter, cause_counter = 0, 0
    for i, scene_data in enumerate(scenes):
        scene_id = f"{trip_id}_SCENE_{i}"
        
        with driver.session(database="neo4j") as session:
            # Create Scene node and link to Trip and previous Scene
            create_scene_node(session, scene_id, i, scene_data["startTimeInSec"], scene_data["endTimeInSec"])
            session.run("MATCH (t:Trip {id: $trip_id}), (s:Scene {sceneId: $scene_id}) CREATE (t)-[:PART_OF]->(s)", trip_id=trip_id, scene_id=scene_id)
            if i > 0: link_to_previous_scene(session, scene_id, f"{trip_id}_SCENE_{i-1}")

            # Create and link Sensor nodes
            start_idx = int(scene_data["startTimeInSec"] * cfg.sample_rate)
            end_idx = int(scene_data["endTimeInSec"] * cfg.sample_rate)

            for key, vector in sensor_vectors.items():
                meta = cfg.sensor_metadata[key]
                sensor_node_data = {
                    "id": f"{scene_id}_{key}",
                    "name": meta["name"],
                    "unit": meta["unit"],
                    "sampleRateHz": cfg.sample_rate,
                    "description": meta["desc"],
                    "values": vector[start_idx:end_idx].tolist()
                }
                create_sensor_node(session, scene_id, key, sensor_node_data)
            
            # Create and link Action/Cause nodes
            goal_action_type = scene_data["goalAction"]
            stimulus_cause_type = scene_data.get("stimulusDrivenAction")
            if goal_action_type != "Background":
                action_id = f"{trip_id}_ACTION_{action_counter}"
                # create_action_node(session, scene_id, action_id, "GoalOrientedAction", goal_action_type)
                create_action_node(session, scene_id, action_id, "Action", goal_action_type)
                action_counter += 1
            if stimulus_cause_type:
                action_id = f"{trip_id}_ACTION_{action_counter}"
                cause_id = f"{trip_id}_CAUSE_{cause_counter}"
                # create_action_node(session, scene_id, action_id, "StimulusDrivenAction", "Stop")
                create_action_node(session, scene_id, action_id, "Action", "Stop")
                create_cause_node(session, scene_id, cause_id, stimulus_cause_type, action_id)
                action_counter += 1; cause_counter += 1
            if goal_action_type == "Background" and not stimulus_cause_type:
                action_id = f"{trip_id}_ACTION_{action_counter}"
                # create_action_node(session, scene_id, action_id, "GoalOrientedAction", "Go Straight")
                create_action_node(session, scene_id, action_id, "Action", "Go Straight")
                action_counter += 1

# --- Main Execution Logic ---

def main():
    """Iterates through all sessions, processes them, and creates a graph in Neo4j."""
    cfg = GeneralCfg()
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        print("Successfully connected to Neo4j.")
        for session_name in tqdm(cfg.validation_session_set, desc="Processing all sessions"):
            process_and_load_session(driver, cfg, session_name)
    print("\nGraph creation for all sessions is complete.")

if __name__ == "__main__":
    # To run this script, create dummy data files:
    # os.makedirs('./target_data', exist_ok=True)
    # os.makedirs('./cause_data', exist_ok=True)
    # os.makedirs('./sensor_data', exist_ok=True)
    # for session in ['session_1', 'session_2']:
    #     np.save(f'./target_data/{session}.npy', np.random.randint(0, 5, 3600))
    #     np.save(f'./cause_data/{session}.npy', np.random.randint(0, 3, 3600))
    #     np.save(f'./sensor_data/{session}.npy', np.random.rand(3600, 7) * 100)
    main()
